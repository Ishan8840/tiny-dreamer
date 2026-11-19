import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx, serialization

from collect import collect_episodes
from env import make_env
from model import Decoder, DynamicsModel, Encoder
from replay import ReplayBuffer
from train import (
    BATCH_SIZE, D_MIN, EPOCHS, LATENT_DIM, LEARNING_RATE, dynamics_loss,
    episode_partitions, reconstruction_loss, sample_training_inputs,
)


class WorldModel(nnx.Module):
    def __init__(self, rngs, channels, action_dim):
        self.tokenizer = nnx.Sequential(
            Encoder(rngs, channels, LATENT_DIM), Decoder(rngs, channels, LATENT_DIM)
        )
        self.dynamics = DynamicsModel(rngs, action_dim, latent_dim=LATENT_DIM)


def restore_tokenizer(tokenizer, path):
    state = nnx.state(tokenizer, nnx.Param)
    template = nnx.to_pure_dict(state)
    restored = serialization.from_bytes(template, path.read_bytes())
    for expected, actual in zip(jax.tree.leaves(template), jax.tree.leaves(restored), strict=True):
        if actual.shape != expected.shape:
            raise ValueError(f"Incompatible tokenizer checkpoint: {actual.shape} vs {expected.shape}")
    nnx.replace_by_pure_dict(state, jax.tree.map(jnp.asarray, restored))
    nnx.update(tokenizer, state)


def joint_loss(model, obs, actions, key, dynamics_weight, freeze_tokenizer):
    encoder, decoder = model.tokenizer.layers
    latents = encoder(obs.astype(jnp.float32))
    if freeze_tokenizer:
        latents = jax.lax.stop_gradient(latents)
    reconstruction = reconstruction_loss(decoder(latents), obs)
    if freeze_tokenizer:
        reconstruction = jax.lax.stop_gradient(reconstruction)

    history = latents[:, :-1]
    target = jax.lax.stop_gradient(latents[:, -1])
    noisy, t, d = sample_training_inputs(target, key)
    predicted = model.dynamics(history, actions, noisy, t, d)
    prediction = dynamics_loss(
        history, actions, noisy, t, d, predicted, target, model.dynamics, D_MIN
    )
    return reconstruction + dynamics_weight * prediction, (reconstruction, prediction)


@nnx.jit(static_argnames=("freeze_tokenizer",))
def update_step(model, optimizer, obs, actions, key, dynamics_weight, freeze_tokenizer):
    (loss, (rec, dyn)), grads = nnx.value_and_grad(joint_loss, has_aux=True)(
        model, obs, actions, key, dynamics_weight, freeze_tokenizer
    )
    optimizer.update(model, grads)
    return loss, rec, dyn


DEFAULT_DENOISING_STEPS = 4


@nnx.jit(static_argnames=("denoising_steps",))
def evaluate(model, obs, actions, key, dynamics_weight, denoising_steps=DEFAULT_DENOISING_STEPS):
    loss_key, rollout_key = jax.random.split(key)
    loss, (rec, dyn) = joint_loss(model, obs, actions, loss_key, dynamics_weight, False)
    encoder, decoder = model.tokenizer.layers
    history = encoder(obs[:, :-1].astype(jnp.float32))
    noise = jax.random.normal(rollout_key, history[:, -1].shape)
    step_size = jnp.full((len(obs),), 1.0 / denoising_steps, dtype=noise.dtype)

    def advance(index, latent):
        t = index * step_size
        clean = model.dynamics(history, actions, latent, t, step_size)
        # Advance a fraction of the remaining distance to the predicted clean latent.
        return latent + (clean - latent) / (denoising_steps - index)

    prediction = jax.lax.fori_loop(0, denoising_steps, advance, noise)
    next_frame_bce = reconstruction_loss(decoder(prediction), obs[:, -1])
    return loss, rec, dyn, next_frame_bce


def make_replay(episodes, context, seed):
    buffer = ReplayBuffer(capacity=len(episodes), rng=np.random.default_rng(seed))
    for episode in episodes:
        if len(episode["actions"]) >= context:
            buffer.add_episode(episode)
    if not buffer.episodes:
        raise ValueError(f"No episodes with {context} transitions in this split")
    return buffer


def main(args):
    env, params = make_env(args.game)
    channels = env.observation_space(params).shape[-1]
    model = WorldModel(nnx.Rngs(args.seed), channels, env.action_space(params).n)
    restore_tokenizer(model.tokenizer, args.tokenizer_checkpoint)
    optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)
    args.output.mkdir(parents=True, exist_ok=False)

    data_key, train_key, val_key = jax.random.split(jax.random.key(args.seed), 3)
    episodes, _ = collect_episodes(env, params, data_key, args.episodes)
    train_eps, val_eps = episode_partitions(episodes, args.seed)
    replay = make_replay(train_eps, args.context, args.seed)
    val = make_replay(val_eps, args.context, args.seed + 1)
    best = float("inf")
    for epoch in range(args.epochs):
        train_metrics = []
        for _ in range(args.steps_per_epoch):
            batch = replay.sample_batch(args.batch_size, args.context)
            train_key, step_key = jax.random.split(train_key)
            train_metrics.append(update_step(
                model, optimizer, batch["observations"], batch["actions"], step_key,
                args.dynamics_weight, args.freeze_tokenizer,
            ))
        # Repeat the same held-out windows and noise draws for comparable epochs.
        val.rng = np.random.default_rng(args.seed + 1)
        val_metrics = []
        for index in range(args.val_batches):
            batch = val.sample_batch(args.batch_size, args.context)
            val_metrics.append(evaluate(
                model, batch["observations"], batch["actions"],
                jax.random.fold_in(val_key, index), args.dynamics_weight,
                denoising_steps=args.denoising_steps,
            ))
        total, rec, dyn = np.mean(train_metrics, axis=0)
        vtotal, vrec, vdyn, next_bce = np.mean(val_metrics, axis=0)
        print(f"Epoch {epoch + 1}  train total {total:.5f} recon {rec:.5f} dyn {dyn:.5f}"
              f"  val total {vtotal:.5f} recon {vrec:.5f} dyn {vdyn:.5f}"
              f" next-frame BCE {next_bce:.5f}", flush=True)
        if next_bce < best:
            best = next_bce
            weights = nnx.to_pure_dict(nnx.state(model, nnx.Param))
            temporary = args.output / "best.msgpack.tmp"
            temporary.write_bytes(serialization.to_bytes(weights))
            temporary.replace(args.output / "best.msgpack")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train action-conditioned shortcut dynamics")
    parser.add_argument("--game", default="Breakout")
    parser.add_argument("--tokenizer-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New run directory")
    parser.add_argument("--freeze-tokenizer", action="store_true")
    parser.add_argument("--dynamics-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=16)
    parser.add_argument("--denoising-steps", type=int, choices=(1, 2, 4, 8, 16),
                        default=DEFAULT_DENOISING_STEPS, help="Denoising steps per evaluated next frame")
    args = parser.parse_args()
    if args.episodes < 2 or min(args.context, args.epochs, args.batch_size,
                                args.steps_per_epoch, args.val_batches) < 1:
        parser.error("Require at least two episodes and positive training dimensions")
    if not np.isfinite(args.dynamics_weight) or args.dynamics_weight <= 0:
        parser.error("--dynamics-weight must be positive and finite")
    main(args)
