import argparse
from pathlib import Path

import jax
import numpy as np
import optax
from flax import nnx, serialization
import jax.numpy as jnp

from collect import collect_episodes
from env import make_env
from model import OUTCOME_COMPONENTS, WorldModel, denoise_next, make_tokenizer, restore_weights
from replay import ReplayBuffer


BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 20
D_MIN = 1 / 16


def reconstruction_loss(logits, observations):
    loss = optax.sigmoid_binary_cross_entropy(
        logits,
        observations.astype(jnp.float32)
    )

    return jnp.mean(loss)


def batch_loss(model, observations):
    return reconstruction_loss(model(observations), observations)


def dynamics_loss(
    history_latents,
    history_actions,
    noisy_latent,
    noise_level,
    step_size,
    predicted_latent,
    target_latent,
    model,
    d_min,
):
    half_step = step_size / 2

    b_1 = (
        model(
            history_latents,
            history_actions,
            noisy_latent,
            noise_level,
            half_step,
        ) - noisy_latent
    ) / (1 - noise_level)[:, None]

    z_prime = noisy_latent + half_step[:, None] * b_1
    t_prime = noise_level + half_step

    b_2 = (
        model(
            history_latents,
            history_actions,
            z_prime,
            t_prime,
            half_step,
        ) - z_prime
    ) / (1 - t_prime)[:, None]

    target_velocity = jax.lax.stop_gradient((b_1 + b_2) / 2)

    predicted_velocity = (
        predicted_latent - noisy_latent
    ) / (1 - noise_level)[:, None]

    reconstruction_loss = jnp.mean(
        (predicted_latent - target_latent) ** 2,
        axis=-1,
    )

    shortcut_loss = ((1 - noise_level) ** 2) * jnp.mean(
        (predicted_velocity - target_velocity) ** 2,
        axis=-1,
    )

    loss = jnp.where(
        jnp.isclose(step_size, d_min),
        reconstruction_loss,
        shortcut_loss,
    )

    ramp_weight = 0.9 * noise_level + 0.1

    return jnp.mean(loss * ramp_weight)


validation_step = nnx.jit(batch_loss)


def sample_step_sizes(rng, batch_size):
    levels = jax.random.randint(
        rng,
        shape=(batch_size,),
        minval=0,
        maxval=5,
    )

    step_size = D_MIN * (2.0 ** levels)

    return step_size


def sample_training_inputs(target_latent, rng):
    B = target_latent.shape[0]

    step_key, time_key, noise_key = jax.random.split(rng, 3)

    # Sample shortcut step sizes
    d = sample_step_sizes(step_key, B)

    # Sample valid signal levels
    u = jax.random.uniform(time_key, (B,))
    t = u * (1.0 - d)

    # Construct noisy latents
    noise = jax.random.normal(noise_key, target_latent.shape)

    noisy_latent = (
        t[:, None] * target_latent
        + (1.0 - t[:, None]) * noise
    )

    return noisy_latent, t, d


def episode_partitions(episodes, seed=0):
    if len(episodes) < 2:
        raise ValueError("A train/validation split requires at least two episodes")
    order = np.random.default_rng(seed).permutation(len(episodes))
    val_count = max(1, round(0.1 * len(episodes)))
    return [episodes[i] for i in order[val_count:]], [episodes[i] for i in order[:val_count]]


def split_episodes(episodes, seed=0):
    train_eps, val_eps = episode_partitions(episodes, seed)
    train_obs = np.concatenate([ep["observations"] for ep in train_eps])
    val_obs = np.concatenate([ep["observations"] for ep in val_eps])
    return train_obs, val_obs


@nnx.jit
def train_step(model, optimizer, batch):
    loss, grads = nnx.value_and_grad(batch_loss)(model, batch)
    optimizer.update(model, grads)
    return loss


def train(observations, seed=0, *, validation_observations=None, model=None,
          batch_size=BATCH_SIZE, epochs=EPOCHS, output=None, learning_rate=LEARNING_RATE,
          transform=None):
    if len(observations) == 0:
        raise ValueError("Training requires at least one observation")
    if validation_observations is not None and len(validation_observations) == 0:
        raise ValueError("Validation requires at least one observation")

    if model is None:
        model = make_tokenizer(nnx.Rngs(seed), observations.shape[-1])
    if transform is None:
        transform = optax.adam(learning_rate)
    optimizer = nnx.Optimizer(model, transform, wrt=nnx.Param)
    rng = np.random.default_rng(seed)

    best = float("inf")
    for epoch in range(epochs):
        indices = rng.permutation(len(observations))
        total_loss = 0.0
        for start in range(0, len(indices), batch_size):
            batch = observations[indices[start:start + batch_size]]
            loss = train_step(model, optimizer, batch)
            total_loss += float(loss) * len(batch)

        metrics = f"Epoch {epoch + 1}/{epochs}  train BCE {total_loss / len(indices):.5f}"
        if validation_observations is not None:
            val_loss = 0.0
            for start in range(0, len(validation_observations), batch_size):
                batch = validation_observations[start:start + batch_size]
                val_loss += float(validation_step(model, batch)) * len(batch)
            val_loss /= len(validation_observations)
            metrics += f"  val BCE {val_loss:.5f}"
            if output is not None and val_loss < best:
                save_weights(model, output)
                best = val_loss
        print(metrics)
    return model


def save_weights(model, directory):
    weights = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    temporary = directory / "best.msgpack.tmp"
    temporary.write_bytes(serialization.to_bytes(weights))
    temporary.replace(directory / "best.msgpack")


def shortcut_loss(dynamics, latents, actions, key):
    history = latents[:, :-1]
    target = jax.lax.stop_gradient(latents[:, -1])
    noisy, t, d = sample_training_inputs(target, key)
    predicted = dynamics(history, actions, noisy, t, d)
    return dynamics_loss(history, actions, noisy, t, d, predicted, target, dynamics, D_MIN)


def joint_loss(model, obs, actions, key, dynamics_weight):
    encoder, decoder = model.tokenizer.layers
    latents = encoder(obs.astype(jnp.float32))
    reconstruction = reconstruction_loss(decoder(latents), obs)
    prediction = shortcut_loss(model.dynamics, latents, actions, key)
    return reconstruction + dynamics_weight * prediction, (reconstruction, prediction)


@nnx.jit
def update_step(model, optimizer, obs, actions, key, dynamics_weight):
    (loss, (rec, dyn)), grads = nnx.value_and_grad(joint_loss, has_aux=True)(
        model, obs, actions, key, dynamics_weight
    )
    optimizer.update(model, grads)
    return loss, rec, dyn


@nnx.jit
def frozen_update_step(dynamics, optimizer, latents, actions, key, dynamics_weight):
    def loss_fn(net):
        return dynamics_weight * shortcut_loss(net, latents, actions, key)

    loss, grads = nnx.value_and_grad(loss_fn)(dynamics)
    optimizer.update(dynamics, grads)
    return loss


@nnx.jit
def encode_frames(encoder, observations):
    return encoder(observations.astype(jnp.float32))


DEFAULT_DENOISING_STEPS = 4


@nnx.jit(static_argnames=("denoising_steps",))
def evaluate(model, obs, actions, key, dynamics_weight, denoising_steps=DEFAULT_DENOISING_STEPS):
    loss_key, rollout_key = jax.random.split(key)
    loss, (rec, dyn) = joint_loss(model, obs, actions, loss_key, dynamics_weight)
    encoder, decoder = model.tokenizer.layers
    history = encoder(obs[:, :-1].astype(jnp.float32))
    noise = jax.random.normal(rollout_key, history[:, -1].shape)
    prediction = denoise_next(model.dynamics, history, actions, noise, denoising_steps)
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


def cache_latents(encoder, replay):
    observations = np.concatenate([ep["observations"] for ep in replay.episodes])
    encoded = np.concatenate([np.asarray(encode_frames(encoder, observations[i:i + 512]))
                              for i in range(0, len(observations), 512)])
    offset = 0
    for ep in replay.episodes:
        end = offset + len(ep["observations"])
        ep["observations"] = encoded[offset:end]
        offset = end


def is_outcome_param(path, variable):
    return isinstance(variable, nnx.Param) and path[0] in OUTCOME_COMPONENTS


def outcome_metrics(dynamics, history, actions, targets):
    reward, terminal = dynamics.predict_outcomes(history, actions)
    logits = jnp.stack((reward, terminal), axis=-1)
    bce = optax.sigmoid_binary_cross_entropy(logits, targets).mean(axis=0)
    predictions = logits >= 0
    counts = jnp.stack(((predictions & targets).sum(axis=0), predictions.sum(axis=0),
                        targets.sum(axis=0)))
    return bce.sum(), (bce, counts)


def is_head_param(path, variable):
    return isinstance(variable, nnx.Param) and path[0] in ("reward_head", "terminal_head")


@nnx.jit(static_argnames=("param_filter",))
def outcome_update(dynamics, optimizer, history, actions, targets, param_filter=is_outcome_param):
    (loss, _), grads = nnx.value_and_grad(
        outcome_metrics, argnums=nnx.DiffState(0, param_filter), has_aux=True
    )(dynamics, history, actions, targets)
    optimizer.update(dynamics, grads)
    return loss


evaluate_outcomes = nnx.jit(outcome_metrics)


@nnx.jit
def dynamics_outcome_update(dynamics, optimizer, latents, actions, targets, key, weight):
    def objective(net):
        event_loss, _ = outcome_metrics(net, latents[:, :-1], actions, targets)
        shortcut = shortcut_loss(net, latents, actions, key)
        return event_loss + weight * shortcut, event_loss

    (_, event_loss), grads = nnx.value_and_grad(objective, has_aux=True)(dynamics)
    optimizer.update(dynamics, grads)
    return event_loss


def outcome_validation(dynamics, episodes, context, batch_size):
    # Score each eligible transition exactly once, including the last action.
    losses = np.zeros(2)
    counts = np.zeros((3, 2), dtype=np.int64)
    windows = [(ep, start) for ep in episodes for start in range(len(ep["actions"]) - context + 1)]
    for offset in range(0, len(windows), batch_size):
        group = windows[offset:offset + batch_size]
        history = np.stack([ep["observations"][s:s + context] for ep, s in group])
        actions = np.stack([ep["actions"][s:s + context] for ep, s in group])
        targets = np.array([(ep["rewards"][s + context - 1] > 0, ep["terminals"][s + context - 1])
                            for ep, s in group], dtype=bool)
        _, (bce, batch_counts) = evaluate_outcomes(dynamics, history, actions, targets)
        losses += np.asarray(bce) * len(group)
        counts += np.asarray(batch_counts)
    return losses / len(windows), counts, len(windows)


def train_outcomes(args, model, transform, replay, val, train_key):
    dynamics = model.dynamics
    joint = args.train == "dynamics-outcomes"
    param_filter = is_head_param if args.train == "heads" else is_outcome_param
    optimizer = nnx.Optimizer(dynamics, transform, wrt=nnx.Param if joint else param_filter)
    best = float("inf")
    stale = 0
    for epoch in range(args.epochs):
        train_losses = []
        for _ in range(args.steps_per_epoch):
            batch = replay.sample_batch(args.batch_size, args.context)
            targets = np.stack((batch["rewards"][:, -1] > 0, batch["terminals"][:, -1]), axis=-1)
            if joint:
                train_key, key = jax.random.split(train_key)
                loss = dynamics_outcome_update(dynamics, optimizer, batch["observations"],
                                               batch["actions"], targets, key, args.dynamics_weight)
            else:
                loss = outcome_update(dynamics, optimizer, batch["observations"][:, :-1],
                                      batch["actions"], targets, param_filter=param_filter)
            train_losses.append(loss)
        bce, counts, n = outcome_validation(dynamics, val.episodes, args.context, args.batch_size)
        total = bce.sum()
        print(f"Step {(epoch + 1) * args.steps_per_epoch} train BCE {np.mean(train_losses):.6f}"
              f" val BCE {total:.6f} ({n} windows)", flush=True)
        for name, loss, (tp, predicted, positives) in zip(("reward", "terminal"), bce, counts.T):
            print(f"  {name} BCE {loss:.6f}  precision {tp / predicted if predicted else float('nan'):.3f}"
                  f"  recall {tp / positives if positives else float('nan'):.3f}"
                  f"  positives {positives}  predicted {predicted}", flush=True)
        if total < best:
            save_weights(model, args.output)
            print(f"Saved best at step {(epoch + 1) * args.steps_per_epoch}, outcome BCE {total:.9g}", flush=True)
            best = total
            stale = 0
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                break


def train_dynamics(args, model, env, params, transform):
    freeze_tokenizer = args.train == "dynamics"

    data_key, train_key, val_key = jax.random.split(jax.random.key(args.seed), 3)
    episodes, _ = collect_episodes(env, params, data_key, args.episodes)
    train_eps, val_eps = episode_partitions(episodes, args.seed)
    replay = make_replay(train_eps, args.context, args.seed)
    val = make_replay(val_eps, args.context, args.seed + 1)
    outcome_mode = args.train in ("outcomes", "heads", "dynamics-outcomes")
    if freeze_tokenizer or outcome_mode:
        cache_latents(model.tokenizer.layers[0], replay)
    if outcome_mode:
        print(f"Training: {len(train_eps)} episodes, {len(replay.episodes)} eligible; "
              f"validation: {len(val_eps)} episodes, {len(val.episodes)} eligible", flush=True)
        cache_latents(model.tokenizer.layers[0], val)
        return train_outcomes(args, model, transform, replay, val, train_key)
    optimizer = nnx.Optimizer(model.dynamics if freeze_tokenizer else model,
                              transform, wrt=nnx.Param)
    best = float("inf")
    stale = 0
    for epoch in range(args.epochs):
        train_metrics = []
        for _ in range(args.steps_per_epoch):
            batch = replay.sample_batch(args.batch_size, args.context)
            train_key, step_key = jax.random.split(train_key)
            if freeze_tokenizer:
                train_metrics.append(frozen_update_step(
                    model.dynamics, optimizer, batch["observations"], batch["actions"], step_key,
                    args.dynamics_weight))
            else:
                train_metrics.append(update_step(
                    model, optimizer, batch["observations"], batch["actions"], step_key,
                    args.dynamics_weight))
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
        avg = np.mean(train_metrics, axis=0)
        if freeze_tokenizer:
            train_summary = f"train weighted dynamics {avg:.6f}"
        else:
            train_summary = f"train total {avg[0]:.6f} recon {avg[1]:.6f} dyn {avg[2]:.6f}"
        vtotal, vrec, vdyn, next_bce = np.mean(val_metrics, axis=0)
        print(f"Step {(epoch + 1) * args.steps_per_epoch}  {train_summary}"
              f"  val total {vtotal:.5f} recon {vrec:.5f} dyn {vdyn:.5f}"
              f" next-frame BCE {next_bce:.5f}", flush=True)
        if next_bce < best:
            best = next_bce
            stale = 0
            save_weights(model, args.output)
            print(f"Saved best at step {(epoch + 1) * args.steps_per_epoch}: BCE {next_bce:.9g}", flush=True)
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                print(f"Early stopping after {stale} validations without improvement", flush=True)
                break


def main():
    parser = argparse.ArgumentParser(description="Train the tokenizer and QKNorm shortcut dynamics")
    parser.add_argument("--train", choices=("tokenizer", "dynamics", "both", "outcomes", "heads",
                                            "dynamics-outcomes"), default="dynamics",
                        help="Components to update (default: dynamics with a frozen tokenizer)")
    init = parser.add_mutually_exclusive_group()
    init.add_argument("--tokenizer-checkpoint", type=Path)
    init.add_argument("--model-checkpoint", type=Path, help="Initialize from full QKNorm model weights")
    parser.add_argument("--output", type=Path, required=True, help="New checkpoint directory")
    parser.add_argument("--game", default="Breakout")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, help="Defaults to 10,000 for tokenizer or 1,000 for dynamics")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--optimizer", choices=("adamw", "adam"),
                        help="Defaults to Adam for tokenizer or AdamW for dynamics and joint training")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--epochs", type=int, help="Defaults to 20 for tokenizer or 50 validation intervals for dynamics")
    dyn = parser.add_argument_group("dynamics and joint training")
    dyn.add_argument("--context", type=int, default=16)
    dyn.add_argument("--steps-per-epoch", "--val-every", type=int, default=2000)
    dyn.add_argument("--val-batches", type=int, default=16)
    dyn.add_argument("--patience", type=int, default=5, help="Validation checks without improvement (0 disables)")
    dyn.add_argument("--dynamics-weight", type=float, default=1.0)
    dyn.add_argument("--denoising-steps", type=int, choices=(1, 2, 4, 8, 16), default=DEFAULT_DENOISING_STEPS)
    args = parser.parse_args()
    tokenizer_only = args.train == "tokenizer"
    if args.episodes is None:
        args.episodes = 10000 if tokenizer_only else 1000
    if args.epochs is None:
        args.epochs = EPOCHS if tokenizer_only else 50
    if args.train == "dynamics" and not (args.tokenizer_checkpoint or args.model_checkpoint):
        parser.error("dynamics training needs pretrained weights: supply --tokenizer-checkpoint or --model-checkpoint")
    if args.train in ("outcomes", "heads", "dynamics-outcomes") and not args.model_checkpoint:
        parser.error("outcome training requires --model-checkpoint with pretrained dynamics")
    if args.episodes < 2 or min(args.epochs, args.batch_size, args.context, args.steps_per_epoch, args.val_batches) < 1:
        parser.error("require at least two episodes and positive training dimensions")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if args.patience < 0 or not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--patience and --weight-decay must be nonnegative with finite weight decay")
    if not np.isfinite(args.dynamics_weight) or args.dynamics_weight <= 0:
        parser.error("--dynamics-weight must be finite and positive")

    env, params = make_env(args.game)
    channels = env.observation_space(params).shape[-1]
    if tokenizer_only and args.model_checkpoint is None:
        tokenizer = make_tokenizer(nnx.Rngs(args.seed), channels)
    else:
        model = WorldModel(nnx.Rngs(args.seed), channels, env.action_space(params).n)
        if args.model_checkpoint:
            restore_weights(model, args.model_checkpoint)
        tokenizer = model.tokenizer
    if args.tokenizer_checkpoint:
        restore_weights(tokenizer, args.tokenizer_checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    optimizer = args.optimizer or ("adam" if tokenizer_only else "adamw")
    transform = (optax.adamw(args.learning_rate, weight_decay=args.weight_decay)
                 if optimizer == "adamw" else optax.adam(args.learning_rate))
    if tokenizer_only:
        episodes, _ = collect_episodes(env, params, jax.random.PRNGKey(args.seed), args.episodes)
        observations, validation = split_episodes(episodes, args.seed)
        train(observations, args.seed, validation_observations=validation, model=tokenizer,
              epochs=args.epochs, batch_size=args.batch_size, output=args.output, transform=transform)
    else:
        train_dynamics(args, model, env, params, transform)


if __name__ == "__main__":
    main()
