"""Stage orchestration: collect replay, fit the world model and train a policy."""

import csv
import json
from dataclasses import asdict

import jax
import numpy as np
import optax
from flax import nnx, serialization

from .actor_critic import (
    actor_critic_update,
    adaptive_update,
    episode_update,
    imagined_ppo_update,
    policy_returns,
    short_update,
)
from .checkpoints import restore_weights, save_weights
from .config import TrainConfig
from .data import (
    RewardSampler,
    cache_latents,
    collect_episodes,
    encode_frames,
    episode_partitions,
    make_env,
    make_replay,
    split_episodes,
)
from .model import (
    REWARD_CLASSES,
    Actor,
    Critic,
    WorldModel,
    make_reward_head,
    make_tokenizer,
)
from .world_model_learning import (
    dynamics_outcome_update,
    evaluate,
    frozen_update_step,
    is_head_param,
    is_outcome_param,
    outcome_update,
    outcome_validation,
    train_step,
    update_step,
    validation_step,
)


def train_tokenizer(
    observations,
    seed=0,
    *,
    validation_observations=None,
    model=None,
    batch_size=TrainConfig.batch_size,
    epochs=TrainConfig.tokenizer_epochs,
    output=None,
    learning_rate=TrainConfig.learning_rate,
    transform=None,
):
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
            batch = observations[indices[start : start + batch_size]]
            loss = train_step(model, optimizer, batch)
            total_loss += float(loss) * len(batch)

        metrics = (
            f"Epoch {epoch + 1}/{epochs}  train BCE {total_loss / len(indices):.5f}"
        )
        if validation_observations is not None:
            val_loss = 0.0
            for start in range(0, len(validation_observations), batch_size):
                batch = validation_observations[start : start + batch_size]
                val_loss += float(validation_step(model, batch)) * len(batch)
            val_loss /= len(validation_observations)
            metrics += f"  val BCE {val_loss:.5f}"

            if output is not None and val_loss < best:
                save_weights(model, output)
                best = val_loss
        print(metrics)
    return model


def train_outcomes(args, model, transform, replay, val, train_key):
    dynamics = model.dynamics
    for buffer in (replay, val):
        for episode in buffer.episodes:
            r = episode["rewards"]
            if not np.all((r >= 0) & (r < REWARD_CLASSES) & (r == np.floor(r))):
                raise ValueError(
                    f"Expected integer MinAtar rewards from 0 to {REWARD_CLASSES - 1}"
                )

    if dynamics.reward_target != "categorical_points":
        dynamics.reward_head = make_reward_head(
            dynamics.reward_head.kernel.shape[0], nnx.Rngs(args.seed)
        )
        dynamics.reward_target = "categorical_points"
        print("Initialized categorical points head for outcome training", flush=True)

    joint = args.train == "dynamics-outcomes"
    param_filter = is_head_param if args.train == "heads" else is_outcome_param
    optimizer = nnx.Optimizer(
        dynamics, transform, wrt=nnx.Param if joint else param_filter
    )
    horizon = args.training_horizon if joint else 1
    sampler = (
        RewardSampler(replay.episodes, args.context) if args.reward_fraction else None
    )

    best = float("inf")
    stale = 0
    for epoch in range(args.epochs):
        train_losses = []
        for _ in range(args.steps_per_epoch):
            batch = replay.sample_batch(args.batch_size, args.context + horizon - 1)
            targets = np.stack(
                (batch["rewards"][:, -horizon:], batch["terminals"][:, -horizon:]),
                axis=-1,
            )

            if joint:
                train_key, key = jax.random.split(train_key)
                loss = dynamics_outcome_update(
                    dynamics,
                    optimizer,
                    batch["observations"],
                    batch["actions"],
                    targets,
                    key,
                    args.dynamics_weight,
                    horizon,
                    args.denoising_steps,
                    sampler.sample(replay.rng, args.batch_size, args.reward_fraction)
                    if sampler
                    else None,
                )
            else:
                loss = outcome_update(
                    dynamics,
                    optimizer,
                    batch["observations"][:, :-1],
                    batch["actions"],
                    targets[:, 0],
                    param_filter=param_filter,
                )
            train_losses.append(loss)

        losses, counts, n, mse = outcome_validation(
            dynamics,
            val.episodes,
            args.context,
            args.batch_size,
            horizon,
            args.denoising_steps,
        )
        total = losses.sum()
        print(
            f"Step {(epoch + 1) * args.steps_per_epoch} train outcome loss {np.mean(train_losses):.6f}"
            f" val loss {total:.6f} reward points MSE {mse:.6f} ({n} windows)",
            flush=True,
        )
        for name, loss, (tp, predicted, positives) in zip(
            ("reward CE", "terminal BCE"), losses, counts.T
        ):
            print(
                f"  {name} {loss:.6f}  precision {tp / predicted if predicted else float('nan'):.3f}"
                f"  recall {tp / positives if positives else float('nan'):.3f}"
                f"  positives {positives}  predicted {predicted}",
                flush=True,
            )

        if total < best:
            save_weights(model, args.output)
            print(
                f"Saved best at step {(epoch + 1) * args.steps_per_epoch}, outcome loss {total:.9g}",
                flush=True,
            )
            best = total
            stale = 0
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                break


def train_dynamics(args, model, env, params, transform):
    freeze_tokenizer = args.train == "dynamics"

    data_key, train_key, val_key = jax.random.split(jax.random.key(args.seed), 3)
    policy = None
    if args.train == "rollout-error" and args.policy_checkpoint:
        width = model.dynamics.latent_emb.kernel.shape[0] * 2
        rngs = nnx.Rngs(args.seed)
        agent = nnx.Dict(
            actor=Actor(rngs, env.action_space(params).n, width),
            critic=Critic(rngs, width),
        )
        restore_weights(agent, args.policy_checkpoint)
        policy = (agent.actor, model.tokenizer.layers[0])

    episodes, _ = collect_episodes(env, params, data_key, args.episodes, policy=policy)
    window = args.context + (
        0 if args.train in ("heads", "outcomes") else args.training_horizon - 1
    )
    if args.train == "rollout-error":
        window = args.context + args.imagination_horizon - 1

    eligible = [ep for ep in episodes if len(ep["actions"]) >= window]
    train_eps, val_eps = episode_partitions(eligible, args.seed)
    replay = make_replay(train_eps, window, args.seed)
    val = make_replay(val_eps, window, args.seed + 1)

    if args.train == "rollout-error":
        from .rollout_error import fit_probe

        for buffer in (replay, val):
            cache_latents(model.tokenizer.layers[0], buffer)
        return fit_probe(args, model.dynamics, transform, replay, val, train_key)

    outcome_mode = args.train in ("outcomes", "heads", "dynamics-outcomes")
    if freeze_tokenizer or outcome_mode:
        cache_latents(model.tokenizer.layers[0], replay)
    if outcome_mode:
        print(
            f"Training: {len(train_eps)} episodes, {len(replay.episodes)} eligible; "
            f"validation: {len(val_eps)} episodes, {len(val.episodes)} eligible",
            flush=True,
        )
        cache_latents(model.tokenizer.layers[0], val)
        return train_outcomes(args, model, transform, replay, val, train_key)

    optimizer = nnx.Optimizer(
        model.dynamics if freeze_tokenizer else model, transform, wrt=nnx.Param
    )
    best = float("inf")
    stale = 0
    for epoch in range(args.epochs):
        train_metrics = []
        for _ in range(args.steps_per_epoch):
            batch = replay.sample_batch(args.batch_size, window)
            train_key, step_key = jax.random.split(train_key)
            if freeze_tokenizer:
                train_metrics.append(
                    frozen_update_step(
                        model.dynamics,
                        optimizer,
                        batch["observations"],
                        batch["actions"],
                        step_key,
                        args.dynamics_weight,
                        args.training_horizon,
                        args.denoising_steps,
                    )
                )
            else:
                train_metrics.append(
                    update_step(
                        model,
                        optimizer,
                        batch["observations"],
                        batch["actions"],
                        step_key,
                        args.dynamics_weight,
                        args.training_horizon,
                        args.denoising_steps,
                    )
                )

        # Repeat the same held-out windows and noise draws for comparable epochs.
        val.rng = np.random.default_rng(args.seed + 1)
        val_metrics = []
        for index in range(args.val_batches):
            batch = val.sample_batch(args.batch_size, window)
            val_metrics.append(
                evaluate(
                    model,
                    batch["observations"],
                    batch["actions"],
                    jax.random.fold_in(val_key, index),
                    args.dynamics_weight,
                    denoising_steps=args.denoising_steps,
                    horizon=args.training_horizon,
                )
            )

        avg = np.mean(train_metrics, axis=0)
        if freeze_tokenizer:
            train_summary = f"train weighted dynamics {avg:.6f}"
        else:
            train_summary = (
                f"train total {avg[0]:.6f} recon {avg[1]:.6f} dyn {avg[2]:.6f}"
            )
        vtotal, vrec, vdyn, next_bce = np.mean(val_metrics, axis=0)
        print(
            f"Step {(epoch + 1) * args.steps_per_epoch}  {train_summary}"
            f"  val total {vtotal:.5f} recon {vrec:.5f} dyn {vdyn:.5f}"
            f" rollout BCE {next_bce:.5f}",
            flush=True,
        )

        if next_bce < best:
            best = next_bce
            stale = 0
            save_weights(model, args.output)
            print(
                f"Saved best at step {(epoch + 1) * args.steps_per_epoch}: BCE {next_bce:.9g}",
                flush=True,
            )
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                print(
                    f"Early stopping after {stale} validations without improvement",
                    flush=True,
                )
                break


def train_actor_critic(args, model, env, params, transform):
    data_key, train_key, eval_key, init_key = jax.random.split(
        jax.random.key(args.seed), 4
    )
    episodes, _ = collect_episodes(env, params, data_key, args.episodes)
    replay = make_replay(episodes, args.context, args.seed)
    encoder = model.tokenizer.layers[0]
    cache_latents(encoder, replay)

    input_dim = 2 * replay.episodes[0]["observations"].shape[-1]
    rngs = nnx.Rngs(init_key)
    agent = nnx.Dict(
        actor=Actor(rngs, env.action_space(params).n, input_dim),
        critic=Critic(rngs, input_dim),
    )

    head = None
    if args.error_checkpoint:
        from .rollout_error import load_probe

        head = load_probe(args, model.dynamics)

    ppo = args.policy_method == "ppo"
    if ppo:
        actor_tx = (
            optax.adam(args.actor_learning_rate)
            if args.optimizer == "adam"
            else optax.adamw(args.actor_learning_rate, weight_decay=args.weight_decay)
        )
        optimizers = (
            nnx.Optimizer(agent.actor, actor_tx, wrt=nnx.Param),
            nnx.Optimizer(agent.critic, transform, wrt=nnx.Param),
        )
        update = imagined_ppo_update
    else:
        optimizers = (nnx.Optimizer(agent, transform, wrt=nnx.Param),)
        update = actor_critic_update

    best = -np.inf
    stale = 0
    for epoch in range(args.epochs):
        losses = []
        for _ in range(args.steps_per_epoch):
            batch = replay.sample_batch(args.batch_size, args.context)
            train_key, key = jax.random.split(train_key)
            rollout = (
                model.dynamics,
                batch["observations"][:, :-1],
                batch["actions"][:, :-1],
                key,
                args.imagination_horizon,
                args.denoising_steps,
                args.discount,
                args.lambda_return,
                args.entropy_scale,
            )
            if head is None:
                losses.append(update(agent, *optimizers, *rollout))
            else:
                losses.append(
                    adaptive_update(
                        agent,
                        optimizers,
                        *rollout,
                        head,
                        args.error_threshold,
                        ppo,
                        args.error_low,
                    )
                )

        returns = policy_returns(
            agent.actor, encoder, env, params, eval_key, args.eval_episodes
        )
        score = float(returns.mean())
        actor_loss, critic_loss = np.mean(losses, axis=0)
        if not np.isfinite([actor_loss, critic_loss, score]).all():
            raise FloatingPointError("Nonfinite imagined policy training metrics")
        print(
            f"Step {(epoch + 1) * args.steps_per_epoch * (4 if ppo else 1)}  actor loss {actor_loss:.5f}"
            f"  critic CE {critic_loss:.5f}  real mean return {score:.4f}",
            flush=True,
        )

        if score > best:
            save_weights(agent, args.output)
            best, stale = score, 0
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                break


def train_real_policy(args, model, env, params):
    batch = args.batch_size
    method = args.policy_method
    reset_key, key, eval_key, init_key = jax.random.split(jax.random.key(args.seed), 4)
    obs, env_states = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(reset_key, batch), params
    )
    encoder = model.tokenizer.layers[0]
    z = encode_frames(encoder, obs)
    carry = (key, env_states, z, z)

    rngs = nnx.Rngs(init_key)
    agent = nnx.Dict(
        actor=Actor(rngs, env.action_space(params).n, z.shape[-1] * 2),
        critic=Critic(rngs, z.shape[-1] * 2),
    )

    def make_optimizer(net, lr):
        tx = (
            optax.adamw(lr, weight_decay=args.weight_decay)
            if args.optimizer != "adam"
            else optax.adam(lr)
        )
        return nnx.Optimizer(net, tx, wrt=nnx.Param)

    actor_opt = make_optimizer(agent.actor, args.actor_learning_rate)
    critic_opt = make_optimizer(agent.critic, args.learning_rate)

    interval = batch * args.real_horizon * args.steps_per_epoch
    actions = updates = 0
    best = -np.inf
    with (args.output / "validation.csv").open("w") as f:
        writer = csv.writer(f)
        action_names = ("noop", "left", "up", "right", "down", "fire")
        frequency_names = [action_names[int(i)] for i in env.action_set]
        writer.writerow(
            [
                "actions",
                "optimizer_updates",
                "mean_return",
                "actor_loss",
                "critic_ce",
                "entropy",
                *frequency_names,
            ]
        )

        for epoch in range(args.epochs + 1):
            metrics = []
            target = epoch * interval
            while actions < target:
                if method == "reinforce":
                    key, episode_key = jax.random.split(key)
                    count, m = episode_update(
                        agent,
                        actor_opt,
                        critic_opt,
                        encoder,
                        env,
                        params,
                        episode_key,
                        batch,
                        target - actions,
                        args.discount,
                        args.entropy_scale,
                    )
                else:
                    carry, m = short_update(
                        agent,
                        actor_opt,
                        critic_opt,
                        encoder,
                        env,
                        params,
                        carry,
                        args.real_horizon,
                        args.discount,
                        args.lambda_return,
                        args.entropy_scale,
                        method,
                    )
                    count = batch * args.real_horizon
                actions += count
                updates += 4 if method == "ppo" else 1
                metrics.append(m)

            score = float(
                policy_returns(
                    agent.actor, encoder, env, params, eval_key, args.eval_episodes
                ).mean()
            )
            summary = (
                np.mean(metrics, axis=0)
                if metrics
                else [""] * (3 + len(frequency_names))
            )
            if metrics and not np.isfinite(summary).all():
                raise FloatingPointError(f"Nonfinite metrics at {actions} actions")
            writer.writerow([actions, updates, score, *summary])
            f.flush()

            if score > best:
                best = score
                save_weights(agent, args.output)
            print(
                f"Actions {actions} updates {updates} validation {score:.4f} best {best:.4f}",
                flush=True,
            )

    restore_weights(agent, args.output / "best.msgpack")
    with (args.output / "heldout.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "lane", "return"])
        for seed in range(41001, 41005):
            scores = np.asarray(
                policy_returns(
                    agent.actor, encoder, env, params, jax.random.key(seed), 64
                )
            )
            writer.writerows((seed, i, s) for i, s in enumerate(scores))


def run_training(architecture, args):
    tokenizer_only = args.train == "tokenizer"
    env, params = make_env(args.game)
    channels = env.observation_space(params).shape[-1]

    latent_dim = architecture.latent_dim
    if args.tokenizer_checkpoint:
        saved = serialization.msgpack_restore(args.tokenizer_checkpoint.read_bytes())
        latent_dim = saved["layers"]["0"]["linear"]["bias"].shape[0]

    if tokenizer_only and args.model_checkpoint is None:
        tokenizer = make_tokenizer(nnx.Rngs(args.seed), channels, latent_dim)
    else:
        if args.model_checkpoint:
            saved = serialization.msgpack_restore(args.model_checkpoint.read_bytes())
            latent_dim = saved["dynamics"]["latent_emb"]["kernel"].shape[0]
        model = WorldModel(
            nnx.Rngs(args.seed),
            channels,
            env.action_space(params).n,
            latent_dim=latent_dim,
        )
        if args.model_checkpoint:
            restore_weights(
                model,
                args.model_checkpoint,
                require_outcomes=args.train
                in ("rollout-error", "actor-critic", "online"),
            )
        model.dynamics.detach_rollout = args.detach_rollout
        tokenizer = model.tokenizer

    if args.tokenizer_checkpoint:
        restore_weights(tokenizer, args.tokenizer_checkpoint)
    if args.train == "online" and model.dynamics.reward_target != "categorical_points":
        raise ValueError(
            "Online training needs a categorical points reward head. Fit it first with --train dynamics-outcomes."
        )

    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {"model": {"latent_dim": latent_dim}, "training": asdict(args)}
    (args.output / "config.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )
    if args.train in ("online", "actor-critic", "real-actor-critic"):
        (args.output / "tokenizer").mkdir()
        save_weights(tokenizer, args.output / "tokenizer")

    optimizer = args.optimizer or ("adam" if tokenizer_only else "adamw")
    transform = (
        optax.adamw(args.learning_rate, weight_decay=args.weight_decay)
        if optimizer == "adamw"
        else optax.adam(args.learning_rate)
    )

    if tokenizer_only:
        episodes, _ = collect_episodes(
            env, params, jax.random.PRNGKey(args.seed), args.episodes
        )
        observations, validation = split_episodes(episodes, args.seed)
        train_tokenizer(
            observations,
            args.seed,
            validation_observations=validation,
            model=tokenizer,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output=args.output,
            transform=transform,
        )
    elif args.train == "online":
        train_online(args, model, env, params, transform)
    elif args.train == "actor-critic":
        train_actor_critic(args, model, env, params, transform)
    elif args.train == "real-actor-critic":
        train_real_policy(args, model, env, params)
    else:
        train_dynamics(args, model, env, params, transform)


def train_online(args, model, env, params, transform):
    """Alternate real collection, replay-based dynamics updates and imagined PPO."""
    from .data import ReplayBuffer, collect_online_block

    encoder = model.tokenizer.layers[0]
    collect_key, reset_key, data_key, train_key, eval_key = jax.random.split(
        jax.random.key(args.seed), 5
    )
    obs, state = env.reset(reset_key, params)
    latent = encode_frames(encoder, obs[None])[0]
    carry = (collect_key, state, latent, latent)

    rngs = nnx.Rngs(args.seed)
    agent = nnx.Dict(
        actor=Actor(rngs, env.action_space(params).n, 2 * latent.size),
        critic=Critic(rngs, 2 * latent.size),
    )

    actor_tx = (
        optax.adam(args.actor_learning_rate)
        if args.optimizer == "adam"
        else optax.adamw(args.actor_learning_rate, weight_decay=args.weight_decay)
    )
    actor_opt = nnx.Optimizer(agent.actor, actor_tx, wrt=nnx.Param)
    critic_opt = nnx.Optimizer(agent.critic, transform, wrt=nnx.Param)
    world_opt = nnx.Optimizer(model.dynamics, transform, wrt=nnx.Param)

    length = args.context + args.training_horizon - 1
    episodes, _ = collect_episodes(env, params, data_key, args.episodes)
    offline = make_replay(episodes, length, args.seed)
    cache_latents(encoder, offline)
    offline_sampler = (
        RewardSampler(offline.episodes, args.context) if args.reward_fraction else None
    )
    online = ReplayBuffer(rng=np.random.default_rng(args.seed + 1))
    pending = []

    best = -np.inf
    (args.output / "world").mkdir()

    def pack(transitions):
        before, after, actions, rewards, dones, terminals = zip(*transitions)
        return {
            "observations": np.stack((*before, after[-1])),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "dones": np.array(dones),
            "terminals": np.array(terminals),
        }

    def sample():
        n_online = args.batch_size // 2 if online.episodes else 0
        batch = offline.sample_batch(args.batch_size - n_online, length)
        if n_online:
            other = online.sample_batch(n_online, length)
            batch = {k: np.concatenate((v, other[k])) for k, v in batch.items()}
        return batch

    for cycle in range(1, args.epochs + 1):
        carry, transitions = collect_online_block(
            agent.actor, encoder, env, params, carry, args.collect_steps
        )
        for transition in zip(*jax.device_get(transitions)):
            pending.append(transition)
            if transition[4]:
                if len(pending) >= length:
                    online.add_episode(pack(pending))
                pending.clear()

        # Make the current partial episode available until the next collection block.
        partial = len(pending) >= length
        if partial:
            online.episodes.append(pack(pending))

        # The reward head can oversample scoring transitions from either replay source.
        online_sampler = (
            RewardSampler(online.episodes, args.context)
            if offline_sampler and online.episodes
            else None
        )
        for _ in range(args.world_updates):
            reward_batch = None
            if offline_sampler:
                online_size = args.batch_size // 2 if online_sampler else 0
                reward_batch = offline_sampler.sample(
                    offline.rng, args.batch_size - online_size, args.reward_fraction
                )
                if online_sampler:
                    extra = online_sampler.sample(
                        online.rng, online_size, args.reward_fraction
                    )
                    reward_batch = tuple(
                        np.concatenate(pair) for pair in zip(reward_batch, extra)
                    )

            batch = sample()
            train_key, key = jax.random.split(train_key)
            targets = np.stack((batch["rewards"], batch["terminals"]), axis=-1)[
                :, -args.training_horizon :
            ]
            loss = dynamics_outcome_update(
                model.dynamics,
                world_opt,
                batch["observations"],
                batch["actions"],
                targets,
                key,
                args.dynamics_weight,
                args.training_horizon,
                args.denoising_steps,
                reward_batch,
            )

        # PPO starts from replay contexts and follows the updated world model.
        for _ in range(args.steps_per_epoch):
            batch = sample()
            train_key, key = jax.random.split(train_key)
            policy_losses = imagined_ppo_update(
                agent,
                actor_opt,
                critic_opt,
                model.dynamics,
                batch["observations"][:, : args.context],
                batch["actions"][:, : args.context - 1],
                key,
                args.imagination_horizon,
                args.denoising_steps,
                args.discount,
                args.lambda_return,
                args.entropy_scale,
            )

        if partial:
            online.episodes.pop()

        score = float(
            policy_returns(
                agent.actor, encoder, env, params, eval_key, args.eval_episodes
            ).mean()
        )
        if not np.isfinite([loss, *policy_losses, score]).all():
            raise FloatingPointError(f"Nonfinite online metrics at cycle {cycle}")

        if score > best:
            save_weights(model, args.output / "world")
            save_weights(agent, args.output)
            best = score
        print(
            f"Cycle {cycle}  real actions {cycle * args.collect_steps}  mean return {score:.3f}",
            flush=True,
        )
