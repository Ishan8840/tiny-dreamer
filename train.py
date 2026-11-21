import argparse
from pathlib import Path

import jax
import numpy as np
import optax
from flax import nnx, serialization
import jax.numpy as jnp

from collect import collect_episodes
from env import make_env
from model import (Actor, Critic, OUTCOME_COMPONENTS, REWARD_CLASSES, WorldModel,
                   denoise_next, make_reward_head, make_tokenizer, restore_weights)
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
    if isinstance(model, WorldModel):
        weights["_reward_target"] = model.dynamics.reward_target
    temporary = directory / "best.msgpack.tmp"
    temporary.write_bytes(serialization.to_bytes(weights))
    temporary.replace(directory / "best.msgpack")


DEFAULT_DENOISING_STEPS = 4


def rollout_steps(dynamics, latents, actions, key, horizon, denoising_steps):
    context = actions.shape[1] - horizon + 1
    if context < 1 or horizon < 1 or latents.shape[1] != actions.shape[1] + 1:
        raise ValueError("A rollout needs context + horizon frames and one fewer action")
    history = latents[:, :context]
    for step in range(horizon):
        window = actions[:, step:step + context]
        target = latents[:, context + step]
        yield history, window, target
        if step + 1 < horizon:
            noise = jax.random.normal(jax.random.fold_in(key, step), target.shape)
            prediction = denoise_next(dynamics, history, window, noise, denoising_steps)
            history = jnp.concatenate((history[:, 1:], prediction[:, None]), axis=1)


def rollout_loss(dynamics, latents, actions, key, horizon, denoising_steps, outcomes=None):
    shortcut = event = 0.
    rollout_key, loss_key = jax.random.split(key)
    for step, (history, window, target) in enumerate(
            rollout_steps(dynamics, latents, actions, rollout_key, horizon, denoising_steps)):
        target = jax.lax.stop_gradient(target)
        noise_key = key if horizon == 1 else jax.random.fold_in(loss_key, step)
        noisy, t, d = sample_training_inputs(target, noise_key)
        predicted = dynamics(history, window, noisy, t, d)
        shortcut += dynamics_loss(history, window, noisy, t, d, predicted, target, dynamics, D_MIN)
        if outcomes is not None:
            labels = outcomes[:, step] if outcomes.ndim == 3 else outcomes
            event += outcome_metrics(dynamics, history, window, labels)[0]
    return shortcut / horizon, event / horizon


def shortcut_loss(dynamics, latents, actions, key, horizon=1, denoising_steps=DEFAULT_DENOISING_STEPS):
    return rollout_loss(dynamics, latents, actions, key, horizon, denoising_steps)[0]


def joint_loss(model, obs, actions, key, dynamics_weight, horizon=1, denoising_steps=DEFAULT_DENOISING_STEPS):
    encoder, decoder = model.tokenizer.layers
    latents = encoder(obs.astype(jnp.float32))
    reconstruction = reconstruction_loss(decoder(latents), obs)
    prediction = shortcut_loss(model.dynamics, latents, actions, key, horizon, denoising_steps)
    return reconstruction + dynamics_weight * prediction, (reconstruction, prediction)


@nnx.jit(static_argnames=("horizon", "denoising_steps"))
def update_step(model, optimizer, obs, actions, key, dynamics_weight, horizon=1, denoising_steps=DEFAULT_DENOISING_STEPS):
    (loss, (rec, dyn)), grads = nnx.value_and_grad(joint_loss, has_aux=True)(
        model, obs, actions, key, dynamics_weight, horizon, denoising_steps
    )
    optimizer.update(model, grads)
    return loss, rec, dyn


@nnx.jit(static_argnames=("horizon", "denoising_steps"))
def frozen_update_step(dynamics, optimizer, latents, actions, key, dynamics_weight, horizon=1, denoising_steps=DEFAULT_DENOISING_STEPS):
    def loss_fn(net):
        return dynamics_weight * shortcut_loss(net, latents, actions, key, horizon, denoising_steps)

    loss, grads = nnx.value_and_grad(loss_fn)(dynamics)
    optimizer.update(dynamics, grads)
    return loss


@nnx.jit
def encode_frames(encoder, observations):
    return encoder(observations.astype(jnp.float32))


@nnx.jit(static_argnames=("denoising_steps", "horizon"))
def evaluate(model, obs, actions, key, dynamics_weight, denoising_steps=DEFAULT_DENOISING_STEPS, horizon=1):
    loss_key, rollout_key = jax.random.split(key)
    loss, (rec, dyn) = joint_loss(model, obs, actions, loss_key, dynamics_weight, horizon, denoising_steps)
    encoder, decoder = model.tokenizer.layers
    context = actions.shape[1] - horizon + 1
    history = encoder(obs[:, :context].astype(jnp.float32))
    bce = 0.
    for step in range(horizon):
        noise_key = rollout_key if horizon == 1 else jax.random.fold_in(rollout_key, step)
        noise = jax.random.normal(noise_key, history[:, -1].shape)
        prediction = denoise_next(model.dynamics, history, actions[:, step:step + context], noise, denoising_steps)
        bce += reconstruction_loss(decoder(prediction), obs[:, context + step])
        history = jnp.concatenate((history[:, 1:], prediction[:, None]), axis=1)
    return loss, rec, dyn, bce / horizon


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
    reward, terminal = dynamics.outcome_outputs(history, actions)
    points = targets[:, 0]
    reward_ce = optax.softmax_cross_entropy_with_integer_labels(reward, points.astype(jnp.int32))
    terminal_bce = optax.sigmoid_binary_cross_entropy(terminal, targets[:, 1])
    losses = jnp.stack((reward_ce.mean(), terminal_bce.mean()))
    predictions = jnp.stack((dynamics.scoring_logit(reward) >= 0, terminal >= 0), axis=-1)
    events = targets > 0
    counts = jnp.stack(((predictions & events).sum(axis=0), predictions.sum(axis=0),
                        events.sum(axis=0)))
    reward_mse = jnp.mean(jnp.square(dynamics.expected_reward(reward) - points))
    return losses.sum(), (losses, counts, reward_mse)


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


@nnx.jit(static_argnames=("horizon", "denoising_steps"))
def dynamics_outcome_update(dynamics, optimizer, latents, actions, targets, key, weight, horizon=1, denoising_steps=DEFAULT_DENOISING_STEPS):
    def objective(net):
        shortcut, event_loss = rollout_loss(net, latents, actions, key, horizon, denoising_steps, targets)
        return event_loss + weight * shortcut, event_loss

    (_, event_loss), grads = nnx.value_and_grad(objective, has_aux=True)(dynamics)
    optimizer.update(dynamics, grads)
    return event_loss


@nnx.jit(static_argnames=("horizon", "denoising_steps"))
def evaluate_rollout_outcomes(dynamics, latents, actions, targets, key, horizon, denoising_steps):
    metrics = [outcome_metrics(dynamics, history, window, targets[:, step])[1]
               for step, (history, window, _) in enumerate(
                   rollout_steps(dynamics, latents, actions, key, horizon, denoising_steps))]
    losses, counts, mses = zip(*metrics)
    return jnp.mean(jnp.stack(losses), axis=0), sum(counts), jnp.mean(jnp.stack(mses))


def outcome_validation(dynamics, episodes, context, batch_size, horizon=1, denoising_steps=DEFAULT_DENOISING_STEPS):
    losses = np.zeros(2)
    squared_error = 0.
    counts = np.zeros((3, 2), dtype=np.int64)
    length = context + horizon - 1
    windows = [(ep, start) for ep in episodes for start in range(len(ep["actions"]) - length + 1)]
    for offset in range(0, len(windows), batch_size):
        group = windows[offset:offset + batch_size]
        latents = np.stack([ep["observations"][s:s + length + 1] for ep, s in group])
        actions = np.stack([ep["actions"][s:s + length] for ep, s in group])
        targets = np.stack([np.stack((ep["rewards"][s + context - 1:s + length],
                                      ep["terminals"][s + context - 1:s + length]), axis=-1)
                            for ep, s in group]).astype(np.float32)
        batch_losses, batch_counts, mse = evaluate_rollout_outcomes(
            dynamics, latents, actions, targets, jax.random.fold_in(jax.random.key(0), offset),
            horizon, denoising_steps)
        losses += np.asarray(batch_losses) * len(group)
        squared_error += float(mse) * len(group)
        counts += np.asarray(batch_counts)
    return losses / len(windows), counts, len(windows), squared_error / len(windows)


def train_outcomes(args, model, transform, replay, val, train_key):
    dynamics = model.dynamics
    for buffer in (replay, val):
        for episode in buffer.episodes:
            r = episode["rewards"]
            if not np.all((r >= 0) & (r < REWARD_CLASSES) & (r == np.floor(r))):
                raise ValueError(f"Expected integer MinAtar rewards from 0 to {REWARD_CLASSES - 1}")
    if dynamics.reward_target != "categorical_points":
        dynamics.reward_head = make_reward_head(dynamics.reward_head.kernel.shape[0], nnx.Rngs(args.seed))
        dynamics.reward_target = "categorical_points"
        print("Initialized categorical points head for outcome training", flush=True)
    joint = args.train == "dynamics-outcomes"
    param_filter = is_head_param if args.train == "heads" else is_outcome_param
    optimizer = nnx.Optimizer(dynamics, transform, wrt=nnx.Param if joint else param_filter)
    horizon = args.training_horizon if joint else 1
    best = float("inf")
    stale = 0
    for epoch in range(args.epochs):
        train_losses = []
        for _ in range(args.steps_per_epoch):
            batch = replay.sample_batch(args.batch_size, args.context + horizon - 1)
            targets = np.stack((batch["rewards"][:, -horizon:], batch["terminals"][:, -horizon:]), axis=-1)
            if joint:
                train_key, key = jax.random.split(train_key)
                loss = dynamics_outcome_update(dynamics, optimizer, batch["observations"],
                                               batch["actions"], targets, key, args.dynamics_weight, horizon, args.denoising_steps)
            else:
                loss = outcome_update(dynamics, optimizer, batch["observations"][:, :-1],
                                      batch["actions"], targets[:, 0], param_filter=param_filter)
            train_losses.append(loss)
        losses, counts, n, mse = outcome_validation(dynamics, val.episodes, args.context, args.batch_size, horizon, args.denoising_steps)
        total = losses.sum()
        print(f"Step {(epoch + 1) * args.steps_per_epoch} train outcome loss {np.mean(train_losses):.6f}"
              f" val loss {total:.6f} reward points MSE {mse:.6f} ({n} windows)", flush=True)
        for name, loss, (tp, predicted, positives) in zip(("reward CE", "terminal BCE"), losses, counts.T):
            print(f"  {name} {loss:.6f}  precision {tp / predicted if predicted else float('nan'):.3f}"
                  f"  recall {tp / positives if positives else float('nan'):.3f}"
                  f"  positives {positives}  predicted {predicted}", flush=True)
        if total < best:
            save_weights(model, args.output)
            print(f"Saved best at step {(epoch + 1) * args.steps_per_epoch}, outcome loss {total:.9g}", flush=True)
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
    window = args.context + (0 if args.train in ("heads", "outcomes") else args.training_horizon - 1)
    replay = make_replay(train_eps, window, args.seed)
    val = make_replay(val_eps, window, args.seed + 1)
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
            batch = replay.sample_batch(args.batch_size, window)
            train_key, step_key = jax.random.split(train_key)
            if freeze_tokenizer:
                train_metrics.append(frozen_update_step(
                    model.dynamics, optimizer, batch["observations"], batch["actions"], step_key,
                    args.dynamics_weight, args.training_horizon, args.denoising_steps))
            else:
                train_metrics.append(update_step(
                    model, optimizer, batch["observations"], batch["actions"], step_key,
                    args.dynamics_weight, args.training_horizon, args.denoising_steps))
        # Repeat the same held-out windows and noise draws for comparable epochs.
        val.rng = np.random.default_rng(args.seed + 1)
        val_metrics = []
        for index in range(args.val_batches):
            batch = val.sample_batch(args.batch_size, window)
            val_metrics.append(evaluate(
                model, batch["observations"], batch["actions"],
                jax.random.fold_in(val_key, index), args.dynamics_weight,
                denoising_steps=args.denoising_steps, horizon=args.training_horizon,
            ))
        avg = np.mean(train_metrics, axis=0)
        if freeze_tokenizer:
            train_summary = f"train weighted dynamics {avg:.6f}"
        else:
            train_summary = f"train total {avg[0]:.6f} recon {avg[1]:.6f} dyn {avg[2]:.6f}"
        vtotal, vrec, vdyn, next_bce = np.mean(val_metrics, axis=0)
        print(f"Step {(epoch + 1) * args.steps_per_epoch}  {train_summary}"
              f"  val total {vtotal:.5f} recon {vrec:.5f} dyn {vdyn:.5f}"
              f" rollout BCE {next_bce:.5f}", flush=True)
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


def policy_state(history):
    return jnp.concatenate((history[:, -2], history[:, -1]), axis=-1)


def lambda_returns(rewards, discounts, next_values, trace_decay):
    def backup(future, transition):
        reward, discount, next_value = transition
        target = reward + discount * ((1 - trace_decay) * next_value + trace_decay * future)
        return target, target

    _, targets = jax.lax.scan(backup, next_values[-1], (rewards, discounts, next_values), reverse=True)
    return targets


def imagined_batch(agent, dynamics, history, past_actions, keys, denoising_steps, gamma, trace_decay):
    def imagine(carry, step_key):
        history, past_actions = carry
        action_key, noise_key = jax.random.split(step_key)
        state = policy_state(history)
        action = jax.random.categorical(action_key, agent.actor(state)).astype(past_actions.dtype)
        actions = jnp.concatenate((past_actions, action[:, None]), axis=1)
        reward, terminal = dynamics.outcome_outputs(history, actions)
        noise = jax.random.normal(noise_key, history[:, -1].shape)
        next_z = denoise_next(dynamics, history, actions, noise, denoising_steps)
        next_history = jnp.concatenate((history[:, 1:], next_z[:, None]), axis=1)
        transition = (state, action, dynamics.expected_reward(reward), gamma * jax.nn.sigmoid(-terminal))
        return (next_history, actions[:, 1:]), transition

    (last_history, _), (states, actions, rewards, discounts) = jax.lax.scan(
        imagine, (history, past_actions), keys)
    bootstrap_states = jnp.concatenate((states[1:], policy_state(last_history)[None]), axis=0)
    next_values = agent.critic(bootstrap_states)[..., 0]
    targets = jax.lax.stop_gradient(lambda_returns(rewards, discounts, next_values, trace_decay))
    weights = jnp.cumprod(jnp.concatenate((jnp.ones_like(discounts[:1]), discounts[:-1])), axis=0)
    return states, actions, targets, weights


@nnx.jit(static_argnames=("horizon", "denoising_steps"))
def actor_critic_update(agent, optimizer, dynamics, history, past_actions, key, horizon,
                        denoising_steps, gamma, trace_decay, entropy_scale):
    batch = imagined_batch(agent, dynamics, history, past_actions, jax.random.split(key, horizon),
                           denoising_steps, gamma, trace_decay)
    return policy_update(agent, optimizer, *batch, entropy_scale)


@nnx.jit(static_argnames=("horizon", "denoising_steps"))
def imagined_ppo_update(agent, actor_opt, critic_opt, dynamics, history, past_actions, key, horizon,
                        denoising_steps, gamma, trace_decay, entropy_scale):
    from policy_ablation import learn
    # Match the noise and sampling keys for shared steps when comparing horizons.
    keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(horizon))
    batch = imagined_batch(agent, dynamics, history, past_actions, keys, denoising_steps, gamma, trace_decay)
    return learn(agent, actor_opt, critic_opt, *batch, "ppo", entropy_scale)[:2]


def policy_update(agent, optimizer, states, actions, targets, weights, entropy_scale):
    def loss_fn(net):
        logits = net.critic.logits(states)
        values = net.critic.value(logits)[..., 0]
        advantage = jax.lax.stop_gradient(targets - values)
        log_probs = jax.nn.log_softmax(net.actor(states))
        chosen = jnp.take_along_axis(log_probs, actions[..., None], axis=-1)[..., 0]
        entropy = -(jnp.exp(log_probs) * log_probs).sum(axis=-1)
        actor_loss = -(weights * (chosen * advantage + entropy_scale * entropy)).sum() / weights.sum()
        critic_loss = (weights * net.critic.return_loss(logits, targets)).sum() / weights.sum()
        return actor_loss + critic_loss, (actor_loss, critic_loss)

    (_, losses), grads = nnx.value_and_grad(loss_fn, has_aux=True)(agent)
    optimizer.update(agent, grads)
    return losses


@nnx.jit(static_argnames=("env", "episodes"))
def policy_returns(actor, encoder, env, params, key, episodes):
    reset_key, rollout_key = jax.random.split(key)
    obs, env_states = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(reset_key, episodes), params)
    latent = encoder(obs.astype(jnp.float32))

    def advance(carry):
        key, env_states, previous, current, active, returns = carry
        key, action_key, step_key = jax.random.split(key, 3)
        actions = jax.random.categorical(action_key, actor(jnp.concatenate((previous, current), axis=-1)))
        obs, next_states, reward, done, _ = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))(
            jax.random.split(step_key, episodes), env_states, actions, params)
        done |= jax.vmap(env.is_truncated, in_axes=(0, None))(next_states, params)
        # Finished lanes keep their last valid state until the longest episode ends.
        def keep_active(new, old):
            mask = (active & ~done).reshape((episodes,) + (1,) * (new.ndim - 1))
            return jnp.where(mask, new, old)
        next_states = jax.tree.map(keep_active, next_states, env_states)
        return (key, next_states, current, encoder(obs.astype(jnp.float32)), active & ~done,
                returns + jnp.where(active, reward, 0.0))

    initial = (rollout_key, env_states, latent, latent, jnp.ones(episodes, bool), jnp.zeros(episodes))
    return jax.lax.while_loop(lambda carry: carry[4].any(), advance, initial)[-1]


def train_actor_critic(args, model, env, params, transform):
    data_key, train_key, eval_key, init_key = jax.random.split(jax.random.key(args.seed), 4)
    episodes, _ = collect_episodes(env, params, data_key, args.episodes)
    replay = make_replay(episodes, args.context, args.seed)
    encoder = model.tokenizer.layers[0]
    cache_latents(encoder, replay)
    input_dim = 2 * replay.episodes[0]["observations"].shape[-1]
    rngs = nnx.Rngs(init_key)
    agent = nnx.Dict(actor=Actor(rngs, env.action_space(params).n, input_dim),
                     critic=Critic(rngs, input_dim))
    ppo = args.policy_method == "ppo"
    if ppo:
        actor_tx = (optax.adam(args.actor_learning_rate) if args.optimizer == "adam" else
                    optax.adamw(args.actor_learning_rate, weight_decay=args.weight_decay))
        optimizers = (nnx.Optimizer(agent.actor, actor_tx, wrt=nnx.Param),
                      nnx.Optimizer(agent.critic, transform, wrt=nnx.Param))
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
            losses.append(update(
                agent, *optimizers, model.dynamics, batch["observations"][:, :-1],
                batch["actions"][:, :-1], key, args.imagination_horizon, args.denoising_steps,
                args.discount, args.lambda_return, args.entropy_scale))
        returns = policy_returns(agent.actor, encoder, env, params, eval_key, args.eval_episodes)
        score = float(returns.mean())
        actor_loss, critic_loss = np.mean(losses, axis=0)
        if not np.isfinite([actor_loss, critic_loss, score]).all():
            raise FloatingPointError("Nonfinite imagined policy training metrics")
        print(f"Step {(epoch + 1) * args.steps_per_epoch * (4 if ppo else 1)}  actor loss {actor_loss:.5f}"
              f"  critic CE {critic_loss:.5f}  real mean return {score:.4f}", flush=True)
        if score > best:
            save_weights(agent, args.output)
            best, stale = score, 0
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                break


def real_rollout(agent, encoder, env, params, carry, horizon, gamma, bootstrap_timeouts=True):
    def transition(carry, _):
        key, env_states, previous, current = carry
        key, action_key, step_key, reset_key = jax.random.split(key, 4)
        state = jnp.concatenate((previous, current), axis=-1)
        action = jax.random.categorical(action_key, agent.actor(state))
        batch = action.shape[0]
        obs, next_env, reward, done, _ = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))(
            jax.random.split(step_key, batch), env_states, action, params)
        terminal = jax.vmap(env.is_terminated, in_axes=(0, None))(next_env, params)
        done |= jax.vmap(env.is_truncated, in_axes=(0, None))(next_env, params)
        next_z = encoder(obs.astype(jnp.float32))
        next_value = agent.critic(jnp.concatenate((current, next_z), axis=-1))[..., 0]
        # Bootstrap at time limits, but never propagate a return across an episode reset.
        if bootstrap_timeouts:
            reward = reward + gamma * (done & ~terminal) * next_value
        reset_obs, reset_env = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(reset_key, batch), params)
        def reset_lanes(reset, ongoing):
            mask = done.reshape((batch,) + (1,) * (ongoing.ndim - 1))
            return jnp.where(mask, reset, ongoing)
        next_env = jax.tree.map(reset_lanes, reset_env, next_env)
        reset_z = encoder(reset_obs.astype(jnp.float32))
        carry = (key, next_env, reset_lanes(reset_z, current), reset_lanes(reset_z, next_z))
        return carry, (state, action, reward, gamma * ~done, next_value, done)

    return jax.lax.scan(transition, carry, xs=None, length=horizon)


@nnx.jit(static_argnames=("env", "horizon"))
def real_actor_critic_update(agent, optimizer, encoder, env, params, carry, horizon, gamma,
                             trace_decay, entropy_scale):
    carry, (states, actions, rewards, discounts, next_values, _) = real_rollout(
        agent, encoder, env, params, carry, horizon, gamma)
    targets = lambda_returns(rewards, discounts, next_values, trace_decay)
    losses = policy_update(agent, optimizer, states, actions, targets, jnp.ones_like(rewards), entropy_scale)
    return carry, losses


def train_real_agent(args, model, env, params, transform):
    reset_key, key, eval_key, init_key = jax.random.split(jax.random.key(args.seed), 4)
    obs, env_states = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(reset_key, args.batch_size), params)
    encoder = model.tokenizer.layers[0]
    latent = encode_frames(encoder, obs)
    carry = (key, env_states, latent, latent)
    rngs = nnx.Rngs(init_key)
    agent = nnx.Dict(actor=Actor(rngs, env.action_space(params).n, latent.shape[-1] * 2),
                     critic=Critic(rngs, latent.shape[-1] * 2))
    optimizer = nnx.Optimizer(agent, transform, wrt=nnx.Param)
    best = -np.inf
    for epoch in range(args.epochs + 1):
        if epoch:
            for _ in range(args.steps_per_epoch):
                carry, losses = real_actor_critic_update(agent, optimizer, encoder, env, params, carry,
                    args.real_horizon, args.discount, args.lambda_return, args.entropy_scale)
        step = epoch * args.steps_per_epoch
        returns = policy_returns(agent.actor, encoder, env, params, eval_key, args.eval_episodes)
        score = float(returns.mean())
        print(f"Update {step} actions {step * args.real_horizon * args.batch_size} "
              f"mean return {score:.6f}", flush=True)
        if epoch and not np.isfinite(np.asarray(losses)).all():
            raise FloatingPointError("Nonfinite actor-critic loss")
        if score > best:
            save_weights(agent, args.output)
            best = score


def main():
    parser = argparse.ArgumentParser(description="Train the tokenizer and QKNorm shortcut dynamics")
    parser.add_argument("--train", choices=("tokenizer", "dynamics", "both", "outcomes", "heads",
                                            "dynamics-outcomes", "actor-critic", "real-actor-critic"), default="dynamics",
                        help="Components to update (default: dynamics with a frozen tokenizer)")
    init = parser.add_mutually_exclusive_group()
    init.add_argument("--tokenizer-checkpoint", type=Path)
    init.add_argument("--model-checkpoint", type=Path, help="Initialize from full QKNorm model weights")
    parser.add_argument("--output", type=Path, required=True, help="New checkpoint directory")
    parser.add_argument("--game", default="Breakout")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, help="Defaults to 10,000 for tokenizer or 1,000 for dynamics")
    parser.add_argument("--batch-size", type=int, help="Defaults to 32 for real-game training and 64 otherwise")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--optimizer", choices=("adamw", "adam"),
                        help="Defaults to Adam for tokenizer or AdamW for dynamics and joint training")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--epochs", type=int, help="Defaults to 20 for tokenizer or 50 validation intervals for dynamics")
    dyn = parser.add_argument_group("dynamics and joint training")
    dyn.add_argument("--training-horizon", type=int, default=3, help="Autoregressive prediction steps for dynamics training")
    dyn.add_argument("--context", type=int, default=5)
    dyn.add_argument("--steps-per-epoch", "--val-every", type=int, default=2000)
    dyn.add_argument("--val-batches", type=int, default=16)
    dyn.add_argument("--patience", type=int, default=5, help="Validation checks without improvement (0 disables)")
    dyn.add_argument("--dynamics-weight", type=float, default=1.0)
    dyn.add_argument("--denoising-steps", type=int, choices=(1, 2, 4, 8, 16), default=DEFAULT_DENOISING_STEPS)
    policy = parser.add_argument_group("actor-critic")
    policy.add_argument("--imagination-horizon", type=int, default=5)
    policy.add_argument("--discount", type=float, default=0.99)
    policy.add_argument("--lambda-return", type=float, default=0.95)
    policy.add_argument("--entropy-scale", type=float, default=0.001)
    policy.add_argument("--eval-episodes", type=int, default=64)
    policy.add_argument("--real-horizon", type=int, default=5, help="Real transitions per environment per update")
    policy.add_argument("--policy-method", choices=("a2c", "ppo", "reinforce", "pmpo"),
                        help="Real-game training method (default: ppo)")
    policy.add_argument("--actor-learning-rate", type=float, help="Defaults to 1e-4 for PPO; other methods use --learning-rate")
    args = parser.parse_args()
    real_game = args.train == "real-actor-critic"
    if args.batch_size is None:
        args.batch_size = 32 if real_game else BATCH_SIZE
    if args.actor_learning_rate is not None and (not np.isfinite(args.actor_learning_rate) or args.actor_learning_rate <= 0):
        parser.error("--actor-learning-rate must be finite and positive")
    if args.train == "actor-critic" and args.policy_method == "ppo":
        if args.actor_learning_rate is None:
            args.actor_learning_rate = 1e-4
    elif real_game:
        args.policy_method = args.policy_method or "ppo"
        if args.actor_learning_rate is None:
            args.actor_learning_rate = 1e-4 if args.policy_method == "ppo" else args.learning_rate
    elif args.policy_method or args.actor_learning_rate is not None:
        parser.error("use --train real-actor-critic, or --train actor-critic --policy-method ppo for imagination")
    if args.train in ("actor-critic", "real-actor-critic"):
        if args.train == "actor-critic" and not args.model_checkpoint:
            parser.error("actor-critic requires --model-checkpoint with trained reward and terminal heads")
        if args.train == "real-actor-critic" and (not (args.model_checkpoint or args.tokenizer_checkpoint)
                                                  or args.real_horizon < 1):
            parser.error("real-actor-critic requires pretrained encoder weights and a positive real-horizon")
        if args.context < 2 or min(args.imagination_horizon, args.eval_episodes) < 1:
            parser.error("actor-critic requires context >= 2 and positive imagination horizon and eval episodes")
        if not (0 <= args.discount <= 1 and 0 <= args.lambda_return <= 1
                and np.isfinite(args.entropy_scale) and args.entropy_scale >= 0):
            parser.error("discount and lambda-return must be in [0,1] and entropy-scale finite and nonnegative")
    tokenizer_only = args.train == "tokenizer"
    if args.episodes is None:
        args.episodes = 10000 if tokenizer_only else 1000
    if args.epochs is None:
        args.epochs = EPOCHS if tokenizer_only else 50
    if args.train == "dynamics" and not (args.tokenizer_checkpoint or args.model_checkpoint):
        parser.error("dynamics training needs pretrained weights: supply --tokenizer-checkpoint or --model-checkpoint")
    if args.train in ("outcomes", "heads", "dynamics-outcomes") and not args.model_checkpoint:
        parser.error("outcome training requires --model-checkpoint with pretrained dynamics")
    if args.episodes < 2 or min(args.epochs, args.batch_size, args.context, args.training_horizon, args.steps_per_epoch, args.val_batches) < 1:
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
            restore_weights(model, args.model_checkpoint, require_outcomes=args.train == "actor-critic")
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
    elif args.train == "actor-critic":
        train_actor_critic(args, model, env, params, transform)
    elif args.train == "real-actor-critic":
        from policy_ablation import run
        run(args, model, env, params)
    else:
        train_dynamics(args, model, env, params, transform)


if __name__ == "__main__":
    main()
