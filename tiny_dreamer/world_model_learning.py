"""Reconstruction, shortcut, reward and continuation objectives on replay sequences."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from .config import TrainConfig
from .model import OUTCOME_COMPONENTS, denoise_next, predict_next

DEFAULT_DENOISING_STEPS = TrainConfig.denoising_steps
D_MIN = 1 / 16


def reconstruction_loss(logits, observations):
    loss = optax.sigmoid_binary_cross_entropy(logits, observations.astype(jnp.float32))

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
    history_signal=None,
):
    half_step = step_size / 2

    b_1 = (
        model(
            history_latents,
            history_actions,
            noisy_latent,
            noise_level,
            half_step,
            history_signal,
        )
        - noisy_latent
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
            history_signal,
        )
        - z_prime
    ) / (1 - t_prime)[:, None]

    target_velocity = jax.lax.stop_gradient((b_1 + b_2) / 2)

    predicted_velocity = (predicted_latent - noisy_latent) / (1 - noise_level)[:, None]

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

    step_size = D_MIN * (2.0**levels)

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

    noisy_latent = t[:, None] * target_latent + (1.0 - t[:, None]) * noise

    return noisy_latent, t, d


@nnx.jit
def train_step(model, optimizer, batch):
    loss, grads = nnx.value_and_grad(batch_loss)(model, batch)
    optimizer.update(model, grads)
    return loss


def rollout_steps(
    dynamics, latents, actions, key, horizon, denoising_steps, *, inference=False
):
    context = actions.shape[1] - horizon + 1
    if context < 1 or horizon < 1 or latents.shape[1] != actions.shape[1] + 1:
        raise ValueError(
            "A rollout needs context + horizon frames and one fewer action"
        )

    history = latents[:, :context]
    for step in range(horizon):
        window = actions[:, step : step + context]
        target = latents[:, context + step]
        yield history, window, target

        if step + 1 < horizon:
            step_key = jax.random.fold_in(key, step)
            if inference:
                prediction = predict_next(
                    dynamics, history, window, step_key, denoising_steps, step
                )
            else:
                noise = jax.random.normal(step_key, target.shape)
                prediction = denoise_next(
                    dynamics, history, window, noise, denoising_steps
                )
            if dynamics.detach_rollout:
                prediction = jax.lax.stop_gradient(prediction)
            history = jnp.concatenate((history[:, 1:], prediction[:, None]), axis=1)


def noise_history(history, key):
    """Draw an independent flow interpolation for every batch item and frame."""
    level_key, noise_key = jax.random.split(key)
    signal = jax.random.randint(level_key, history.shape[:-1], 0, 17) / 16.0
    noise = jax.random.normal(noise_key, history.shape, dtype=history.dtype)
    mixed = signal[..., None] * history + (1.0 - signal[..., None]) * noise
    return mixed, signal


def rollout_loss(
    dynamics, latents, actions, key, horizon, denoising_steps, outcomes=None
):
    shortcut = event = 0.0
    rollout_key, loss_key = jax.random.split(key)
    for step, (history, window, target) in enumerate(
        rollout_steps(dynamics, latents, actions, rollout_key, horizon, denoising_steps)
    ):
        target = jax.lax.stop_gradient(target)
        noise_key = key if horizon == 1 else jax.random.fold_in(loss_key, step)
        noisy, t, d = sample_training_inputs(target, noise_key)
        denoising_history = history
        history_signal = None
        if dynamics.diffusion_forcing:
            # Keep the baseline target draws and autoregressive rollout identical.
            history_key = jax.random.fold_in(jax.random.fold_in(key, 0xDF), step)
            denoising_history, history_signal = noise_history(history, history_key)

        predicted = dynamics(denoising_history, window, noisy, t, d, history_signal)
        shortcut += dynamics_loss(
            denoising_history,
            window,
            noisy,
            t,
            d,
            predicted,
            target,
            dynamics,
            D_MIN,
            history_signal,
        )

        if outcomes is not None:
            labels = outcomes[:, step] if outcomes.ndim == 3 else outcomes
            event += outcome_metrics(dynamics, history, window, labels)[0]
    return shortcut / horizon, event / horizon


def shortcut_loss(
    dynamics, latents, actions, key, horizon=1, denoising_steps=DEFAULT_DENOISING_STEPS
):
    return rollout_loss(dynamics, latents, actions, key, horizon, denoising_steps)[0]


def joint_loss(
    model,
    obs,
    actions,
    key,
    dynamics_weight,
    horizon=1,
    denoising_steps=DEFAULT_DENOISING_STEPS,
):
    encoder, decoder = model.tokenizer.layers
    latents = encoder(obs.astype(jnp.float32))
    reconstruction = reconstruction_loss(decoder(latents), obs)

    prediction = shortcut_loss(
        model.dynamics, latents, actions, key, horizon, denoising_steps
    )
    return reconstruction + dynamics_weight * prediction, (reconstruction, prediction)


@nnx.jit(static_argnames=("horizon", "denoising_steps"))
def update_step(
    model,
    optimizer,
    obs,
    actions,
    key,
    dynamics_weight,
    horizon=1,
    denoising_steps=DEFAULT_DENOISING_STEPS,
):
    (loss, (rec, dyn)), grads = nnx.value_and_grad(joint_loss, has_aux=True)(
        model, obs, actions, key, dynamics_weight, horizon, denoising_steps
    )
    optimizer.update(model, grads)
    return loss, rec, dyn


@nnx.jit(static_argnames=("horizon", "denoising_steps"))
def frozen_update_step(
    dynamics,
    optimizer,
    latents,
    actions,
    key,
    dynamics_weight,
    horizon=1,
    denoising_steps=DEFAULT_DENOISING_STEPS,
):
    def loss_fn(net):
        return dynamics_weight * shortcut_loss(
            net, latents, actions, key, horizon, denoising_steps
        )

    loss, grads = nnx.value_and_grad(loss_fn)(dynamics)
    optimizer.update(dynamics, grads)
    return loss


@nnx.jit(static_argnames=("denoising_steps", "horizon"))
def evaluate(
    model,
    obs,
    actions,
    key,
    dynamics_weight,
    denoising_steps=DEFAULT_DENOISING_STEPS,
    horizon=1,
):
    loss_key, rollout_key = jax.random.split(key)
    loss, (rec, dyn) = joint_loss(
        model, obs, actions, loss_key, dynamics_weight, horizon, denoising_steps
    )
    encoder, decoder = model.tokenizer.layers
    context = actions.shape[1] - horizon + 1

    history = encoder(obs[:, :context].astype(jnp.float32))
    bce = 0.0
    for step in range(horizon):
        noise_key = (
            rollout_key if horizon == 1 else jax.random.fold_in(rollout_key, step)
        )
        prediction = predict_next(
            model.dynamics,
            history,
            actions[:, step : step + context],
            noise_key,
            denoising_steps,
            step,
        )
        bce += reconstruction_loss(decoder(prediction), obs[:, context + step])
        history = jnp.concatenate((history[:, 1:], prediction[:, None]), axis=1)
    return loss, rec, dyn, bce / horizon


def is_outcome_param(path, variable):
    return isinstance(variable, nnx.Param) and path[0] in OUTCOME_COMPONENTS


def outcome_metrics(dynamics, history, actions, targets):
    reward, terminal = dynamics.outcome_outputs(history, actions)
    points = targets[:, 0]

    reward_ce = optax.softmax_cross_entropy_with_integer_labels(
        reward, points.astype(jnp.int32)
    )
    terminal_bce = optax.sigmoid_binary_cross_entropy(terminal, targets[:, 1])
    losses = jnp.stack((reward_ce.mean(), terminal_bce.mean()))

    predictions = jnp.stack(
        (dynamics.scoring_logit(reward) >= 0, terminal >= 0), axis=-1
    )
    events = targets > 0
    counts = jnp.stack(
        (
            (predictions & events).sum(axis=0),
            predictions.sum(axis=0),
            events.sum(axis=0),
        )
    )

    reward_mse = jnp.mean(jnp.square(dynamics.expected_reward(reward) - points))
    return losses.sum(), (losses, counts, reward_mse)


def is_head_param(path, variable):
    return isinstance(variable, nnx.Param) and path[0] in (
        "reward_head",
        "terminal_head",
    )


@nnx.jit(static_argnames=("param_filter",))
def outcome_update(
    dynamics, optimizer, history, actions, targets, param_filter=is_outcome_param
):
    (loss, _), grads = nnx.value_and_grad(
        outcome_metrics, argnums=nnx.DiffState(0, param_filter), has_aux=True
    )(dynamics, history, actions, targets)
    optimizer.update(dynamics, grads)
    return loss


evaluate_outcomes = nnx.jit(outcome_metrics)


@nnx.jit(static_argnames=("horizon", "denoising_steps"))
def dynamics_outcome_update(
    dynamics,
    optimizer,
    latents,
    actions,
    targets,
    key,
    weight,
    horizon=1,
    denoising_steps=DEFAULT_DENOISING_STEPS,
    reward_batch=None,
):
    def objective(net):
        if reward_batch is not None:
            history, reward_actions, points, weights = reward_batch
            reward, _ = net.outcome_outputs(history, reward_actions)
            _, terminal = net.outcome_outputs(latents[:, :-1], actions)

            reward_ce = optax.softmax_cross_entropy_with_integer_labels(
                reward, points.astype(jnp.int32)
            )
            event_loss = (
                weights * reward_ce
            ).mean() + optax.sigmoid_binary_cross_entropy(
                terminal, targets[:, 0, 1]
            ).mean()

            latent_loss = shortcut_loss(net, latents, actions, key)
            return event_loss + weight * latent_loss, event_loss

        shortcut, event_loss = rollout_loss(
            net, latents, actions, key, horizon, denoising_steps, targets
        )
        return event_loss + weight * shortcut, event_loss

    (_, event_loss), grads = nnx.value_and_grad(objective, has_aux=True)(dynamics)
    optimizer.update(dynamics, grads)
    return event_loss


@nnx.jit(static_argnames=("horizon", "denoising_steps"))
def evaluate_rollout_outcomes(
    dynamics, latents, actions, targets, key, horizon, denoising_steps
):
    metrics = [
        outcome_metrics(dynamics, history, window, targets[:, step])[1]
        for step, (history, window, _) in enumerate(
            rollout_steps(
                dynamics,
                latents,
                actions,
                key,
                horizon,
                denoising_steps,
                inference=True,
            )
        )
    ]
    losses, counts, mses = zip(*metrics)
    return jnp.mean(jnp.stack(losses), axis=0), sum(counts), jnp.mean(jnp.stack(mses))


def outcome_validation(
    dynamics,
    episodes,
    context,
    batch_size,
    horizon=1,
    denoising_steps=DEFAULT_DENOISING_STEPS,
):
    losses = np.zeros(2)
    squared_error = 0.0
    counts = np.zeros((3, 2), dtype=np.int64)

    length = context + horizon - 1
    windows = [
        (ep, start)
        for ep in episodes
        for start in range(len(ep["actions"]) - length + 1)
    ]

    for offset in range(0, len(windows), batch_size):
        group = windows[offset : offset + batch_size]
        latents = np.stack([ep["observations"][s : s + length + 1] for ep, s in group])
        actions = np.stack([ep["actions"][s : s + length] for ep, s in group])
        targets = np.stack(
            [
                np.stack(
                    (
                        ep["rewards"][s + context - 1 : s + length],
                        ep["terminals"][s + context - 1 : s + length],
                    ),
                    axis=-1,
                )
                for ep, s in group
            ]
        ).astype(np.float32)

        batch_losses, batch_counts, mse = evaluate_rollout_outcomes(
            dynamics,
            latents,
            actions,
            targets,
            jax.random.fold_in(jax.random.key(0), offset),
            horizon,
            denoising_steps,
        )

        losses += np.asarray(batch_losses) * len(group)
        squared_error += float(mse) * len(group)
        counts += np.asarray(batch_counts)
    return losses / len(windows), counts, len(windows), squared_error / len(windows)
