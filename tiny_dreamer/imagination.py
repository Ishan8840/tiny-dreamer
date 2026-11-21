"""Roll a policy forward from a real replay context using the learned dynamics."""

import jax
import jax.numpy as jnp

from .model import denoise_next


def policy_state(history):
    return jnp.concatenate((history[:, -2], history[:, -1]), axis=-1)


def imagine(
    agent,
    dynamics,
    history,
    past_actions,
    keys,
    denoising_steps,
    gamma,
    error_head=None,
    error_threshold=None,
    error_low=None,
):
    def imagine(carry, step):
        step_key, depth = step
        history, past_actions = carry
        action_key, noise_key = jax.random.split(step_key)

        state = policy_state(history)
        action = jax.random.categorical(action_key, agent.actor(state)).astype(
            past_actions.dtype
        )
        actions = jnp.concatenate((past_actions, action[:, None]), axis=1)

        features = dynamics.outcome_features(history, actions)
        reward = dynamics.reward_head(features)
        terminal = dynamics.terminal_head(features)[:, 0]

        noise = jax.random.normal(noise_key, history[:, -1].shape)
        next_z = denoise_next(dynamics, history, actions, noise, denoising_steps)
        next_history = jnp.concatenate((history[:, 1:], next_z[:, None]), axis=1)
        transition = (
            state,
            action,
            dynamics.expected_reward(reward),
            gamma * jax.nn.sigmoid(-terminal),
        )

        trust = jnp.ones_like(action, dtype=bool)
        if error_head is not None:
            error = error_head(features, next_z, depth)
            if error_low is None:
                trust = error <= error_threshold
            else:
                scale = (error_threshold - error_low) / jnp.log(9.0)
                trust = jax.nn.sigmoid((error_threshold - error) / scale)

        return (next_history, actions[:, 1:]), (*transition, trust)

    (last_history, _), transitions = jax.lax.scan(
        imagine, (history, past_actions), (keys, jnp.arange(1, len(keys) + 1))
    )
    return policy_state(last_history), transitions
