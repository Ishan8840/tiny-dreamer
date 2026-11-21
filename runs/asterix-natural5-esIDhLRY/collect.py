from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def collect_episode(env, env_params, key, step_env):
    key, reset_key = jax.random.split(key)

    obs, state = env.reset(reset_key, env_params)

    observations = [np.asarray(obs)]
    actions = []
    rewards = []
    dones = []
    terminals = []

    done = False

    while not done:
        key, action_key, step_key = jax.random.split(key, 3)

        action = env.action_space(env_params).sample(action_key)

        obs, state, reward, done, info = step_env(
            step_key, state, action, env_params
        )
        terminal = env.is_terminated(state, env_params)
        done = done | env.is_truncated(state, env_params)

        observations.append(np.asarray(obs))
        actions.append(int(action))
        rewards.append(float(reward))
        dones.append(bool(done))
        terminals.append(bool(terminal))

    episode = {
        "observations": np.stack(observations),
        "actions": np.asarray(actions),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "terminals": np.asarray(terminals, dtype=np.bool_),
    }

    return episode, key


@partial(jax.jit, static_argnums=0)
def collect_batch_step(env, params, keys, states):
    def step(key, state):
        key, action_key, step_key = jax.random.split(key, 3)
        action = env.action_space(params).sample(action_key)
        obs, next_state, reward, done, _ = env.step_env(step_key, state, action, params)
        terminal = env.is_terminated(next_state, params)
        done = done | env.is_truncated(next_state, params)
        # Keep a valid input state for padding after this game's final transition.
        next_state = jax.tree.map(lambda old, new: jnp.where(done, old, new), state, next_state)
        return key, next_state, obs, action, reward, done, terminal

    return jax.vmap(step)(keys, states)


@partial(jax.jit, static_argnames=("env", "scan_steps"))
def collect_block(env, params, keys, states, scan_steps):
    def step(carry, _):
        keys, states = carry
        keys, states, obs, action, reward, done, terminal = collect_batch_step(env, params, keys, states)
        return (keys, states), (obs, action, reward, done, terminal)

    return jax.lax.scan(step, (keys, states), xs=None, length=scan_steps)


def collect_episodes(env, params, key, num_episodes, scan_steps=8):
    if num_episodes < 1:
        raise ValueError("num_episodes must be positive")
    if scan_steps < 1:
        raise ValueError("scan_steps must be positive")
    key, reset_key, rollout_key = jax.random.split(key, 3)
    reset_keys = jax.random.split(reset_key, num_episodes)
    obs, states = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, params)
    keys = jax.random.split(rollout_key, num_episodes)

    observations = [np.asarray(obs)[None, ...]]
    actions, rewards, dones, terminals = [], [], [], []
    lengths = np.zeros(num_episodes, dtype=int)
    active = np.ones(num_episodes, dtype=bool)
    while active.any():
        (keys, states), transitions = collect_block(env, params, keys, states, scan_steps)
        obs, action, reward, done, terminal = jax.device_get(transitions)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        terminals.append(terminal)
        finished = done.any(axis=0)
        # Include the first terminal transition and omit any later padding.
        valid_steps = np.where(finished, done.argmax(axis=0) + 1, scan_steps)
        lengths[active] += valid_steps[active]
        active &= ~finished

    observations = np.concatenate(observations)
    actions = np.concatenate(actions)
    rewards = np.concatenate(rewards)
    dones = np.concatenate(dones)
    terminals = np.concatenate(terminals)
    episodes = []
    for game, length in enumerate(lengths):
        episodes.append({
            "observations": observations[:length + 1, game].copy(),
            "actions": actions[:length, game].astype(int),
            "rewards": rewards[:length, game].astype(np.float32),
            "dones": dones[:length, game].copy(),
            "terminals": terminals[:length, game].copy(),
        })
    return episodes, key
