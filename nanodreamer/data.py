"""MinAtar collection and episode-local replay sampling."""

from functools import partial

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


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

        obs, state, reward, done, _info = step_env(step_key, state, action, env_params)
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
        next_state = jax.tree.map(
            lambda old, new: jnp.where(done, old, new), state, next_state
        )
        return key, next_state, obs, action, reward, done, terminal

    return jax.vmap(step)(keys, states)


@partial(jax.jit, static_argnames=("env", "scan_steps"))
def collect_block(env, params, keys, states, scan_steps):
    def step(carry, _):
        keys, states = carry
        keys, states, obs, action, reward, done, terminal = collect_batch_step(
            env, params, keys, states
        )
        return (keys, states), (obs, action, reward, done, terminal)

    return jax.lax.scan(step, (keys, states), xs=None, length=scan_steps)


def collect_episodes(env, params, key, num_episodes, scan_steps=8, *, policy=None):
    if num_episodes < 1:
        raise ValueError("num_episodes must be positive")
    if scan_steps < 1:
        raise ValueError("scan_steps must be positive")
    key, reset_key, rollout_key = jax.random.split(key, 3)
    reset_keys = jax.random.split(reset_key, num_episodes)
    obs, states = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, params)
    keys = jax.random.split(rollout_key, num_episodes)

    block = collect_block
    if policy is not None:
        actor, encoder = policy
        initial = encoder(obs.astype(jnp.float32))
        states = (states, initial, initial)

        @partial(jax.jit, static_argnums=(0, 4))
        def block(env, params, keys, states, steps):
            def advance(carry, _):
                keys, (states, previous, current) = carry
                keys, act, move = jnp.moveaxis(
                    jax.vmap(lambda k: jax.random.split(k, 3))(keys), 1, 0
                )
                logits = actor(jnp.concatenate((previous, current), axis=-1))
                actions = jax.vmap(jax.random.categorical)(act, logits)
                obs, following, rewards, done, _ = jax.vmap(
                    env.step_env, in_axes=(0, 0, 0, None)
                )(move, states, actions, params)
                terminal = jax.vmap(env.is_terminated, in_axes=(0, None))(
                    following, params
                )
                done |= jax.vmap(env.is_truncated, in_axes=(0, None))(following, params)

                def keep_valid(old, new):
                    return jnp.where(
                        done.reshape((len(done),) + (1,) * (old.ndim - 1)), old, new
                    )

                following = jax.tree.map(keep_valid, states, following)
                latents = encoder(obs.astype(jnp.float32))
                return (keys, (following, current, latents)), (
                    obs,
                    actions,
                    rewards,
                    done,
                    terminal,
                )

            return jax.lax.scan(advance, (keys, states), None, length=steps)

    observations = [np.asarray(obs)[None, ...]]
    actions, rewards, dones, terminals = [], [], [], []
    lengths = np.zeros(num_episodes, dtype=int)
    active = np.ones(num_episodes, dtype=bool)
    while active.any():
        (keys, states), transitions = block(env, params, keys, states, scan_steps)
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
        episodes.append(
            {
                "observations": observations[: length + 1, game].copy(),
                "actions": actions[:length, game].astype(int),
                "rewards": rewards[:length, game].astype(np.float32),
                "dones": dones[:length, game].copy(),
                "terminals": terminals[:length, game].copy(),
            }
        )
    return episodes, key


def make_env(game="Breakout"):
    env, env_params = gymnax.make(f"{game}-MinAtar")
    return env, env_params


class ReplayBuffer:
    def __init__(self, capacity=1000, rng=None):
        self.capacity = capacity
        self.episodes = []
        self.rng = rng

    def add_episode(self, episode):
        self.episodes.append(episode)

        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)

    def sample_sequence(self, seq_len=16):
        valid_episodes = [ep for ep in self.episodes if len(ep["actions"]) >= seq_len]

        if not valid_episodes:
            raise ValueError("No episodes long enough to sample")

        return self._sample_window(valid_episodes, seq_len)

    def _sample_window(self, valid_episodes, seq_len):
        randint = np.random.randint if self.rng is None else self.rng.integers
        episode = valid_episodes[randint(len(valid_episodes))]

        num_steps = len(episode["actions"])
        start = randint(0, num_steps - seq_len + 1)
        end = start + seq_len

        sequence = {
            "observations": episode["observations"][start : end + 1],
            "actions": episode["actions"][start:end],
            "rewards": episode["rewards"][start:end],
            "dones": episode["dones"][start:end],
        }
        if "terminals" in episode:
            sequence["terminals"] = episode["terminals"][start:end]
        return sequence

    def sample_batch(self, batch_size=32, seq_len=16):
        eligible = [ep for ep in self.episodes if len(ep["actions"]) >= seq_len]
        if not eligible:
            raise ValueError("No episodes long enough to sample")
        sequences = [self._sample_window(eligible, seq_len) for _ in range(batch_size)]

        return {
            key: np.stack([seq[key] for seq in sequences])
            for key in sequences[0]
            if all(key in seq for seq in sequences)
        }


def episode_partitions(episodes, seed=0):
    if len(episodes) < 2:
        raise ValueError("A train/validation split requires at least two episodes")
    order = np.random.default_rng(seed).permutation(len(episodes))
    val_count = max(1, round(0.1 * len(episodes)))
    return [episodes[i] for i in order[val_count:]], [
        episodes[i] for i in order[:val_count]
    ]


def split_episodes(episodes, seed=0):
    train_eps, val_eps = episode_partitions(episodes, seed)
    train_obs = np.concatenate([ep["observations"] for ep in train_eps])
    val_obs = np.concatenate([ep["observations"] for ep in val_eps])
    return train_obs, val_obs


@nnx.jit
def encode_frames(encoder, observations):
    return encoder(observations.astype(jnp.float32))


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
    encoded = np.concatenate(
        [
            np.asarray(encode_frames(encoder, observations[i : i + 512]))
            for i in range(0, len(observations), 512)
        ]
    )
    offset = 0
    for ep in replay.episodes:
        end = offset + len(ep["observations"])
        ep["observations"] = encoded[offset:end]
        offset = end


@nnx.jit(static_argnames=("env", "steps"))
def collect_online_block(actor, encoder, env, params, carry, steps):
    def step(carry, _):
        key, state, prev, z = carry
        key, act, move, reset = jax.random.split(key, 4)
        a = jax.random.categorical(act, actor(jnp.concatenate((prev, z))[None])[0])
        obs, state, r, done, _ = env.step_env(move, state, a, params)
        term = env.is_terminated(state, params)
        done |= env.is_truncated(state, params)
        after = encoder(obs.astype(jnp.float32)[None])[0]

        def restart(_):
            obs, state = env.reset(reset, params)
            start = encoder(obs.astype(jnp.float32)[None])[0]
            return state, start, start

        next_state, last, current = jax.lax.cond(
            done, restart, lambda _: (state, z, after), None
        )
        return (key, next_state, last, current), (z, after, a, r, done, term)

    return jax.lax.scan(step, carry, None, length=steps)


class RewardSampler:
    """Sample reward endpoints by class and correct to episode-uniform replay."""

    def __init__(self, episodes, context):
        self.episodes = [ep for ep in episodes if len(ep["actions"]) >= context]
        self.context = context
        self.pools = []
        for positive in (False, True):
            endpoints, masses = [], []
            for episode in self.episodes:
                labels = episode["rewards"][context - 1 :] > 0
                ends = np.flatnonzero(labels == positive) + context - 1
                endpoints.append(ends)
                masses.append(len(ends) / len(labels))
            # Replay chooses an episode uniformly, then a window within it.
            cumulative = np.cumsum(masses)
            prevalence = cumulative[-1] / len(self.episodes) if len(cumulative) else 0.0
            self.pools.append((endpoints, cumulative, prevalence))

    def sample(self, rng, size, positive_fraction):
        histories, actions, targets, weights = [], [], [], []
        positive_count = int(np.clip(round(size * positive_fraction), 1, size - 1))
        counts = [size - positive_count, positive_count]
        if self.pools[0][2] == 0:
            counts = [0, size]
        elif self.pools[1][2] == 0:
            counts = [size, 0]
        for (endpoints, cumulative, prevalence), n in zip(self.pools, counts):
            if n == 0:
                continue
            for episode_index in cumulative.searchsorted(
                rng.random(n) * cumulative[-1], side="right"
            ):
                end = rng.choice(endpoints[episode_index])
                episode = self.episodes[episode_index]
                window = slice(end - self.context + 1, end + 1)
                histories.append(episode["observations"][window])
                actions.append(episode["actions"][window])
                targets.append(episode["rewards"][end])
            weights.extend([prevalence * size / n] * n)
        return (
            np.stack(histories),
            np.stack(actions),
            np.array(targets),
            np.array(weights, np.float32),
        )
