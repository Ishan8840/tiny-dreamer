"""Policy and value learning from real or imagined transitions."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .data import encode_frames
from .imagination import imagine


def lambda_returns(rewards, discounts, next_values, trace_decay):
    def backup(future, transition):
        reward, discount, next_value = transition
        target = reward + discount * (
            (1 - trace_decay) * next_value + trace_decay * future
        )
        return target, target

    _, targets = jax.lax.scan(
        backup, next_values[-1], (rewards, discounts, next_values), reverse=True
    )
    return targets


def imagined_batch(
    agent,
    dynamics,
    history,
    past_actions,
    keys,
    denoising_steps,
    gamma,
    trace_decay,
    error_head=None,
    error_threshold=None,
    error_low=None,
):
    last_state, (states, actions, rewards, discounts, trusted) = imagine(
        agent,
        dynamics,
        history,
        past_actions,
        keys,
        denoising_steps,
        gamma,
        error_head,
        error_threshold,
        error_low,
    )
    bootstrap_states = jnp.concatenate((states[1:], last_state[None]), axis=0)
    next_values = agent.critic(bootstrap_states)[..., 0]
    if error_low is not None:
        rewards = trusted * rewards + (1 - trusted) * agent.critic(states)[..., 0]
        discounts = trusted * discounts
    elif error_head is not None:
        # The first rejected transition has return V(current), bootstrapping its predecessor.
        rewards = jnp.where(trusted, rewards, agent.critic(states)[..., 0])
        discounts = jnp.where(trusted, discounts, 0.0)
    targets = jax.lax.stop_gradient(
        lambda_returns(rewards, discounts, next_values, trace_decay)
    )
    weights = jnp.cumprod(
        jnp.concatenate((jnp.ones_like(discounts[:1]), discounts[:-1])), axis=0
    )
    return states, actions, targets, weights * trusted


@nnx.jit(static_argnames=("horizon", "denoising_steps"))
def actor_critic_update(
    agent,
    optimizer,
    dynamics,
    history,
    past_actions,
    key,
    horizon,
    denoising_steps,
    gamma,
    trace_decay,
    entropy_scale,
):
    batch = imagined_batch(
        agent,
        dynamics,
        history,
        past_actions,
        jax.random.split(key, horizon),
        denoising_steps,
        gamma,
        trace_decay,
    )
    return policy_update(agent, optimizer, *batch, entropy_scale)


@nnx.jit(static_argnames=("horizon", "denoising_steps"))
def imagined_ppo_update(
    agent,
    actor_opt,
    critic_opt,
    dynamics,
    history,
    past_actions,
    key,
    horizon,
    denoising_steps,
    gamma,
    trace_decay,
    entropy_scale,
):
    # Match the noise and sampling keys for shared steps when comparing horizons.
    keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(horizon))
    batch = imagined_batch(
        agent,
        dynamics,
        history,
        past_actions,
        keys,
        denoising_steps,
        gamma,
        trace_decay,
    )
    return learn(agent, actor_opt, critic_opt, *batch, "ppo", entropy_scale)[:2]


@nnx.jit(static_argnames=("horizon", "denoising_steps", "ppo"))
def adaptive_update(
    agent,
    optimizers,
    dynamics,
    history,
    past_actions,
    key,
    horizon,
    denoising_steps,
    gamma,
    trace_decay,
    entropy_scale,
    head,
    threshold,
    ppo,
    error_low=None,
):
    keys = (
        jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(horizon))
        if ppo
        else jax.random.split(key, horizon)
    )
    batch = imagined_batch(
        agent,
        dynamics,
        history,
        past_actions,
        keys,
        denoising_steps,
        gamma,
        trace_decay,
        head,
        threshold,
        error_low,
    )

    def update(agent, optimizers):
        if ppo:
            return learn(agent, *optimizers, *batch, "ppo", entropy_scale)[:2]
        return jnp.stack(policy_update(agent, optimizers[0], *batch, entropy_scale))

    # Skip the entire update (including Adam momentum) if every first step was rejected.
    return nnx.cond(
        jnp.any(batch[-1] > 0), update, lambda *_: jnp.zeros(2), agent, optimizers
    )


def policy_update(agent, optimizer, states, actions, targets, weights, entropy_scale):
    def loss_fn(net):
        logits = net.critic.logits(states)
        values = net.critic.value(logits)[..., 0]
        advantage = jax.lax.stop_gradient(targets - values)
        log_probs = jax.nn.log_softmax(net.actor(states))
        chosen = jnp.take_along_axis(log_probs, actions[..., None], axis=-1)[..., 0]
        entropy = -(jnp.exp(log_probs) * log_probs).sum(axis=-1)
        actor_loss = (
            -(weights * (chosen * advantage + entropy_scale * entropy)).sum()
            / weights.sum()
        )
        critic_loss = (
            weights * net.critic.return_loss(logits, targets)
        ).sum() / weights.sum()
        return actor_loss + critic_loss, (actor_loss, critic_loss)

    (_, losses), grads = nnx.value_and_grad(loss_fn, has_aux=True)(agent)
    optimizer.update(agent, grads)
    return losses


@nnx.jit(static_argnames=("env", "episodes"))
def policy_returns(actor, encoder, env, params, key, episodes):
    reset_key, rollout_key = jax.random.split(key)
    obs, env_states = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(reset_key, episodes), params
    )
    latent = encoder(obs.astype(jnp.float32))

    def advance(carry):
        key, env_states, previous, current, active, returns = carry
        key, action_key, step_key = jax.random.split(key, 3)
        actions = jax.random.categorical(
            action_key, actor(jnp.concatenate((previous, current), axis=-1))
        )
        obs, next_states, reward, done, _ = jax.vmap(
            env.step_env, in_axes=(0, 0, 0, None)
        )(jax.random.split(step_key, episodes), env_states, actions, params)
        done |= jax.vmap(env.is_truncated, in_axes=(0, None))(next_states, params)

        # Finished lanes keep their last valid state until the longest episode ends.
        def keep_active(new, old):
            mask = (active & ~done).reshape((episodes,) + (1,) * (new.ndim - 1))
            return jnp.where(mask, new, old)

        next_states = jax.tree.map(keep_active, next_states, env_states)
        return (
            key,
            next_states,
            current,
            encoder(obs.astype(jnp.float32)),
            active & ~done,
            returns + jnp.where(active, reward, 0.0),
        )

    initial = (
        rollout_key,
        env_states,
        latent,
        latent,
        jnp.ones(episodes, bool),
        jnp.zeros(episodes),
    )
    return jax.lax.while_loop(lambda carry: carry[4].any(), advance, initial)[-1]


def real_rollout(
    agent, encoder, env, params, carry, horizon, gamma, bootstrap_timeouts=True
):
    def transition(carry, _):
        key, env_states, previous, current = carry
        key, action_key, step_key, reset_key = jax.random.split(key, 4)
        state = jnp.concatenate((previous, current), axis=-1)
        action = jax.random.categorical(action_key, agent.actor(state))
        batch = action.shape[0]
        obs, next_env, reward, done, _ = jax.vmap(
            env.step_env, in_axes=(0, 0, 0, None)
        )(jax.random.split(step_key, batch), env_states, action, params)
        terminal = jax.vmap(env.is_terminated, in_axes=(0, None))(next_env, params)
        done |= jax.vmap(env.is_truncated, in_axes=(0, None))(next_env, params)
        next_z = encoder(obs.astype(jnp.float32))
        next_value = agent.critic(jnp.concatenate((current, next_z), axis=-1))[..., 0]
        # Bootstrap at time limits, but never propagate a return across an episode reset.
        if bootstrap_timeouts:
            reward = reward + gamma * (done & ~terminal) * next_value
        reset_obs, reset_env = jax.vmap(env.reset, in_axes=(0, None))(
            jax.random.split(reset_key, batch), params
        )

        def reset_lanes(reset, ongoing):
            mask = done.reshape((batch,) + (1,) * (ongoing.ndim - 1))
            return jnp.where(mask, reset, ongoing)

        next_env = jax.tree.map(reset_lanes, reset_env, next_env)
        reset_z = encoder(reset_obs.astype(jnp.float32))
        carry = (
            key,
            next_env,
            reset_lanes(reset_z, current),
            reset_lanes(reset_z, next_z),
        )
        return carry, (state, action, reward, gamma * ~done, next_value, done)

    return jax.lax.scan(transition, carry, xs=None, length=horizon)


def pmpo_loss(chosen, advantage, alpha=0.5):
    positive = advantage >= 0
    negative = ~positive

    def group_mean(mask):
        return jnp.where(mask, chosen, 0.0).sum() / jnp.maximum(mask.sum(), 1)

    return (1 - alpha) * group_mean(negative) - alpha * group_mean(positive)


def learn(
    agent, actor_opt, critic_opt, states, actions, targets, valid, method, entropy_scale
):
    targets = jax.lax.stop_gradient(targets)
    advantage = jax.lax.stop_gradient(targets - agent.critic(states)[..., 0])
    behavior = jax.lax.stop_gradient(jax.nn.log_softmax(agent.actor(states)))

    def select(lp):
        return jnp.take_along_axis(lp, actions[..., None], axis=-1)[..., 0]

    old_chosen = select(behavior)

    def mean(x):
        return (jnp.where(valid != 0, x, 0.0) * valid).sum() / valid.sum()

    def actor_loss(actor):
        lp = jax.nn.log_softmax(actor(states))
        chosen = select(lp)
        entropy = -(jnp.exp(lp) * lp).sum(axis=-1)
        if method == "pmpo":
            return pmpo_loss(chosen, advantage)
        if method == "ppo":
            ratio = jnp.exp(chosen - old_chosen)
            objective = jnp.minimum(
                ratio * advantage, jnp.clip(ratio, 0.8, 1.2) * advantage
            )
        else:
            objective = chosen * advantage
        return -mean(objective + entropy_scale * entropy)

    def critic_loss(critic):
        return mean(critic.return_loss(critic.logits(states), targets))

    for _ in range(4 if method == "ppo" else 1):
        loss, gradient = nnx.value_and_grad(actor_loss)(agent.actor)
        actor_opt.update(agent.actor, gradient)
        value_loss, gradient = nnx.value_and_grad(critic_loss)(agent.critic)
        critic_opt.update(agent.critic, gradient)
    entropy = mean(-(jnp.exp(behavior) * behavior).sum(-1))
    frequency = jnp.stack([mean(actions == i) for i in range(behavior.shape[-1])])
    return jnp.concatenate((jnp.stack((loss, value_loss, entropy)), frequency))


@nnx.jit(static_argnames=("env", "method", "horizon"))
def short_update(
    agent,
    actor_opt,
    critic_opt,
    encoder,
    env,
    params,
    carry,
    horizon,
    gamma,
    lam,
    entropy_scale,
    method,
):
    carry, (states, actions, rewards, discounts, next_values, _) = real_rollout(
        agent, encoder, env, params, carry, horizon, gamma
    )
    targets = lambda_returns(rewards, discounts, next_values, lam)
    metrics = learn(
        agent,
        actor_opt,
        critic_opt,
        states,
        actions,
        targets,
        jnp.ones_like(actions, dtype=bool),
        method,
        entropy_scale,
    )
    return carry, metrics


@nnx.jit(static_argnames=("env",))
def mc_chunk(agent, encoder, env, params, carry):
    carry, (states, actions, rewards, _, _, done) = real_rollout(
        agent, encoder, env, params, carry, 16, 1.0, bootstrap_timeouts=False
    )
    return carry, (states, actions, rewards, done)


@nnx.jit
def mc_update(
    agent, actor_opt, critic_opt, states, actions, rewards, valid, gamma, entropy_scale
):
    def backup(future, transition):
        reward, mask = transition
        value = jnp.where(mask, reward + gamma * future, 0.0)
        return value, value

    _, targets = jax.lax.scan(
        backup, jnp.zeros_like(rewards[0]), (rewards, valid), reverse=True
    )
    return learn(
        agent,
        actor_opt,
        critic_opt,
        states,
        actions,
        targets,
        valid,
        "reinforce",
        entropy_scale,
    )


def episode_update(
    agent,
    actor_opt,
    critic_opt,
    encoder,
    env,
    params,
    key,
    batch,
    remaining,
    gamma,
    entropy_scale,
):
    reset_key, act_key = jax.random.split(key)
    obs, env_states = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(reset_key, batch), params
    )
    z = encode_frames(encoder, obs)
    carry = (act_key, env_states, z, z)
    active = np.ones(batch, bool)
    chunks = []
    count = 0
    while active.any() and count < remaining:
        carry, (states, actions, rewards, done) = mc_chunk(
            agent, encoder, env, params, carry
        )
        done = np.asarray(done)
        masks = np.concatenate(
            (active[None], active[None] & ~np.logical_or.accumulate(done[:-1])), axis=0
        )
        masks &= np.cumsum(masks).reshape(masks.shape) <= remaining - count
        count += int(masks.sum())
        chunks.append((states, actions, rewards, masks))
        active &= ~done.any(axis=0)
    arrays = [jnp.concatenate(xs) for xs in zip(*chunks)]
    # Bound the number of JIT specializations for ragged episode lengths.
    size = 1 << (arrays[0].shape[0] - 1).bit_length()
    arrays = [
        jnp.pad(a, ((0, size - a.shape[0]),) + ((0, 0),) * (a.ndim - 1)) for a in arrays
    ]
    return count, mc_update(agent, actor_opt, critic_opt, *arrays, gamma, entropy_scale)
