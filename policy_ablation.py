"""Matched on-policy Breakout experiments. Invoked by train.py."""
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from model import Actor, Critic, restore_weights


def pmpo_loss(chosen, advantage, alpha=0.5):
    positive = advantage >= 0
    negative = ~positive
    def group_mean(mask):
        return jnp.where(mask, chosen, 0.).sum() / jnp.maximum(mask.sum(), 1)
    return (1 - alpha) * group_mean(negative) - alpha * group_mean(positive)


def learn(agent, actor_opt, critic_opt, states, actions, targets, valid, method, entropy_scale):
    targets = jax.lax.stop_gradient(targets)
    advantage = jax.lax.stop_gradient(targets - agent.critic(states)[..., 0])
    behavior = jax.lax.stop_gradient(jax.nn.log_softmax(agent.actor(states)))
    def select(lp):
        return jnp.take_along_axis(lp, actions[..., None], axis=-1)[..., 0]
    old_chosen = select(behavior)
    def mean(x):
        return (jnp.where(valid != 0, x, 0.) * valid).sum() / valid.sum()
    def actor_loss(actor):
        lp = jax.nn.log_softmax(actor(states))
        chosen = select(lp)
        entropy = -(jnp.exp(lp) * lp).sum(axis=-1)
        if method == "pmpo":
            return pmpo_loss(chosen, advantage)
        if method == "ppo":
            ratio = jnp.exp(chosen - old_chosen)
            objective = jnp.minimum(ratio * advantage, jnp.clip(ratio, 0.8, 1.2) * advantage)
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
def short_update(agent, actor_opt, critic_opt, encoder, env, params, carry, horizon, gamma, lam, entropy_scale, method):
    from train import real_rollout, lambda_returns
    carry, (states, actions, rewards, discounts, next_values, _) = real_rollout(
        agent, encoder, env, params, carry, horizon, gamma)
    targets = lambda_returns(rewards, discounts, next_values, lam)
    metrics = learn(agent, actor_opt, critic_opt, states, actions, targets,
                    jnp.ones_like(actions, dtype=bool), method, entropy_scale)
    return carry, metrics


@nnx.jit(static_argnames=("env",))
def mc_chunk(agent, encoder, env, params, carry):
    from train import real_rollout
    carry, (states, actions, rewards, _, _, done) = real_rollout(
        agent, encoder, env, params, carry, 16, 1., bootstrap_timeouts=False)
    return carry, (states, actions, rewards, done)


@nnx.jit
def mc_update(agent, actor_opt, critic_opt, states, actions, rewards, valid, gamma, entropy_scale):
    def backup(future, transition):
        reward, mask = transition
        value = jnp.where(mask, reward + gamma * future, 0.)
        return value, value
    _, targets = jax.lax.scan(backup, jnp.zeros_like(rewards[0]), (rewards, valid), reverse=True)
    return learn(agent, actor_opt, critic_opt, states, actions, targets, valid, "reinforce", entropy_scale)


def episode_update(agent, actor_opt, critic_opt, encoder, env, params, key, batch, remaining, gamma, entropy_scale):
    from train import encode_frames
    reset_key, act_key = jax.random.split(key)
    obs, env_states = jax.vmap(env.reset, in_axes=(0,None))(jax.random.split(reset_key,batch),params)
    z = encode_frames(encoder, obs)
    carry = (act_key, env_states, z, z)
    active = np.ones(batch, bool)
    chunks = []
    count = 0
    while active.any() and count < remaining:
        carry, (states, actions, rewards, done) = mc_chunk(agent, encoder, env, params, carry)
        done = np.asarray(done)
        masks = np.concatenate((active[None], active[None] & ~np.logical_or.accumulate(done[:-1])), axis=0)
        masks &= (np.cumsum(masks).reshape(masks.shape) <= remaining-count)
        count += int(masks.sum())
        chunks.append((states, actions, rewards, masks))
        active &= ~done.any(axis=0)
    arrays = [jnp.concatenate(xs) for xs in zip(*chunks)]
    # Bound the number of JIT specializations for ragged episode lengths.
    size = 1 << (arrays[0].shape[0]-1).bit_length()
    arrays = [jnp.pad(a, ((0,size-a.shape[0]),) + ((0,0),)*(a.ndim-1)) for a in arrays]
    return count, mc_update(agent, actor_opt, critic_opt, *arrays, gamma, entropy_scale)


def run(args, model, env, params):
    from train import encode_frames, policy_returns, save_weights
    batch = args.batch_size
    method = args.policy_method
    reset_key, key, eval_key, init_key = jax.random.split(jax.random.key(args.seed),4)
    obs, env_states = jax.vmap(env.reset,in_axes=(0,None))(jax.random.split(reset_key,batch),params)
    encoder = model.tokenizer.layers[0]
    z = encode_frames(encoder, obs)
    carry = (key,env_states,z,z)
    rngs = nnx.Rngs(init_key)
    agent = nnx.Dict(actor=Actor(rngs,env.action_space(params).n,z.shape[-1]*2),critic=Critic(rngs,z.shape[-1]*2))
    def optimizer(net, lr):
        tx = optax.adamw(lr,weight_decay=args.weight_decay) if args.optimizer != "adam" else optax.adam(lr)
        return nnx.Optimizer(net,tx,wrt=nnx.Param)
    actor_opt = optimizer(agent.actor, args.actor_learning_rate)
    critic_opt = optimizer(agent.critic, args.learning_rate)
    interval = batch * args.real_horizon * args.steps_per_epoch
    actions = updates = 0
    best = -np.inf
    with (args.output/'validation.csv').open('w') as f:
        writer = csv.writer(f)
        action_names = ("noop", "left", "up", "right", "down", "fire")
        frequency_names = [action_names[int(i)] for i in env.action_set]
        writer.writerow(['actions','optimizer_updates','mean_return','actor_loss','critic_ce','entropy',*frequency_names])
        for epoch in range(args.epochs+1):
            metrics = []
            target = epoch * interval
            while actions < target:
                if method == "reinforce":
                    key, episode_key = jax.random.split(key)
                    count, m = episode_update(agent,actor_opt,critic_opt,encoder,env,params,episode_key,
                        batch,target-actions,args.discount,args.entropy_scale)
                else:
                    carry, m = short_update(agent,actor_opt,critic_opt,encoder,env,params,carry,
                        args.real_horizon,args.discount,args.lambda_return,args.entropy_scale,method)
                    count = batch * args.real_horizon
                actions += count
                updates += 4 if method == "ppo" else 1
                metrics.append(m)
            score = float(policy_returns(agent.actor,encoder,env,params,eval_key,args.eval_episodes).mean())
            summary = np.mean(metrics,axis=0) if metrics else ['']*(3 + len(frequency_names))
            if metrics and not np.isfinite(summary).all():
                raise FloatingPointError(f"Nonfinite metrics at {actions} actions")
            writer.writerow([actions,updates,score,*summary])
            f.flush()
            if score > best:
                best = score
                save_weights(agent,args.output)
            print(f"Actions {actions} updates {updates} validation {score:.4f} best {best:.4f}",flush=True)
    restore_weights(agent, args.output/'best.msgpack')
    with (args.output/'heldout.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['seed','lane','return'])
        for seed in range(41001,41005):
            scores = np.asarray(policy_returns(agent.actor,encoder,env,params,jax.random.key(seed),64))
            writer.writerows((seed,i,s) for i,s in enumerate(scores))
