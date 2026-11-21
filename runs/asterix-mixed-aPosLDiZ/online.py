from pathlib import Path
import sys
import csv
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from train import (Actor, Critic, WorldModel, cache_latents, collect_episodes, make_env,
                   make_replay, imagined_ppo_update, dynamics_outcome_update, policy_returns,
                   save_weights, restore_weights)

@nnx.jit(static_argnames=("env",))
def collect(actor, encoder, env, params, carry):
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
        next_state, last, current = jax.lax.cond(done, restart, lambda _: (state, z, after), None)
        return (key, next_state, last, current), (z, after, a, r, done, term, prev)
    return jax.lax.scan(step, carry, None, length=1000)

def sample(episodes, rng, n):
    windows = []
    for index in rng.integers(len(episodes), size=n):
        ep = episodes[index]
        start = rng.integers(len(ep["actions"]) - 4)
        end = start + 5
        windows.append((ep["observations"][start:end+1], ep["actions"][start:end],
                        (ep["rewards"][end-1], ep["terminals"][end-1])))
    return windows

from train import imagined_batch, lambda_returns
from policy_ablation import learn

@nnx.jit
def mixed_update(agent, actor_opt, critic_opt, dynamics, history, past, key, block):
    keys=jax.vmap(lambda i:jax.random.fold_in(key,i))(jnp.arange(5))
    imagined=imagined_batch(agent,dynamics,history,past,keys,4,.99,.95)
    z,after,_,reward,done,_,prev=block
    starts=jax.random.randint(jax.random.fold_in(key,101),(80,),0,len(z))
    # Stop at episode ends. Bootstrap at the collection boundary if still alive.
    ix=starts[None,:]+jnp.arange(5)[:,None]
    safe=jnp.minimum(ix,len(z)-1)
    values=agent.critic(jnp.concatenate([z[safe],after[safe]],axis=-1))[...,0]
    discounts=.99*(~done[safe])
    def backup(future,items):
        idx,r,d,v=items
        future=jnp.where(idx==len(z)-1,v,future)
        target=r+d*(.05*v+.95*future)
        return target,target
    _,ret=jax.lax.scan(backup,values[-1],(ix,reward[safe],discounts,values),reverse=True)
    real_states=jnp.concatenate([prev[starts],z[starts]],axis=-1)
    lanes=jax.random.permutation(jax.random.fold_in(key,102),32)[:16]
    return learn(agent,actor_opt,critic_opt,*imagined,'ppo',.001,
                 (real_states,jax.lax.stop_gradient(ret[0]),lanes))[:2]

def run(seed):
    out = Path(f"seed-{seed}")
    out.mkdir()
    (out/"world").mkdir()
    env, params = make_env("Asterix")
    wm = WorldModel(nnx.Rngs(seed), env.observation_space(params).shape[-1], env.action_space(params).n)
    restore_weights(wm, Path("/workspace/asterix-natural5-esIDhLRY/pretrain/best.msgpack"), require_outcomes=True)
    assert wm.dynamics.reward_target == "categorical_points"
    assert jax.default_backend() == "gpu"
    enc = wm.tokenizer.layers[0]
    data, train, evaluate, init = jax.random.split(jax.random.key(seed), 4)
    episodes, _ = collect_episodes(env, params, data, 10000)
    offline = make_replay(episodes, 5, seed)
    cache_latents(enc, offline)
    rng = np.random.default_rng(seed)
    init = nnx.Rngs(init)
    agent = nnx.Dict(actor=Actor(init, env.action_space(params).n, 256), critic=Critic(init, 256))
    agent.critic.f3.bias[...] = jnp.zeros(255).at[0].set(12.)
    from model import return_bins
    assert float(return_bins()[0]) == 0 and float(return_bins()[-1]) == 100
    print("initial critic value",float(agent.critic(jnp.zeros((1,256)))[0,0]), flush=True)
    def opt(module, rate):
        return nnx.Optimizer(module, optax.adamw(rate, weight_decay=1e-4), wrt=nnx.Param)
    actor, critic, dynamics = opt(agent.actor, 1e-4), opt(agent.critic, 3e-4), opt(wm.dynamics, 1e-4)
    online = []
    pending = []
    collect_key, reset = jax.random.split(jax.random.key(52700+seed))
    obs, state = env.reset(reset, params)
    z = enc(obs.astype(jnp.float32)[None])[0]
    carry = (collect_key, state, z, z)
    best = -np.inf
    with (out/"metrics.csv").open("w", buffering=1) as f:
        log = csv.writer(f)
        log.writerow(["cycle","real_actions","imagined_transitions","world_updates","ppo_updates","world_outcome_loss","actor_loss","critic_ce","validation_return"])
        for cycle in range(1,101):
            carry, block = collect(agent.actor, enc, env, params, carry)
            for before, after, a, r, done, terminal in zip(*jax.device_get(block[:6])):
                pending.append((before, after, a, r, terminal))
                if done:
                    if len(pending) >= 5:
                        zs, next_zs, actions, rewards, terminals = zip(*pending)
                        online.append(dict(observations=np.stack((*zs,next_zs[-1])), actions=np.array(actions), rewards=np.array(rewards), terminals=np.array(terminals)))
                    pending = []
            def batch(n):
                rows = sample(offline.episodes, rng, n//2) + sample(online, rng, n//2)
                return tuple(np.stack(values) for values in zip(*rows))
            event = []
            for _ in range(100):
                zs, actions, targets = batch(64)
                train, key = jax.random.split(train)
                event.append(dynamics_outcome_update(wm.dynamics, dynamics, zs, actions, targets, key, 1.))
            print("cycle",cycle,"real",cycle*1000,"world BCE",float(np.mean(event)),flush=True)
            losses = []
            for step in range(1000):
                zs, actions, _ = batch(32)
                train, key = jax.random.split(train)
                losses.append(mixed_update(agent, actor, critic, wm.dynamics, zs[:,:-1], actions[:,:-1], key, block))
                if step == 0 and cycle == 1:
                    print("first PPO batch",np.asarray(losses[-1]),flush=True)
            al, cl = np.mean(losses, axis=0)
            if not np.isfinite([al,cl,np.mean(event)]).all():
                raise ValueError("nonfinite training metrics")
            score = ""
            if cycle % 2 == 0:
                score = float(policy_returns(agent.actor,enc,env,params,evaluate,64).mean())
                if score > best:
                    save_weights(agent,out)
                    save_weights(wm,out/"world")
                    best = score
            log.writerow([cycle,cycle*1000,cycle*160000,cycle*100,cycle*4000,np.mean(event),al,cl,score])
run(int(sys.argv[1]))
