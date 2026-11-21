from pathlib import Path
import csv
import sys
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from train import WorldModel, restore_weights, denoise_next, make_env
from evaluate_imagined_outcomes import archived_starts

root = Path(__file__).resolve().parent
context = int(sys.argv[1])
arm = root / f'c{context}'
with np.load(root / 'windows.npz') as d:
    obs, actions = d['observations'], d['actions']
labels = root / 'matched-rewards.npy'
if not labels.exists():
    env, params = make_env()
    rewards = [ep['rewards'][s+15:s+35] for _, _, s, ep in archived_starts(env, params, root)]
    np.save(labels, np.stack(rewards))
rewards = np.load(labels)

@nnx.jit
def predict(wm, frames, recorded, ids):
    encoder, decoder = wm.tokenizer.layers
    init = encoder(frames[:, 16-context:16].astype(jnp.float32))
    def advance(history, h):
        a = jax.lax.dynamic_slice_in_dim(recorded, 16-context+h, context, axis=1)
        point_logits, _ = wm.dynamics.outcome_outputs(history, a)
        key = jax.random.fold_in(jax.random.key(901), h)
        noise = jax.vmap(lambda i: jax.random.normal(jax.random.fold_in(key,i),(history.shape[-1],)))(ids)
        z = denoise_next(wm.dynamics, history, a, noise, 4)
        return jnp.concatenate((history[:,1:],z[:,None]),axis=1), (decoder(z),wm.dynamics.expected_reward(point_logits))
    return jax.lax.scan(advance,init,jnp.arange(20))[1]

with (arm/'rollout-metrics.csv').open('w') as file:
    writer = csv.writer(file)
    writer.writerow(['checkpoint','horizon','episodes','exact_frame_rate','mean_ball_error','perfect_rollout_rate','reward_mse'])
    for name, checkpoint in [('offline','pretrain/best.msgpack'),('online','seed-0/world/best.msgpack')]:
        wm = WorldModel(nnx.Rngs(0),4,3)
        restore_weights(wm,arm/checkpoint,require_outcomes=True)
        results = []
        for start in range(0,len(obs),64):
            ids = np.minimum(np.arange(start,start+64),len(obs)-1)
            n = min(64,len(obs)-start)
            logits, points = predict(wm,obs[ids],actions[ids],ids)
            logits = np.asarray(logits)[:,:n]
            target = obs[ids[:n],16:].swapaxes(0,1)
            exact = ((logits>=0)==target).all(axis=(2,3,4))
            pos = logits[...,1].reshape(20,n,100).argmax(-1)
            true = target[...,1].reshape(20,n,100).argmax(-1)
            error = np.hypot(pos//10-true//10,pos%10-true%10)
            point_error = np.square(np.asarray(points).reshape(20,64)[:,:n]-rewards[ids[:n]].T)
            results.append(np.stack([exact,error,np.logical_and.accumulate(exact,axis=0),point_error]))
        scores = np.concatenate(results,axis=2).mean(axis=2)
        for h in range(20):
            writer.writerow([name,h+1,len(obs),*scores[:,h]])
