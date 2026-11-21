from pathlib import Path
import csv,sys
import numpy as np
import jax
import jax.numpy as j
from flax import nnx
import optax
from train import make_env,collect_episodes,WorldModel,restore_weights,denoise_next
root=Path(__file__).resolve().parent
out=root/sys.argv[1]
env,p=make_env('Asterix')
wm=WorldModel(nnx.Rngs(0),4,env.action_space(p).n)
rng=np.random.default_rng(68391)
frames=[];acts=[];rewards=[]
for seed in range(67301,67401):
    eps,_=collect_episodes(env,p,jax.random.key(seed),250)
    for ep in eps:
        limit=len(ep['actions'])-9
        if limit<0:continue
        s=rng.integers(limit+1)
        frames.append(ep['observations'][s:s+10]);acts.append(ep['actions'][s:s+9]);rewards.append(ep['rewards'][s+4:s+9])
        if len(acts)==1000:break
    if len(acts)==1000:break
assert len(frames)==1000
frames,acts,rewards=map(np.stack,[frames,acts,rewards])
@nnx.jit
def evaluate(model,o,a,r,ids):
    enc,dec=model.tokenizer.layers;hist=enc(o[:,:5].astype(j.float32))
    rows=[]
    for i in range(5):
        window=a[:,i:i+5]
        logits,_=model.dynamics.outcome_outputs(hist,window)
        reward=model.dynamics.expected_reward(logits)
        key=jax.random.fold_in(jax.random.key(9536),i)
        noise=jax.vmap(lambda x:jax.random.normal(jax.random.fold_in(key,x),(64,)))(ids)
        z=denoise_next(model.dynamics,hist,window,noise,4);logits=dec(z);target=o[:,5+i]
        exact=j.all((logits>=0)==target,axis=(1,2,3))
        bce=optax.sigmoid_binary_cross_entropy(logits,target).mean(axis=(1,2,3))
        rows.append(j.stack([exact,bce,(reward-r[:,i])**2],1))
        hist=j.concatenate([hist[:,1:],z[:,None]],axis=1)
    return j.stack(rows)
with (out/'heldout-rollouts.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['checkpoint','horizon','episodes','exact_frame','bce','reward_mse'])
    for tag,path in [('offline','pretrain/best.msgpack'),('online','seed-0/world/best.msgpack')]:
        restore_weights(wm,out/path,require_outcomes=True)
        results=[]
        for first in range(0,1000,64):
            ids=np.minimum(np.arange(first,first+64),999)
            results.append(np.asarray(evaluate(wm,frames[ids],acts[ids],rewards[ids],ids))[:,:min(64,1000-first)])
        for i,values in enumerate(np.concatenate(results,axis=1).mean(1)):
            w.writerow([tag,i+1,1000,*values])
