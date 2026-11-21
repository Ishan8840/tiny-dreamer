from pathlib import Path
import csv,sys
import numpy as np
import jax
import jax.numpy as j
from flax import nnx
import optax
from train import (make_env,restore_weights,save_weights,WorldModel,collect_episodes,
                   encode_frames,dynamics_outcome_update,denoise_next,evaluate_outcomes)
C=int(sys.argv[1]);assert C in (5,16)
from stratified import update
root=Path(".")
env,p=make_env("Asterix")
assert jax.default_backend()=="gpu"
old=WorldModel(nnx.Rngs(0),env.observation_space(p).shape[-1],env.action_space(p).n)
restore_weights(old,Path('starting-world.msgpack'),require_outcomes=True)
wm=WorldModel(nnx.Rngs(19),env.observation_space(p).shape[-1],env.action_space(p).n);wm.tokenizer=old.tokenizer
enc,dec=wm.tokenizer.layers

def replay(name,n,seed):
    loc=root/(name+'.npz')
    if not loc.exists():
        lengths=[];zs=[];actions=[];labels=[]
        for i in range(0,n,250):
            eps,_=collect_episodes(env,p,jax.random.fold_in(jax.random.key(seed),i),min(250,n-i))
            raw=np.concatenate([e['observations'] for e in eps])
            for k in range(0,len(raw),512):
                group=raw[k:k+512]
                pad=np.resize(group,(512,*group.shape[1:]))
                zs.append(np.asarray(encode_frames(enc,pad))[:len(group)].copy())
            for e in eps:
                lengths.append(len(e['actions']));actions.append(e['actions']);labels.append(np.stack([e['rewards'],e['terminals']],1))
        np.savez(loc,z=np.concatenate(zs),a=np.concatenate(actions),rt=np.concatenate(labels),L=lengths)
    with np.load(loc) as d:z,a,rt,L=[d[k] for k in ('z','a','rt','L')]
    E=np.repeat(np.arange(len(L)),L)
    frame=np.arange(len(E))-np.repeat(np.r_[0,np.cumsum(L)[:-1]],L)
    return (j.asarray(z),j.asarray(a,dtype=j.int32),j.asarray(rt,dtype=j.float32),j.asarray(E)),np.flatnonzero(frame>=C-1)
data,pop=replay('training',10000,147001)
val,vpop=replay('selection',1000,147002)
flags=np.asarray(data[2])[:,0]>0
pos=pop[flags[pop]];neg=pop[~flags[pop]]
print("eligible pickup endpoints",len(pos),"of",len(pop),"total actions",len(flags),"all pickups",flags.sum(),flush=True)
reward_rng=np.random.default_rng(71118)
@jax.jit
def batch(data,end):
    z,a,rt,E=data
    ix=end[:,None]+j.arange(1-C,1)
    iz=end[:,None]+E[end,None]+j.arange(1-C,2)
    return z[iz],a[ix],rt[end]
@nnx.jit
def score(net,decoder,z,a,rt,key):
    pred=denoise_next(net,z[:,:-1],a,jax.random.normal(key,z[:,-1].shape),4)
    loss,(_,_,mse)=evaluate_outcomes(net,z[:,:-1],a,rt)
    # Same frozen decoder and latent targets for both arms.
    target=(decoder(z[:,-1])>=0)
    bce=optax.sigmoid_binary_cross_entropy(decoder(pred),target).mean()
    return j.stack((bce,loss,mse))
opt=nnx.Optimizer(wm.dynamics,optax.adamw(3e-4,weight_decay=1e-4),wrt=nnx.Param)
rng=np.random.default_rng(71018)
vix=np.random.default_rng(933).choice(vpop,4096,replace=len(vpop)<4096).reshape(-1,64)
out=Path('pretrain');out.mkdir()
best=np.inf
with (out/'metrics.csv').open('w',buffering=1) as f:
    w=csv.writer(f);w.writerow(('step','next_frame_bce','outcome_loss','point_mse','selection_loss'))
    for st in range(1,60001):
        z,a,rt=batch(data,rng.choice(pop,64))
        ends=np.r_[reward_rng.choice(pos,16),reward_rng.choice(neg,48)]
        assert flags[ends].sum()==16
        weights=np.where(flags[ends],len(pos)/len(pop)/.25,len(neg)/len(pop)/.75).astype(np.float32)
        loss=update(wm.dynamics,opt,z,a,rt,*batch(data,ends),weights,jax.random.fold_in(jax.random.key(118),st))
        if st%2000==0:
            assert np.isfinite(float(loss))
            b,o,m=np.mean([score(wm.dynamics,dec,*batch(val,ids),jax.random.fold_in(jax.random.key(759),i)) for i,ids in enumerate(vix)],axis=0)
            rank=float(b+o);assert np.isfinite(rank)
            if rank<best:best=rank;save_weights(wm,out)
            w.writerow([st,b,o,m,rank]);print(C,st,rank,'best',best,flush=True)
