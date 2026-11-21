from pathlib import Path
import os
GAME=os.environ["MINATAR_GAME"]
import csv,sys
import numpy as np
import jax
import jax.numpy as j
from flax import nnx
import optax
from train import (make_env,restore_weights,save_weights,WorldModel,collect_episodes,
                   encode_frames,dynamics_outcome_update,denoise_next,evaluate_outcomes)
C=int(sys.argv[1]);assert C in (5,16)
root=Path("/workspace/asterix-pickup25-rxfmveFt")
env,p=make_env(GAME)
assert jax.default_backend()=="gpu"
wm=WorldModel(nnx.Rngs(19),env.observation_space(p).shape[-1],env.action_space(p).n)
restore_weights(wm.tokenizer,Path('tokenizer.msgpack'))
enc,dec=wm.tokenizer.layers
from dataset import dataset
def replay(name,n,seed):
    z,a,rt,L=dataset(env,p,enc,jax.random.key(seed),n)
    E=np.repeat(np.arange(len(L),dtype=np.int32),L)
    frame=np.arange(len(a))-np.repeat(np.r_[0,np.cumsum(L)[:-1]],L)
    return (z,a,rt,E),np.flatnonzero(frame>=C-1)
data,pop=replay('training',10000,147001)
val,vpop=replay('selection',1000,147002)
flags=np.asarray(data[2])[:,0]>0
pos=pop[flags[pop]];neg=pop[~flags[pop]]
print("eligible positive-reward endpoints",len(pos),"of",len(pop),"total actions",len(flags),"positive rewards",flags.sum(),flush=True)
def batch(data,end):
    z,a,rt,E=data
    ix=end[:,None]+np.arange(1-C,1)
    iz=end[:,None]+E[end,None]+np.arange(1-C,2)
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
        loss=dynamics_outcome_update(wm.dynamics,opt,z,a,rt,jax.random.fold_in(jax.random.key(118),st),1.)
        if st%2000==0:
            assert np.isfinite(float(loss))
            b,o,m=np.mean([score(wm.dynamics,dec,*batch(val,ids),jax.random.fold_in(jax.random.key(759),i)) for i,ids in enumerate(vix)],axis=0)
            rank=float(b+o);assert np.isfinite(rank)
            if rank<best:best=rank;save_weights(wm,out)
            w.writerow([st,b,o,m,rank]);print(C,st,rank,'best',best,flush=True)
