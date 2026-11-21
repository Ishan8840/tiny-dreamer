from pathlib import Path
import os
import csv,json
import numpy as np
import jax
import jax.numpy as j
from flax import nnx
from train import WorldModel,Actor,Critic,restore_weights,make_env

base=Path(os.environ['CKPT'])
env,p=make_env(os.environ['GAME'])
assert jax.default_backend()=='gpu'
wm=WorldModel(nnx.Rngs(0),env.observation_space(p).shape[-1],env.action_space(p).n)
restore_weights(wm,base/'world/best.msgpack',require_outcomes=True)
rng=nnx.Rngs(0)
agent=nnx.Dict(actor=Actor(rng,env.action_space(p).n,256),critic=Critic(rng,256))
restore_weights(agent,base/'best.msgpack')
@jax.jit
def encode(obs):return wm.tokenizer.layers[0](obs.astype(j.float32))
from train import imagined_batch,lambda_returns
@jax.jit
def real(key,st,prev,z):
    key,ak,sk=jax.random.split(key,3)
    a=jax.random.categorical(ak,agent.actor(j.concatenate([prev,z],axis=-1)))
    obs,st,r,d,_=jax.vmap(env.step_env,(0,0,0,None))(jax.random.split(sk,len(a)),st,a,p)
    return key,st,z,encode(obs),a,r,d|jax.vmap(env.is_truncated,(0,None))(st,p)
@jax.jit
def real_target(key,st,prev,z):
    def step(carry,_):
        key,st,prev,z,a,r,d=real(*carry)
        v=agent.critic(j.concatenate([prev,z],axis=-1))[:,0]
        return (key,st,prev,z),(r,.99*(~d),v)
    _,(r,disc,v)=jax.lax.scan(step,(key,st,prev,z),None,length=5)
    return lambda_returns(r,disc,v,.95)[0]
@jax.jit
def imagined(key,h,a):
    keys=jax.vmap(lambda i:jax.random.fold_in(key,i))(j.arange(5))
    return imagined_batch(agent,wm.dynamics,h,a,keys,4,.99,.95)[2][0]
prior=Path('states.csv')
with prior.open() as f:
    refs={(int(r['seed']),int(r['lane']),int(r['step'])):r for r in csv.DictReader(f)}
rows=[]
for seed in range(41001,41005):
    reset,key=jax.random.split(jax.random.key(seed))
    obs,st=jax.vmap(env.reset,(0,None))(jax.random.split(reset,64),p)
    prev=z=encode(obs); h=j.repeat(z[:,None],5,axis=1);a=j.zeros((64,4),j.int32);alive=np.ones(64,bool)
    for t in range(65):
        if t in (0,32,64):
            # Replicate eight per state per batch, eight independent batches.
            st8,p8,z8,h8,a8=jax.tree.map(lambda x:j.repeat(x,8,axis=0),(st,prev,z,h,a))
            ir=[];rr=[]
            for b in range(8):
                k=jax.random.fold_in(jax.random.key(seed+210000),t*8+b)
                ki,kr=jax.random.split(k)
                ir.append(np.asarray(imagined(ki,h8,a8)).reshape(64,8))
                rr.append(np.asarray(real_target(kr,st8,p8,z8)).reshape(64,8))
            iv=np.concatenate(ir,1).mean(1);rv=np.concatenate(rr,1).mean(1)
            v=np.asarray(agent.critic(j.concatenate([prev,z],axis=-1)))[:,0]
            for i in np.flatnonzero(alive):
                ref=refs[seed,i,t];np.testing.assert_allclose(v[i],float(ref['critic']),atol=1e-5)
                rows.append((seed,i,t,v[i],iv[i],rv[i],float(ref['mc64'])))
        key,st,prev,z,act,_,done=real(key,st,prev,z)
        alive &= ~np.asarray(done)
        h=j.concatenate([h[:,1:],z[:,None]],1);a=j.concatenate([a[:,1:],act[:,None]],1)
assert len(rows)==len(refs)
with open('targets.csv','w') as f:
    w=csv.writer(f);w.writerow(['seed','lane','step','critic','imagined_lambda','real_lambda','real_mc64']);w.writerows(rows)
x=np.array(rows);summary={}
for label,group in [('full_context',x[x[:,2]>0]),('reset_padded',x[x[:,2]==0]),('all',x)]:
    c,i,r,g=group[:,3:].T
    def rm(a,b):return float(np.mean((a-b)**2)**.5)
    summary[label]=dict(n=len(c),means=group[:,3:].mean(0).tolist(),rmse_to_mc=[rm(s,g) for s in [c,i,r]],rmse_critic_imagined=rm(c,i),rmse_imagined_real_target=rm(i,r),correlation_to_mc=[float(np.corrcoef(s,g)[0,1]) if s.std()>1e-7 else None for s in [c,i,r]])
Path('targets-summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
