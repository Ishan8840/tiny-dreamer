from pathlib import Path
import csv,json
import numpy as np
import jax
import jax.numpy as j
from flax import nnx
from train import WorldModel,Actor,Critic,restore_weights,make_env

base=Path('/workspace/asterix-natural5-esIDhLRY/seed-0')
env,p=make_env('Asterix')
assert jax.default_backend()=='gpu'
wm=WorldModel(nnx.Rngs(0),4,env.action_space(p).n)
restore_weights(wm,base/'world/best.msgpack',require_outcomes=True)
rng=nnx.Rngs(0)
agent=nnx.Dict(actor=Actor(rng,env.action_space(p).n,256),critic=Critic(rng,256))
restore_weights(agent,base/'best.msgpack')
@jax.jit
def encode(obs):return wm.tokenizer.layers[0](obs.astype(j.float32))
def advance(key,st,prev,z):
    key,ak,sk=jax.random.split(key,3)
    a=jax.random.categorical(ak,agent.actor(j.concatenate([prev,z],axis=-1)))
    o,st,r,d,_=jax.vmap(env.step_env,(0,0,0,None))(jax.random.split(sk,len(a)),st,a,p)
    return key,st,z,encode(o),r,d|jax.vmap(env.is_truncated,(0,None))(st,p)
prefix=jax.jit(advance)
@jax.jit
def simulate(key,st,prev,z):
    # 64 independent continuations of each of 64 real states.
    st,prev,z=jax.tree.map(lambda x:j.repeat(x,64,axis=0),(st,prev,z))
    def step(carry):
        key,st,prev,z,active,ret,discount=carry
        key,st,prev,z,r,d=advance(key,st,prev,z)
        return key,st,prev,z,active & ~d,ret+j.where(active,r*discount,0.),discount*.99
    carry=(key,st,prev,z,j.ones(4096,bool),j.zeros(4096),j.float32(1.))
    return jax.lax.while_loop(lambda c:c[4].any(),step,carry)[5].reshape(64,64)
rows=[]
for seed in range(41001,41005):
    reset,key=jax.random.split(jax.random.key(seed))
    o,st=jax.vmap(env.reset,(0,None))(jax.random.split(reset,64),p)
    prev=z=encode(o);alive=np.ones(64,bool)
    for t in range(65):
        if t in (0,32,64):
            v=np.asarray(agent.critic(j.concatenate([prev,z],axis=-1)))[:,0]
            ret=np.asarray(simulate(jax.random.fold_in(jax.random.key(seed+100000),t),st,prev,z))
            for i in np.flatnonzero(alive):
                rows.append([seed,int(i),t,float(v[i]),float(ret[i].mean()),float(ret[i].std(ddof=1)/8)])
        key,st,prev,z,_,done=prefix(key,st,prev,z)
        alive &= ~np.asarray(done)
with open('states.csv','w') as f:
    w=csv.writer(f);w.writerow(['seed','lane','step','critic','mc64','mc_se']);w.writerows(rows)
x=np.asarray(rows);v=x[:,3];g=x[:,4]
# Compare a cross-fitted constant fitted on the other three reset batches.
baseline=np.array([g[x[:,0]!=seed].mean() for seed in x[:,0]])
def metrics(v,g,b):
    return dict(n=len(v),pred=float(v.mean()),actual=float(g.mean()),rmse=float(np.mean((v-g)**2)**.5),constant_rmse=float(np.mean((b-g)**2)**.5),pearson=float(np.corrcoef(v,g)[0,1]) if v.std()>1e-7 and g.std()>0 else None)
summary=metrics(v,g,baseline)
summary['by_step']={str(t):metrics(v[x[:,2]==t],g[x[:,2]==t],baseline[x[:,2]==t]) for t in (0,32,64)}
with np.load(base/'critic-eval.npz') as archive:
    a=archive['actual']; pr=archive['value']
    summary['single_rollout']=metrics(pr,a,np.full_like(a,a.mean()))
Path('summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
