from pathlib import Path
import csv,json
import jax
import jax.numpy as j
import numpy as np
from flax import nnx
def average_precision_score(y,s):
    order=np.argsort(-s)
    tp=np.cumsum(y[order])
    ends=np.r_[np.flatnonzero(np.diff(s[order])),len(s)-1]
    return np.sum(np.diff(np.r_[0,tp[ends]])*tp[ends]/(ends+1))/tp[-1]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import WorldModel,Actor,Critic,restore_weights,make_env,denoise_next

base=Path('/workspace/asterix-online-Z7n3PzzZ/seed-0')
env,p=make_env('Asterix')
A=env.action_space(p).n
wm=WorldModel(nnx.Rngs(0),4,A)
restore_weights(wm,base/'world/best.msgpack',require_outcomes=True)
rng=nnx.Rngs(0)
agent=nnx.Dict(actor=Actor(rng,A,256),critic=Critic(rng,256))
restore_weights(agent,base/'best.msgpack')
enc=wm.tokenizer.layers[0]

@jax.jit
def step(key,st,prev,z,active):
    key,ak,sk=jax.random.split(key,3)
    a=jax.random.categorical(ak,agent.actor(j.concatenate((prev,z),axis=-1)))
    keys=jax.random.split(sk,64)
    batched=jax.vmap(env.step_env,(0,0,0,None))
    o,n,r,d,_=batched(keys,st,a,p)
    altr=jax.vmap(lambda act:batched(keys,st,j.full_like(a,act),p)[2])(j.arange(A)).T
    live=active & ~(d|jax.vmap(env.is_truncated,(0,None))(n,p))
    def keep(new,old):
        return j.where(live.reshape((64,)+(1,)*(new.ndim-1)),new,old)
    return key,jax.tree.map(keep,n,st),enc(o.astype(j.float32)),live,a,r,altr

episodes=[]
for seed in range(41001,41005):
    reset,key=jax.random.split(jax.random.key(seed))
    o,st=jax.vmap(env.reset,(0,None))(jax.random.split(reset,64),p)
    prev=z=enc(o.astype(j.float32));active=j.ones(64,bool)
    rows=[]
    while np.any(active):
        before=np.asarray(z);valid=np.asarray(active)
        key,st,nz,active,a,r,alt=step(key,st,prev,z,active)
        rows.append((before,np.asarray(a),np.asarray(r),np.asarray(alt),valid))
        prev,z=z,nz
    zs,acts,rew,other,mask=map(np.stack,zip(*rows))
    for lane in range(64):
        n=int(mask[:,lane].sum())
        episodes.append(tuple(x[:n,lane].copy() for x in (zs,acts,rew,other)))
L=np.array([len(ep[1]) for ep in episodes])
scores=np.array([ep[2].sum() for ep in episodes])
with (base/'heldout.csv').open() as f:
    ref=np.array([float(r['return']) for r in csv.DictReader(f)])
np.testing.assert_array_equal(scores,ref)
z,a,r,other=map(np.concatenate,zip(*episodes))
np.testing.assert_array_equal(other[np.arange(len(a)),a],r)
assert set(np.unique(other))<={0,1}
starts=np.r_[0,np.cumsum(L)[:-1]]; ep=np.repeat(np.arange(len(L)),L); frame=np.arange(len(a))-starts[ep]
avail=np.minimum(5,L[ep]-frame)
actual=np.full((len(a),5),np.nan)
points=actual.copy();prob=actual.copy();allpts=np.empty_like(other,dtype=float)
@jax.jit
def read(hist,ac):
    out,_=wm.dynamics.outcome_outputs(hist,ac)
    return wm.dynamics.expected_reward(out),jax.nn.sigmoid(wm.dynamics.scoring_logit(out))
@jax.jit
def options(hist,ac):
    return jax.vmap(lambda action:read(hist,ac.at[:,-1].set(action))[0])(j.arange(A)).T
@jax.jit
def advance(hist,ac,keys,h):
    noise=jax.vmap(lambda key:jax.random.normal(jax.random.fold_in(key,h),(128,)))(keys)
    return denoise_next(wm.dynamics,hist,ac,noise,4)
for ctx in range(1,17):
    rows=np.flatnonzero(np.minimum(16,frame+1)==ctx)
    for off in range(0,len(rows),128):
        group=rows[off:off+128];ix=np.pad(group,(0,128-len(group)),mode='edge')
        hidx=ix[:,None]+np.arange(1-ctx,1);hist=j.asarray(z[hidx]);ac=j.asarray(a[hidx[:,:-1]])
        keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.key(11372),i))(ix)
        for h in range(5):
            at=np.minimum(ix+h,starts[ep[ix]]+L[ep[ix]]-1)
            ac=j.concatenate((ac,j.asarray(a[at,None])),axis=1)[:,-16:]
            if h==0:allpts[group]=np.asarray(options(hist,ac))[:len(group)]
            v,s=read(hist,ac);ok=avail[group]>h
            for dest,values in [(actual,r[at]),(points,v),(prob,s)]:
                dest[group[ok],h]=np.asarray(values)[:len(group)][ok]
            if h<4:
                nz=advance(hist,ac,keys,h)
                hist=j.concatenate((hist,nz[:,None]),axis=1)[:,-16:]
    print('context',ctx,flush=True)
np.testing.assert_allclose(allpts[np.arange(len(a)),a],points[:,0],atol=1e-4)
rows=[]
for h in range(5):
    g=np.flatnonzero(avail>h);y=actual[g,h];endpoint=g+h
    np.testing.assert_array_equal(r[endpoint],y)
    for kind,pts,s in [('real',points[endpoint,0],prob[endpoint,0]),('imagined',points[g,h],prob[g,h])]:
        pos=y>0;hit=s>=.5;tp=int(sum(pos&hit));fp=int(sum(~pos&hit));fn=int(sum(pos&~hit))
        rows.append(dict(horizon=h+1,history=kind,steps=len(y),coins=int(sum(pos)),tp=tp,fp=fp,fn=fn,recall=tp/sum(pos),precision=tp/(tp+fp) if tp+fp else None,ap=average_precision_score(y,s),prevalence=y.mean(),mean_pred_on_coin=pts[pos].mean(),mean_pred_on_no_coin=pts[~pos].mean(),mse=np.mean((pts-y)**2),mean_pred=pts.mean()))
with open('metrics.csv','w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
opp=(other.max(1)>other.min(1))
best=allpts[opp]==allpts[opp].max(1,keepdims=True)
fraction=(best*other[opp]).sum(1)/best.sum(1)
summary=dict(episodes=len(L),scoring_episodes=int((scores>0).sum()),coins=int(scores.sum()),steps=len(a),action_sensitive_states=int(opp.sum()),actor_pickup_fraction=float(r[opp].mean()),uniform_pickup_fraction=float(other[opp].mean()),reward_greedy_pickup_fraction=float(fraction.mean()))
Path('summary.json').write_text(json.dumps(summary,indent=2))
np.savez_compressed('paired.npz',episode=ep,frame=frame,reward=actual,expected=points,p_event=prob,action=a,all_action_reward=other,all_action_expected=allpts)
fig,ax=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
for hist,c in [('real','#287aa1'),('imagined','#bc612e')]:
    records=[r for r in rows if r['history']==hist]
    for pane,metric,title in zip(ax,['mean_pred_on_coin','ap'],['Mean predicted points on a pickup','Average precision for coin pickups']):
        pane.plot(range(1,6),[r[metric] for r in records],'o-',label=hist.capitalize()+' history',color=c)
        pane.set(xlabel='Reward horizon (actions)',title=title,ylim=(0,1.02),xticks=range(1,6));pane.legend()
fig.savefig('coin-predictions.png',dpi=160)
print(summary,flush=True)
print(Path('metrics.csv').read_text(),flush=True)
