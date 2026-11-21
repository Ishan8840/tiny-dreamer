import csv,json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from flax import nnx
import imageio_ffmpeg
from train import WorldModel,restore_weights,make_env
base=Path('/workspace/asterix-natural5-esIDhLRY/seed-0')
env,p=make_env('Asterix')
A=env.action_space(p).n
wm=WorldModel(nnx.Rngs(0),env.observation_space(p).shape[-1],A)
restore_weights(wm,base/'world/best.msgpack',require_outcomes=True)
assert jax.default_backend()=='gpu'
@jax.jit
def encode(o):return wm.tokenizer.layers[0](o.astype(j.float32))
@jax.jit
def step(key,st,hist,past):
    key,ak,sk=jax.random.split(key,3)
    def reward(action):
        acts=j.concatenate([past,j.full((64,1),action,j.int32)],axis=1)
        logits,_=wm.dynamics.outcome_outputs(hist,acts)
        return wm.dynamics.expected_reward(logits)
    scores=jax.vmap(reward)(j.arange(A)).T
    a=jax.random.categorical(ak,j.where(scores==scores.max(1,keepdims=True),0.,-j.inf))
    o,st,r,d,_=jax.vmap(env.step_env,(0,0,0,None))(jax.random.split(sk,64),st,a,p)
    d |= jax.vmap(env.is_truncated,(0,None))(st,p)
    return key,st,encode(o),a,r,d,o[0],j.all(j.isfinite(scores))
def rgb(frame):
    im=np.zeros((10,10,3),np.uint8)+22
    for k,color in [(2,(112,108,138)),(1,(213,91,84)),(3,(255,208,68)),(0,(93,215,169))]:
        im[np.asarray(frame[:,:,k],bool)]=color
    return np.repeat(np.repeat(im,32,axis=0),32,axis=1)
with (base/'heldout.csv').open() as f:
    ref={(int(r['seed']),int(r['lane'])):float(r['return']) for r in csv.DictReader(f)}
rows=[];frames=[]
for seed in range(41001,41005):
    reset,key=jax.random.split(jax.random.key(seed))
    o,st=jax.vmap(env.reset,(0,None))(jax.random.split(reset,64),p)
    h=encode(o)[:,None];past=j.empty((64,0),j.int32)
    active=np.ones(64,bool);ret=np.zeros(64);length=np.zeros(64,int)
    if seed==41001:frames.append(rgb(o[0]))
    while active.any():
        key,st,z,a,r,d,frame,ok=step(key,st,h,past)
        assert bool(ok)
        if seed==41001 and active[0]:frames.append(rgb(frame))
        ret+=np.where(active,np.asarray(r),0);length+=active
        active &= ~np.asarray(d)
        h=j.concatenate([h,z[:,None]],axis=1)[:,-5:]
        past=j.concatenate([past,a[:,None]],axis=1)[:,-4:]
    rows.extend((seed,i,ref[seed,i],ret[i],length[i]) for i in range(64))
assert len(ref)==len(rows)==256
with open('episodes.csv','w') as f:
    w=csv.writer(f);w.writerow(['seed','lane','actor','greedy','greedy_actions']);w.writerows(rows)
actor,greedy=np.array(rows)[:,2:4].T
diff=greedy-actor
ci=np.quantile(np.random.default_rng(71).choice(diff,(10000,len(diff))).mean(1),[.025,.975])
summary=dict(actor=actor.mean(),greedy=greedy.mean(),difference=diff.mean(),interval=ci.tolist(),wins=int(sum(diff>0)),ties=int(sum(diff==0)),losses=int(sum(diff<0)),video_score=greedy[0])
Path('summary.json').write_text(json.dumps(summary,indent=2))
w=imageio_ffmpeg.write_frames('greedy.mp4',(320,320),fps=8,codec='libx264',pix_fmt_in='rgb24');w.send(None)
for frame in frames:w.send(frame)
w.close()
print(summary)
