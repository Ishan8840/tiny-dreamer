from pathlib import Path
import os
GAME=os.environ["MINATAR_GAME"]
import csv,json,sys
import numpy as np
import jax
import jax.numpy as j
from flax import nnx
import subprocess
import reward_eval
from train import Actor,Critic,WorldModel,make_env,restore_weights

out=Path(sys.argv[1]);env,p=make_env(GAME)
wm=WorldModel(nnx.Rngs(0),env.observation_space(p).shape[-1],env.action_space(p).n)
restore_weights(wm,out/"world/best.msgpack",require_outcomes=True)
rng=nnx.Rngs(0)
agent=nnx.Dict(actor=Actor(rng,env.action_space(p).n,128),critic=Critic(rng,128))
restore_weights(agent,out/"best.msgpack")
enc=wm.tokenizer.layers[0]
@jax.jit
def step(key,st,prev,z,active):
    key,ak,sk=jax.random.split(key,3)
    x=j.concatenate((prev,z),axis=-1)
    a=jax.random.categorical(ak,agent.actor(x))
    o,n,r,d,_=jax.vmap(env.step_env,(0,0,0,None))(jax.random.split(sk,64),st,a,p)
    live=active & ~(d|jax.vmap(env.is_truncated,(0,None))(n,p))
    def keep(new,old):
        return j.where(live.reshape((64,)+(1,)*(new.ndim-1)),new,old)
    return key,jax.tree.map(keep,n,st),enc(o.astype(j.float32)),live,r,agent.critic(x)[:,0],o[0],a
def rgb(frame):
    im=np.zeros((10,10,3),np.uint8)+22
    import colorsys
    for k in reversed(range(frame.shape[-1])):
        color=np.array(colorsys.hsv_to_rgb(k/frame.shape[-1],0.72,1.0))*255
        im[frame[:,:,k].astype(bool)]=color
    return np.repeat(np.repeat(im,32,axis=0),32,axis=1)
reward_totals=np.zeros((5,7))
pred=[];actual=[];scores=[];frames=[]
for seed in range(41001,41005):
    reset,key=jax.random.split(jax.random.key(seed))
    o,st=jax.vmap(env.reset,(0,None))(jax.random.split(reset,64),p)
    prev=z=enc(o.astype(j.float32));active=j.ones(64,bool)
    rr=[];vv=[];mm=[];zz=[];aa=[]
    if seed==41001:frames.append(rgb(np.asarray(o[0])))
    while np.any(active):
        mask=np.asarray(active)
        zz.append(np.asarray(z))
        key,st,nz,active,r,v,frame,action=step(key,st,prev,z,active)
        aa.append(np.asarray(action))
        prev,z=z,nz
        mm.append(mask);rr.append(np.asarray(r));vv.append(np.asarray(v))
        if seed==41001 and mask[0]:frames.append(rgb(np.asarray(frame)))
    mask=np.stack(mm);rew=np.where(mask,rr,0)
    reward_totals+=reward_eval.measure(wm,np.stack(zz),np.stack(aa),rew,mask,seed)
    scores.extend(rew.sum(0))
    returns=np.zeros_like(rew);future=np.zeros(64)
    for t in reversed(range(len(rew))):
        future=rew[t]+.99*future;returns[t]=future
    pred.extend(np.stack(vv)[mask]);actual.extend(returns[mask])
reward_eval.save(reward_totals,out/"reward-metrics.csv")
with (out/"heldout.csv").open() as f:
    reference=[float(row['return']) for row in csv.DictReader(f)]
# Both evaluators use the same starting states. Compiled policy samplers may differ at near-ties.
pred=np.array(pred);actual=np.array(actual)
assert np.all(np.isfinite(pred)) and np.all((pred>=0)&(pred<=100.001))
summary=dict(mean_value=float(pred.mean()),mean_return=float(actual.mean()),mse=float(np.mean((pred-actual)**2)),max_value=float(pred.max()),states=len(actual),diagnostic_episode_mean=float(np.mean(scores)),paired_score_matches=int(np.sum(np.array(reference)==scores)),video_score=float(scores[0]))
(out/"critic-eval.json").write_text(json.dumps(summary,indent=2))
np.savez_compressed(out/"critic-eval.npz",value=pred,actual=actual)
subprocess.run(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','320x320','-r','8','-i','pipe:0','-c:v','libx264','-pix_fmt','yuv420p',str(out/f'real-{GAME}.mp4')],input=b''.join(f.tobytes() for f in frames),check=True)
print(summary,flush=True)
