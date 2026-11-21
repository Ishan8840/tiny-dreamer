import csv
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from train import denoise_next

@nnx.jit
def predict(dynamics, history, actions, ids):
    def step(history, t):
        ac=jax.lax.dynamic_slice_in_dim(actions,t,5,axis=1)
        logits,_=dynamics.outcome_outputs(history,ac)
        key=jax.random.fold_in(jax.random.key(6201),t)
        noise=jax.vmap(lambda i:jax.random.normal(jax.random.fold_in(key,i),(history.shape[-1],)))(ids)
        z=denoise_next(dynamics,history,ac,noise,4)
        return jnp.concatenate((history[:,1:],z[:,None]),axis=1),(dynamics.expected_reward(logits),jax.nn.sigmoid(dynamics.scoring_logit(logits)))
    return jax.lax.scan(step,history,jnp.arange(5))[1]

def measure(wm,latents,actions,rewards,mask,seed):
    # Equal selection probability for each full five-step trajectory.
    t,lane=np.nonzero(mask[8:])
    rng=np.random.default_rng(seed)
    chosen=rng.random(len(t)) < .125
    t,lane=t[chosen],lane[chosen]
    totals=np.zeros((5,7))
    for i in range(0,len(t),128):
        idx=np.minimum(np.arange(i,i+128),len(t)-1)
        n=min(128,len(t)-i)
        ix=t[idx,None]+np.arange(9)
        hist=latents[ix[:,:5],lane[idx,None]]
        ac=actions[ix,lane[idx,None]]
        pred,prob=map(np.asarray,predict(wm.dynamics,hist,ac,t[idx]*64+lane[idx]))
        true=rewards[ix[:,4:],lane[idx,None]].T[:,:n]
        point=pred[:,:n]; hit=prob[:,:n]>=.5; event=true>0
        totals+=np.stack([np.full(5,n),event.sum(1),hit.sum(1),(hit&event).sum(1),point.sum(1),true.sum(1),np.square(point-true).sum(1)],1)
    return totals

def save(totals,path):
    with path.open('w') as f:
        w=csv.writer(f);w.writerow(['horizon','windows','reward_events','predicted_events','true_positives','precision','recall','expected_points','actual_points','reward_mse'])
        for h,(n,pos,hit,tp,p,y,se) in enumerate(totals,1):
            w.writerow([h,int(n),int(pos),int(hit),int(tp),tp/hit if hit else '',tp/pos if pos else '',p,y,se/n if n else ''])
