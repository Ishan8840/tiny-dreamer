"""Train the joint dynamics and scalar point head on fixed random-policy replay."""
import argparse
import csv
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from crossgame_dynamics import pretrained
from crossgame_replay import Replay
from train import shortcut_loss, restore_weights, save_weights, denoise_next

@nnx.jit
def update(dynamics, opt, z, a, reward, terminal, key):
    def objective(net):
        points, end = net.predict_outcomes(z[:, :-1], a)
        return (shortcut_loss(net, z, a, key) + jnp.square(points-reward).mean()
                + optax.sigmoid_binary_cross_entropy(end,terminal).mean())
    loss, grads = nnx.value_and_grad(objective)(dynamics)
    opt.update(dynamics, grads)
    return loss

def sample_windows(replay, horizon, seed):
    # Stratified by any reward / termination. Inverse sampling fractions recover
    # the transition-weighted mean while retaining rare scoring windows.
    idx = np.concatenate([np.arange(o+15,o+n-horizon+1) for n,o in zip(replay.lengths,replay.offsets)
                          if n >= 15+horizon])
    points = sum(replay.r[idx+i] for i in range(horizon))
    strata = (points!=0).astype(int)*2 + replay.t[idx+horizon-1].astype(int)
    rng=np.random.default_rng(seed)
    endpoints=[]; weights=[]
    for k in range(4):
        subset=idx[strata==k]
        if not len(subset): continue
        n=min(len(subset), 8192 if k==0 else 2048)
        endpoints.extend(rng.choice(subset,n,replace=False))
        weights.extend([len(subset)/n]*n)
    idx=np.asarray(endpoints)
    return idx,np.searchsorted(replay.offsets,idx,side='right')-1,np.asarray(weights)

@nnx.jit(static_argnames=('horizon',))
def predict(net,z,actions,key,horizon):
    history=z
    points=jnp.zeros((len(z),))
    for h in range(horizon):
        ac=actions[:,h:h+16]
        reward,terminal=net.predict_outcomes(history,ac)
        points=points+reward
        if h+1<horizon:
            noise=jax.random.normal(jax.random.fold_in(key,h),history[:,-1].shape)
            znew=denoise_next(net,history,ac,noise,4)
            history=jnp.concatenate((history[:,1:],znew[:,None]),axis=1)
    return points,terminal

def evaluate(net,replay,windows,mean_reward,horizon):
    idx,eps,weights=windows
    preds=[];ends=[]
    for i in range(0,len(idx),128):
        ids=idx[i:i+128]
        ep=eps[i:i+128]
        starts=ids-15
        z=replay.z[(starts+ep)[:,None]+np.arange(16)].astype(np.float32)
        actions=replay.a[starts[:,None]+np.arange(15+horizon)].astype(np.int32)
        pts,end=predict(net,z,actions,jax.random.fold_in(jax.random.key(82),i),horizon)
        preds.extend(np.asarray(pts)); ends.extend(np.asarray(end))
    actual=sum(replay.r[idx+i] for i in range(horizon))
    delta=np.asarray(preds)-actual
    nonzero=actual!=0
    terminal=replay.t[idx+horizon-1]
    end_loss=np.logaddexp(0,np.where(terminal,-np.asarray(ends),np.asarray(ends)))
    def avg(values,mask):
        return float(np.average(values[mask],weights=weights[mask])) if mask.any() else None
    return dict(reward_mse=float(np.average(delta**2,weights=weights)),
                baseline_mse=float(np.average((mean_reward*horizon-actual)**2,weights=weights)),
                scoring_mse=avg(delta**2,nonzero),zero_mse=avg(delta**2,~nonzero),
                terminal_bce=float(np.average(end_loss,weights=weights)),
                samples=len(idx),scoring_samples=int(nonzero.sum()))
def main(game):
    wm=pretrained(game)
    restore_weights(wm,Path('dynamics')/game/'best.msgpack')
    wm.dynamics.reward_target='points'
    train=Replay(game,'train',19);val=Replay(game,'val',81)
    mu=float(np.mean(train.r,dtype=np.float64))
    windows=sample_windows(val,1,81)
    folder=Path('world')/game
    folder.mkdir(parents=True)
    optim=nnx.Optimizer(wm.dynamics,optax.adamw(3e-4,weight_decay=1e-4),wrt=nnx.Param)
    best=np.inf; stale=0
    with (folder/'validation.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(('step','train_loss','reward_mse','terminal_bce'))
        for epoch in range(50):
            losses=[]
            for i in range(2000):
                z,a,r,t=train.batch(64)
                losses.append(update(wm.dynamics,optim,z,a,r,t.astype(np.float32),jax.random.fold_in(jax.random.key(19),2000*epoch+i)))
            scores=evaluate(wm.dynamics,val,windows,mu,1)
            loss=float(np.mean(losses))
            metric=scores['reward_mse']+scores['terminal_bce']
            assert np.isfinite([loss,metric]).all()
            step=(epoch+1)*2000
            writer.writerow((step,loss,scores['reward_mse'],scores['terminal_bce']));f.flush()
            if metric<best:
                save_weights(wm,folder)
                best,stale=metric,0
            else:
                stale+=1
            print(step,metric,best,flush=True)
            if stale>=5: break
    restore_weights(wm,folder/'best.msgpack',require_outcomes=True)
    assert wm.dynamics.reward_target=='points'
    scores={str(h):evaluate(wm.dynamics,val,sample_windows(val,h,8217),mu,h) for h in [1,3]}
    (folder/'reward-eval.json').write_text(json.dumps(scores,indent=2))
    (folder/'complete').touch()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('game');main(p.parse_args().game)
