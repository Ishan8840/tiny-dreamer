from pathlib import Path
import argparse
import csv

import jax
import numpy as np
import optax
from flax import nnx
from train import Actor,Critic,WorldModel,restore_weights,make_env,policy_returns,save_weights,imagined_ppo_update
from crossgame_replay import Replay

def main(game,seed):
    root=Path('policies')/f'{game}-s{seed}'
    root.mkdir(parents=True)
    env,params=make_env(game)
    wm=WorldModel(nnx.Rngs(17), env.observation_space(params).shape[-1],env.action_space(params).n)
    restore_weights(wm,Path('world')/game/'best.msgpack',require_outcomes=True)
    assert wm.dynamics.reward_target == 'points'
    replay=Replay(game,'train',seed)
    _,key,eval_key,init_key=jax.random.split(jax.random.key(seed),4)
    rngs=nnx.Rngs(init_key)
    agent=nnx.Dict(actor=Actor(rngs,env.action_space(params).n,256),critic=Critic(rngs,256))
    actor_opt=nnx.Optimizer(agent.actor,optax.adamw(1e-4,weight_decay=1e-4),wrt=nnx.Param)
    critic_opt=nnx.Optimizer(agent.critic,optax.adamw(3e-4,weight_decay=1e-4),wrt=nnx.Param)
    best=-np.inf
    with (root/'validation.csv').open('w') as f:
        w=csv.writer(f)
        w.writerow(('update','actor_loss','critic_mse','score'))
        for epoch in range(50):
            losses=[]
            for i in range(2000):
                z,a,_,_=replay.batch(32)
                key,step_key=jax.random.split(key)
                losses.append(imagined_ppo_update(agent,actor_opt,critic_opt,wm.dynamics,z[:,:-1],
                    a[:,:-1],step_key,3,4,.99,.95,.001))
            actor_loss,critic_mse=np.mean(losses,axis=0)
            score=float(policy_returns(agent.actor,wm.tokenizer.layers[0],env,params,eval_key,64).mean())
            assert np.isfinite((actor_loss,critic_mse,score)).all()
            step=(epoch+1)*8000
            w.writerow((step,actor_loss,critic_mse,score));f.flush()
            if score>best:
                save_weights(agent,root)
                best=score
            print(game,seed,step,score,'best',best,flush=True)
    restore_weights(agent,root/'best.msgpack')
    with (root/'heldout.csv').open('w') as f:
        w=csv.writer(f)
        w.writerow(('seed','lane','return'))
        for test_seed in range(41001,41005):
            scores=np.asarray(policy_returns(agent.actor,wm.tokenizer.layers[0],env,params,jax.random.key(test_seed),64))
            w.writerows((test_seed,i,r) for i,r in enumerate(scores))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('game');p.add_argument('seed',type=int)
    args=p.parse_args();main(args.game,args.seed)
