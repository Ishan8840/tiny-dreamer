import csv
from pathlib import Path
import sys
import jax
import numpy as np
from flax import nnx
from train import Actor,Critic,WorldModel,restore_weights,make_env,policy_returns
out=Path(sys.argv[1])
env,params=make_env("Asterix")
wm=WorldModel(nnx.Rngs(0),env.observation_space(params).shape[-1],env.action_space(params).n)
restore_weights(wm,out/'world/best.msgpack',require_outcomes=True)
rng=nnx.Rngs(0)
agent=nnx.Dict(actor=Actor(rng,env.action_space(params).n,256),critic=Critic(rng,256))
restore_weights(agent,out/'best.msgpack')
with (out/'heldout.csv').open('w') as f:
    w=csv.writer(f)
    w.writerow(['seed','lane','return'])
    for seed in range(41001,41005):
        scores=np.asarray(policy_returns(agent.actor,wm.tokenizer.layers[0],env,params,jax.random.key(seed),64))
        w.writerows((seed,i,r) for i,r in enumerate(scores))
