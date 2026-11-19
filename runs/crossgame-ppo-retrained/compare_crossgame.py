import csv
import json
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from train import policy_returns, make_env, make_tokenizer, restore_weights

class Uniform(nnx.Module):
    def __init__(self, n): self.n = n
    def __call__(self, x): return jnp.zeros((x.shape[0],self.n))
game=sys.argv[1]
env,params=make_env(game)
tokenizer=make_tokenizer(nnx.Rngs(0),env.observation_space(params).shape[-1])
restore_weights(tokenizer,Path('tokenizers')/game/'best.msgpack')
actor=Uniform(env.action_space(params).n)
with open(f'{game}-random.csv','w') as f:
    w=csv.writer(f)
    w.writerow(('seed','lane','return'))
    for seed in range(41001,41005):
        returns=np.asarray(policy_returns(actor,tokenizer.layers[0],env,params,jax.random.key(seed),64))
        w.writerows((seed,i,r) for i,r in enumerate(returns))
