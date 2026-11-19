import csv
from pathlib import Path
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from train import policy_returns, make_env, WorldModel

class UniformPolicy(nnx.Module):
    def __init__(self, actions):
        self.actions = actions

    def __call__(self, state):
        return jnp.zeros((state.shape[0], self.actions))

env, params = make_env("Breakout")
actor = UniformPolicy(env.action_space(params).n)
encoder = WorldModel(nnx.Rngs(0), 4, actor.actions).tokenizer.layers[0]
scores = []
with (Path(__file__).parent/"random-policy.csv").open("w") as f:
    writer = csv.writer(f)
    writer.writerow(["seed","episode","return"])
    for seed in range(41001,41005):
        values = np.asarray(policy_returns(actor, encoder, env, params, jax.random.key(seed), 64))
        writer.writerows((seed, i, v) for i,v in enumerate(values))
        scores.extend(values)
print("uniform random:", len(scores), "episodes, mean", np.mean(scores), "SE", np.std(scores, ddof=1)/np.sqrt(len(scores)))
