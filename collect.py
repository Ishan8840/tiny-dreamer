import jax
import numpy as np


def collect_episode(env, env_params, key):
    key, reset_key = jax.random.split(key)

    obs, state = env.reset(reset_key, env_params)

    observations = [np.asarray(obs)]
    actions = []
    rewards = []
    dones = []