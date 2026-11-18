import jax
import numpy as np
from env import make_env
from replay import ReplayBuffer


def collect_episode(env, env_params, key):
    key, reset_key = jax.random.split(key)

    obs, state = env.reset(reset_key, env_params)

    observations = [np.asarray(obs)]
    actions = []
    rewards = []
    dones = []

    done = False

    while not done:
        key, action_key, step_key = jax.random.split(key, 3)

        action = env.action_space(env_params).sample(action_key)

        obs, state, reward, done, info = env.step_env(
            step_key, state, action, env_params
        )

        observations.append(np.asarray(obs))
        actions.append(int(action))
        rewards.append(float(reward))
        dones.append(bool(done))

    episode = {
        "observations": np.stack(observations),
        "actions": np.asarray(actions),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
    }

    return episode, key


env, env_params = make_env()
buffer = ReplayBuffer(capacity=1000)

key = jax.random.PRNGKey(0)

for i in range(10):
    episode, key = collect_episode(env, env_params, key)
    buffer.add_episode(episode)

print("Collected episodes:", len(buffer.episodes))