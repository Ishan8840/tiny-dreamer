import jax
from env import make_env

env, env_params = make_env()

key = jax.random.PRNGKey(0)
key, reset_key = jax.random.split(key)

obs, state = env.reset(reset_key, env_params)

print("Observation shape:", obs.shape)
print("Action space:", env.action_space(env_params))

for i in range(100):
    key, action_key, step_key = jax.random.split(key, 3)

    action = env.action_space(env_params).sample(action_key)

    obs, state, reward, done, info = env.step(
        step_key, state, action, env_params
    )

    if done:
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key, env_params)

print("Successfully ran 100 environment steps!")