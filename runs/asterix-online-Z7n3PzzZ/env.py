import gymnax


def make_env(game="Breakout"):
    env, env_params = gymnax.make(f"{game}-MinAtar")
    return env, env_params