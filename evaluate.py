"""Evaluate a saved actor in the real MinAtar game."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from flax import nnx, serialization

from tiny_dreamer.actor_critic import policy_returns
from tiny_dreamer.checkpoints import restore_weights
from tiny_dreamer.data import make_env
from tiny_dreamer.model import Actor, Critic, WorldModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument(
        "--model-checkpoint", type=Path, help="World weights for a legacy policy"
    )
    parser.add_argument(
        "--game", help="Game name (defaults to the checkpoint configuration)"
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=41001)

    args = parser.parse_args()
    if min(args.episodes, args.batch_size) < 1:
        parser.error("episodes and batch size must be positive")

    config = args.policy.parent / "config.json"
    game = args.game or (
        json.loads(config.read_text())["training"]["game"] if config.is_file() else None
    )
    if not game:
        parser.error("supply --game for a legacy policy")

    env, params = make_env(game)
    rngs = nnx.Rngs(0)
    policy = serialization.msgpack_restore(args.policy.read_bytes())
    width = policy["actor"]["f1"]["kernel"].shape[0]
    actions = env.action_space(params).n
    model = WorldModel(
        rngs, env.observation_space(params).shape[-1], actions, latent_dim=width // 2
    )
    if args.model_checkpoint:
        restore_weights(model, args.model_checkpoint)
    else:
        restore_weights(
            model.tokenizer, args.policy.parent / "tokenizer" / "best.msgpack"
        )

    agent = nnx.Dict(actor=Actor(rngs, actions, width), critic=Critic(rngs, width))
    restore_weights(agent, args.policy)

    scores = []
    for offset in range(0, args.episodes, args.batch_size):
        key = jax.random.key(args.seed + offset // args.batch_size)
        scores.extend(
            np.asarray(
                policy_returns(
                    agent.actor,
                    model.tokenizer.layers[0],
                    env,
                    params,
                    key,
                    min(args.batch_size, args.episodes - offset),
                )
            )
        )

    print(f"{game}  episodes {len(scores)}  mean return {np.mean(scores):.4f}")


if __name__ == "__main__":
    main()
