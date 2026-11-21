"""Save parameters atomically and reconstruct the matching model from disk."""

import jax
import jax.numpy as jnp
from flax import nnx, serialization

from .model import OUTCOME_COMPONENTS, REWARD_CLASSES, VALUE_BINS, Critic, WorldModel


def save_weights(model, directory):
    weights = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    if isinstance(model, WorldModel):
        weights["_reward_target"] = model.dynamics.reward_target
    temporary = directory / "best.msgpack.tmp"
    temporary.write_bytes(serialization.to_bytes(weights))
    temporary.replace(directory / "best.msgpack")


def restore_weights(model, path, *, require_outcomes=False):
    saved = serialization.msgpack_restore(path.read_bytes())
    state = nnx.state(model, nnx.Param)
    template = nnx.to_pure_dict(state)
    replacement_head = None
    reward_head = None
    if isinstance(model, WorldModel):
        reward_params = saved["dynamics"].get("reward_head")
        legacy = reward_params is not None and reward_params["bias"].shape == (1,)
        reward_target = saved.pop(
            "_reward_target", "event" if legacy else "categorical_points"
        )
        if reward_target not in (
            ("event", "points") if legacy else ("categorical_points",)
        ):
            raise ValueError(f"Incompatible reward target {reward_target} in {path}")
        width = 1 if legacy else REWARD_CLASSES
        if model.dynamics.reward_head.bias.shape != (width,):
            reward_head = nnx.Linear(
                model.dynamics.reward_head.kernel.shape[0], width, rngs=nnx.Rngs(0)
            )
            template["dynamics"]["reward_head"] = nnx.to_pure_dict(
                nnx.state(reward_head, nnx.Param)
            )
    # Actor-critic checkpoints predating the two-hot head store one output.
    if isinstance(model, nnx.Dict) and isinstance(model.get("critic"), Critic):
        kernel = saved.get("critic", {}).get("f3", {}).get("kernel")
        if (
            kernel is not None
            and kernel.shape in ((128, 1), (128, VALUE_BINS))
            and model.critic.f3.kernel.shape != kernel.shape
        ):
            replacement_head = nnx.Linear(128, kernel.shape[1], rngs=nnx.Rngs(0))
            template["critic"]["f3"] = nnx.to_pure_dict(
                nnx.state(replacement_head, nnx.Param)
            )
    expected = serialization.to_state_dict(template)
    if isinstance(model, WorldModel) and "dynamics" in saved and not require_outcomes:
        # Older world models predate the agent token and outcome heads.
        for name in OUTCOME_COMPONENTS:
            saved["dynamics"].setdefault(name, expected["dynamics"][name])
    if jax.tree.structure(saved) != jax.tree.structure(expected):
        raise ValueError(f"Incompatible checkpoint parameter keys: {path}")
    restored = serialization.from_state_dict(template, saved)

    for expected, actual in zip(
        jax.tree.leaves(template), jax.tree.leaves(restored), strict=True
    ):
        if actual.shape != expected.shape:
            raise ValueError(
                f"Incompatible checkpoint: {actual.shape} vs {expected.shape}"
            )

    if replacement_head is not None:
        model.critic.f3 = replacement_head
        state = nnx.state(model, nnx.Param)
    if isinstance(model, WorldModel):
        if reward_head is not None:
            model.dynamics.reward_head = reward_head
            state = nnx.state(model, nnx.Param)
        model.dynamics.reward_target = reward_target
    nnx.replace_by_pure_dict(state, jax.tree.map(jnp.asarray, restored))
    nnx.update(model, state)
