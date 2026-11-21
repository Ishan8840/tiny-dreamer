"""Training configuration and command-line validation."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    latent_dim: int = 64


@dataclass
class TrainConfig:
    train: str = "dynamics"
    output: Path = Path("outputs")
    game: str = "Breakout"
    seed: int = 0
    episodes: int = 1000
    batch_size: int = 64

    learning_rate: float = 3e-4
    optimizer: str | None = None
    weight_decay: float = 1e-4
    epochs: int = 50

    training_horizon: int = 3
    context: int = 5
    steps_per_epoch: int = 2000
    val_batches: int = 16
    patience: int = 5
    dynamics_weight: float = 1.0
    denoising_steps: int = 4

    imagination_horizon: int = 5
    discount: float = 0.99
    lambda_return: float = 0.95
    entropy_scale: float = 0.001
    eval_episodes: int = 64
    real_horizon: int = 5
    policy_method: str | None = None
    actor_learning_rate: float | None = None

    tokenizer_checkpoint: Path | None = None
    model_checkpoint: Path | None = None
    policy_checkpoint: Path | None = None
    error_checkpoint: Path | None = None
    error_threshold: float | None = None
    reward_fraction: float | None = None
    error_low: float | None = None
    detach_rollout: bool = False

    world_updates: int = 1
    collect_steps: int = 64

    tokenizer_epochs: ClassVar[int] = 20
    ppo_learning_rate: ClassVar[float] = 1e-4


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the tokenizer and QKNorm shortcut dynamics"
    )
    parser.add_argument(
        "--train",
        choices=(
            "tokenizer",
            "dynamics",
            "both",
            "outcomes",
            "heads",
            "dynamics-outcomes",
            "rollout-error",
            "online",
            "actor-critic",
            "real-actor-critic",
        ),
        default="dynamics",
        help="Components to update (default: dynamics with a frozen tokenizer)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=ModelConfig.latent_dim,
        help="Latent width for a new tokenizer. Checkpoints supply their own width.",
    )
    parser.add_argument(
        "--collect-steps",
        type=int,
        default=TrainConfig.collect_steps,
        help="Real actions per online cycle",
    )
    parser.add_argument(
        "--world-updates",
        type=int,
        default=TrainConfig.world_updates,
        help="World-model updates per online cycle",
    )

    init = parser.add_mutually_exclusive_group()
    init.add_argument("--tokenizer-checkpoint", type=Path)
    init.add_argument(
        "--model-checkpoint",
        type=Path,
        help="Initialize from full QKNorm model weights",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New checkpoint directory"
    )
    parser.add_argument("--game", default=TrainConfig.game)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument(
        "--episodes",
        type=int,
        help="Defaults to 10,000 for tokenizer or 1,000 for dynamics",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Defaults to 32 for real-game training and 64 otherwise",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=TrainConfig.learning_rate
    )
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "adam"),
        help="Defaults to Adam for tokenizer or AdamW for dynamics and joint training",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=TrainConfig.weight_decay,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Defaults to 20 for tokenizer or 50 validation intervals for dynamics",
    )

    dyn = parser.add_argument_group("dynamics and joint training")
    dyn.add_argument(
        "--training-horizon",
        type=int,
        default=TrainConfig.training_horizon,
        help="Autoregressive prediction steps for dynamics training",
    )
    dyn.add_argument("--context", type=int, default=TrainConfig.context)
    dyn.add_argument(
        "--steps-per-epoch",
        "--val-every",
        type=int,
        default=TrainConfig.steps_per_epoch,
    )
    dyn.add_argument("--val-batches", type=int, default=TrainConfig.val_batches)
    dyn.add_argument(
        "--patience",
        type=int,
        default=TrainConfig.patience,
        help="Validation checks without improvement (0 disables)",
    )
    dyn.add_argument(
        "--dynamics-weight", type=float, default=TrainConfig.dynamics_weight
    )
    dyn.add_argument(
        "--denoising-steps",
        type=int,
        choices=(1, 2, 4, 8, 16),
        default=TrainConfig.denoising_steps,
    )

    policy = parser.add_argument_group("actor-critic")
    policy.add_argument(
        "--imagination-horizon", type=int, default=TrainConfig.imagination_horizon
    )
    policy.add_argument("--discount", type=float, default=TrainConfig.discount)
    policy.add_argument(
        "--lambda-return", type=float, default=TrainConfig.lambda_return
    )
    policy.add_argument(
        "--entropy-scale", type=float, default=TrainConfig.entropy_scale
    )
    policy.add_argument("--eval-episodes", type=int, default=TrainConfig.eval_episodes)
    policy.add_argument(
        "--real-horizon",
        type=int,
        default=TrainConfig.real_horizon,
        help="Real transitions per environment per update",
    )
    policy.add_argument(
        "--policy-method",
        choices=("a2c", "ppo", "reinforce", "pmpo"),
        help="Real-game training method (default: ppo)",
    )
    policy.add_argument(
        "--actor-learning-rate",
        type=float,
        help="Defaults to 1e-4 for PPO; other methods use --learning-rate",
    )
    policy.add_argument(
        "--policy-checkpoint",
        type=Path,
        help="Collect rollout-error replay with this policy",
    )
    policy.add_argument(
        "--error-checkpoint",
        type=Path,
        help="Fitted rollout-error head for adaptive imagination",
    )
    policy.add_argument(
        "--error-threshold", type=float, help="Maximum trusted predicted latent MSE"
    )
    dyn.add_argument(
        "--detach-rollout",
        action="store_true",
        help="Stop gradients through latent predictions used as subsequent inputs",
    )
    dyn.add_argument(
        "--reward-fraction",
        type=float,
        help="Fraction of positive endpoints in the one-step reward batch, with importance correction",
    )
    policy.add_argument(
        "--error-low",
        type=float,
        help="Predicted error with 90%% trust for soft weighting",
    )

    args = parser.parse_args(argv)
    if args.error_low is not None and not (
        args.error_checkpoint
        and args.error_threshold is not None
        and 0 <= args.error_low < args.error_threshold
    ):
        parser.error("--error-low requires a fitted probe and 0 <= low < threshold")
    if args.train == "rollout-error" and (
        not args.model_checkpoint or args.imagination_horizon < 1
    ):
        parser.error(
            "rollout-error needs a trained world checkpoint and a positive imagination horizon"
        )
    if args.policy_checkpoint and args.train != "rollout-error":
        parser.error("--policy-checkpoint is for rollout-error collection")
    if (args.error_checkpoint or args.error_threshold is not None) and (
        args.train != "actor-critic"
        or not args.error_checkpoint
        or args.error_threshold is None
        or not np.isfinite(args.error_threshold)
        or args.error_threshold < 0
    ):
        parser.error(
            "adaptive imagination requires actor-critic, an error checkpoint and a finite nonnegative threshold"
        )

    real_game = args.train == "real-actor-critic"
    if args.batch_size is None:
        args.batch_size = 32 if real_game else TrainConfig.batch_size

    if args.reward_fraction is not None and not (
        args.train in ("online", "dynamics-outcomes")
        and args.training_horizon == 1
        and 0 < args.reward_fraction < 1
        and args.batch_size >= (4 if args.train == "online" else 2)
    ):
        parser.error(
            "reward stratification requires one-step online or dynamics-outcomes training and at least two examples per replay source"
        )

    if args.actor_learning_rate is not None and (
        not np.isfinite(args.actor_learning_rate) or args.actor_learning_rate <= 0
    ):
        parser.error("--actor-learning-rate must be finite and positive")
    if args.train == "online":
        if not args.model_checkpoint or args.policy_method not in (None, "ppo"):
            parser.error("online requires --model-checkpoint and uses PPO")
        args.policy_method = "ppo"
    if args.train in ("actor-critic", "online") and args.policy_method == "ppo":
        if args.actor_learning_rate is None:
            args.actor_learning_rate = TrainConfig.ppo_learning_rate
    elif real_game:
        args.policy_method = args.policy_method or "ppo"
        if args.actor_learning_rate is None:
            args.actor_learning_rate = (
                1e-4 if args.policy_method == "ppo" else args.learning_rate
            )
    elif args.policy_method or args.actor_learning_rate is not None:
        parser.error(
            "use --train real-actor-critic, or --train actor-critic --policy-method ppo for imagination"
        )

    if args.train in ("online", "actor-critic", "real-actor-critic"):
        if args.train == "actor-critic" and not args.model_checkpoint:
            parser.error(
                "actor-critic requires --model-checkpoint with trained reward and terminal heads"
            )
        if args.train == "real-actor-critic" and (
            not (args.model_checkpoint or args.tokenizer_checkpoint)
            or args.real_horizon < 1
        ):
            parser.error(
                "real-actor-critic requires pretrained encoder weights and a positive real-horizon"
            )
        if args.context < 2 or min(args.imagination_horizon, args.eval_episodes) < 1:
            parser.error(
                "actor-critic requires context >= 2 and positive imagination horizon and eval episodes"
            )
        if not (
            0 <= args.discount <= 1
            and 0 <= args.lambda_return <= 1
            and np.isfinite(args.entropy_scale)
            and args.entropy_scale >= 0
        ):
            parser.error(
                "discount and lambda-return must be in [0,1] and entropy-scale finite and nonnegative"
            )

    tokenizer_only = args.train == "tokenizer"
    if args.episodes is None:
        args.episodes = 10000 if tokenizer_only else 1000
    if args.epochs is None:
        args.epochs = (
            TrainConfig.tokenizer_epochs if tokenizer_only else TrainConfig.epochs
        )

    if args.train == "dynamics" and not (
        args.tokenizer_checkpoint or args.model_checkpoint
    ):
        parser.error(
            "dynamics training needs pretrained weights: supply --tokenizer-checkpoint or --model-checkpoint"
        )
    if args.train == "dynamics-outcomes" and not (
        args.tokenizer_checkpoint or args.model_checkpoint
    ):
        parser.error("dynamics-outcomes requires tokenizer or world-model weights")
    if min(args.latent_dim, args.collect_steps, args.world_updates) < 1:
        parser.error("latent width and online training dimensions must be positive")
    if args.train in ("outcomes", "heads") and not args.model_checkpoint:
        parser.error(
            "outcome training requires --model-checkpoint with pretrained dynamics"
        )

    if (
        args.episodes < 2
        or min(
            args.epochs,
            args.batch_size,
            args.context,
            args.training_horizon,
            args.steps_per_epoch,
            args.val_batches,
        )
        < 1
    ):
        parser.error("require at least two episodes and positive training dimensions")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if args.patience < 0 or not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error(
            "--patience and --weight-decay must be nonnegative with finite weight decay"
        )
    if not np.isfinite(args.dynamics_weight) or args.dynamics_weight <= 0:
        parser.error("--dynamics-weight must be finite and positive")

    return ModelConfig(args.latent_dim), TrainConfig(
        **{k: v for k, v in vars(args).items() if k != "latent_dim"}
    )
