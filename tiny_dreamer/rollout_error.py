"""Fit a frozen world model's latent-error probe on real replay trajectories."""

import csv
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .checkpoints import restore_weights, save_weights
from .model import RolloutErrorHead, denoise_next


def probe_settings(args):
    return {
        "game": args.game,
        "context": args.context,
        "horizon": args.imagination_horizon,
        "denoising_steps": args.denoising_steps,
        "world_sha256": hashlib.sha256(args.model_checkpoint.read_bytes()).hexdigest(),
    }


def load_probe(args, dynamics):
    path = args.error_checkpoint
    if json.loads((path.parent / "probe.json").read_text()) != probe_settings(args):
        raise ValueError(
            "Error probe requires its original game, world checkpoint and rollout settings"
        )

    head = RolloutErrorHead(nnx.Rngs(0), dynamics)
    restore_weights(head, path)
    return head


@nnx.jit(static_argnames=("context", "denoising_steps"))
def probe_examples(dynamics, latents, actions, key, context, denoising_steps):
    history = latents[:, :context]
    horizon = actions.shape[1] - context + 1

    def advance(history, depth):
        window = jax.lax.dynamic_slice_in_dim(actions, depth, context, axis=1)
        features = dynamics.outcome_features(history, window)
        noise = jax.random.normal(jax.random.fold_in(key, depth), history[:, -1].shape)
        prediction = denoise_next(dynamics, history, window, noise, denoising_steps)

        target = latents[:, context + depth]
        error = jnp.mean(jnp.square(prediction - target), axis=-1)

        reward_output = dynamics.reward_head(features)
        event = (
            reward_output[:, 0] > 0.5
            if dynamics.reward_target == "points"
            else dynamics.scoring_logit(reward_output) >= 0
        )

        following = jnp.concatenate((history[:, 1:], prediction[:, None]), axis=1)
        return following, (features, prediction, error, event)

    return jax.lax.scan(advance, history, jnp.arange(horizon))[1]


def predict_errors(head, features, predictions):
    return jax.vmap(head)(features, predictions, jnp.arange(1, len(features) + 1))


@nnx.jit
def fit_step(head, optimizer, features, predictions, errors):
    def loss(probe):
        return jnp.mean(
            jnp.square(predict_errors(probe, features, predictions) - errors)
        )

    optimizer.update(head, nnx.grad(loss)(head))


@nnx.jit
def score_probe(head, features, predictions):
    return predict_errors(head, features, predictions)


def fit_probe(args, dynamics, transform, replay, validation, key):
    head = RolloutErrorHead(nnx.Rngs(args.seed), dynamics)
    optimizer = nnx.Optimizer(head, transform, wrt=nnx.Param)
    window = args.context + args.imagination_horizon - 1

    def examples(buffer, key):
        batch = buffer.sample_batch(args.batch_size, window)
        arrays = probe_examples(
            dynamics,
            batch["observations"],
            batch["actions"],
            key,
            args.context,
            args.denoising_steps,
        )
        return (*arrays, batch["rewards"][:, args.context - 1 :].T > 0)

    (args.output / "probe.json").write_text(json.dumps(probe_settings(args), indent=2))
    best, stale = np.inf, 0
    with (args.output / "validation.csv").open("w", buffering=1) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "step",
                "depth",
                "risk_quartile",
                "count",
                "min_predicted_mse",
                "max_predicted_mse",
                "predicted_mse",
                "actual_mse",
                "reward_false_positives",
                "reward_false_negatives",
                "real_reward_events",
            )
        )

        for epoch in range(args.epochs):
            for _ in range(args.steps_per_epoch):
                key, step_key = jax.random.split(key)
                features, predictions, errors, _, _ = examples(replay, step_key)
                fit_step(head, optimizer, features, predictions, errors)

            validation.rng = np.random.default_rng(args.seed + 1)
            rows = []
            for i in range(args.val_batches):
                val_key = jax.random.fold_in(jax.random.key(args.seed + 1), i)
                features, predictions, actual, events, real = examples(
                    validation, val_key
                )
                rows.append(
                    (score_probe(head, features, predictions), actual, events, real)
                )

            scores, actual, events, real = (
                np.concatenate(col, axis=1) for col in zip(*rows)
            )
            mse = np.mean(np.square(scores - actual))
            if not np.isfinite(mse):
                raise FloatingPointError("Nonfinite rollout-error validation loss")

            step = (epoch + 1) * args.steps_per_epoch
            for d, risk in enumerate(scores):
                for q, idx in enumerate(np.array_split(np.argsort(risk), 4), 1):
                    if len(idx):
                        writer.writerow(
                            (
                                step,
                                d + 1,
                                q,
                                len(idx),
                                risk[idx].min(),
                                risk[idx].max(),
                                risk[idx].mean(),
                                actual[d, idx].mean(),
                                (events[d, idx] & ~real[d, idx]).sum(),
                                (~events[d, idx] & real[d, idx]).sum(),
                                real[d, idx].sum(),
                            )
                        )

            print(f"Step {step}  validation error-head MSE {mse:.6g}", flush=True)
            if mse < best:
                save_weights(head, args.output)
                (args.output / "best_step.txt").write_text(f"{step}\n")
                best, stale = mse, 0
            else:
                stale += 1
                if args.patience and stale >= args.patience:
                    break
