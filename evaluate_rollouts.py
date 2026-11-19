"""Compare saved Breakout dynamics checkpoints on fresh random-policy episodes."""
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from flax import nnx

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from collect import collect_episodes
from env import make_env
from train_dynamics import WorldModel, restore_tokenizer


HORIZONS = (1, 5, 10, 20)
CONTEXT = 16
BATCH = 64


def accuracy_counts(prediction, target):
    correct = prediction == target
    return np.array([len(target), correct.sum(), (prediction & target).sum(),
                     prediction.sum() + target.sum(), correct[..., 1].all(axis=(1, 2)).sum(),
                     correct[..., 0].all(axis=(1, 2)).sum(), correct[..., 3].sum()], dtype=np.int64)


def accuracy_row(model, horizon, counts):
    frames, bits, tp, positives, balls, paddles, bricks = counts
    return [model, horizon, frames, bits / (400 * frames), 2 * tp / positives,
            balls / frames, paddles / frames, bricks / (100 * frames)]


@nnx.jit
def reconstruct(model, frames):
    return model.tokenizer(frames) >= 0


@nnx.jit(static_argnames=("horizons",))
def predict(model, context, actions, ids, horizons=HORIZONS):
    encoder, decoder = model.tokenizer.layers
    history = encoder(context)
    d = jnp.full((len(ids),), 0.25)

    def step(history, index):
        action_window = jax.lax.dynamic_slice_in_dim(actions, index, CONTEXT, axis=1)
        key = jax.random.fold_in(jax.random.key(901), index)
        noise = jax.vmap(lambda i: jax.random.normal(jax.random.fold_in(key, i),
                                                     (history.shape[-1],)))(ids)

        def denoise(k, latent):
            clean = model.dynamics(history, action_window, latent, k * d, d)
            return latent + (clean - latent) / (4 - k)

        latent = jax.lax.fori_loop(0, 4, denoise, noise)
        return jnp.concatenate((history[:, 1:], latent[:, None]), axis=1), latent

    _, latents = jax.lax.scan(step, history, jnp.arange(max(horizons)))
    return decoder(latents[jnp.asarray(horizons) - 1])


def plot_example(examples, path):
    colors = ["#43b9f2", "#ffb648", "#ab86cb", "#72b998"]
    fig, axes = plt.subplots(4, len(examples), figsize=(2.4 * len(examples), 8))
    for col, (label, frames) in enumerate(examples.items()):
        axes[0, col].set_title(label)
        for row, frame in enumerate(frames):
            ax = axes[row, col]
            ax.set_facecolor("#141414")
            for channel in (3, 2, 0, 1):
                y, x = np.nonzero(frame[:, :, channel])
                ax.scatter(x, y, c=colors[channel], marker=("s", "o", "x", "s")[channel],
                           s=(75, 55, 38, 70)[channel])
            ax.set(xlim=(-0.5, 9.5), ylim=(9.5, -0.5), xticks=[], yticks=[], aspect="equal")
            if col == 0:
                ax.set_ylabel(f"+{HORIZONS[row]} frame" + ("s" if row else ""))
    fig.legend([Patch(color=c) for c in colors], ["Paddle", "Ball", "Trail", "Brick"],
               loc="lower center", ncol=4)
    fig.suptitle("Breakout • one held-out trajectory")
    fig.tight_layout(rect=(0, .045, 1, .97))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    out = Path("runs/rollout-eval-frozen")
    out.mkdir(exist_ok=True)
    env, params = make_env()
    val = []
    for seed in (1001, 1002, 1003):
        data_key = jax.random.split(jax.random.key(seed), 3)[0]
        episodes, _ = collect_episodes(env, params, data_key, 1000)
        val.extend(episodes)
    windows = [(i, s) for i, ep in enumerate(val) for s in range(len(ep["actions"]) - CONTEXT + 1)]
    max_actions = CONTEXT - 1 + max(HORIZONS)
    example = next(j for j, (i, s) in enumerate(windows) if len(val[i]["actions"]) - s >= max_actions)
    examples = {}
    rows = []
    accuracy_rows = []
    observations = np.concatenate([ep["observations"] for ep in val])
    for name in ("frozen",):
        model = WorldModel(nnx.Rngs(0), 4, 3, qknorm=False)
        restore_tokenizer(model, Path(f"runs/breakout-{name}-0/best.msgpack"))
        reconstruction_counts = np.zeros(7, dtype=np.int64)
        for start in range(0, len(observations), 512):
            batch = observations[start:start + 512]
            reconstruction_counts += accuracy_counts(np.asarray(reconstruct(model, batch)), batch.astype(bool))
        accuracy_rows.append(accuracy_row(name, 0, reconstruction_counts))
        rollout_counts = np.zeros((len(HORIZONS), 7), dtype=np.int64)
        counts = np.zeros((4, 4, 3), dtype=np.int64)
        losses = np.zeros(4)
        samples = np.zeros(4, dtype=np.int64)
        exact = np.zeros_like(samples)
        for start in range(0, len(windows), BATCH):
            batch_ids = np.arange(start, start + BATCH)
            ids = np.minimum(batch_ids, len(windows) - 1)
            frames, actions, truths, lengths = [], [], [], []
            for ident in ids:
                i, s = windows[ident]
                ep = val[i]
                frames.append(ep["observations"][s:s + CONTEXT])
                a = ep["actions"][s:s + max_actions]
                lengths.append(len(a) - CONTEXT + 1)
                actions.append(np.pad(a, (0, max_actions - len(a))))
                offsets = np.minimum(CONTEXT - 1 + np.array(HORIZONS), len(a))
                truths.append(ep["observations"][s + offsets])
            logits = np.asarray(predict(model, np.stack(frames), np.stack(actions), ids))
            truths = np.stack(truths).astype(bool)
            for j, horizon in enumerate(HORIZONS):
                valid = (batch_ids < len(windows)) & (np.array(lengths) >= horizon)
                y, l = truths[valid, j], logits[j, valid]
                p = l >= 0
                rollout_counts[j] += accuracy_counts(p, y)
                samples[j] += len(y)
                exact[j] += (p == y).all(axis=(1, 2, 3)).sum()
                losses[j] += np.logaddexp(0, np.where(y, -l, l)).sum(dtype=np.float64)
                counts[j] += np.stack([q.sum(axis=(0, 1, 2)) for q in (p & y, p & ~y, ~p & y)], axis=-1)
            if start <= example < start + BATCH:
                examples["Ground truth"] = truths[example - start]
                examples[name.title()] = logits[:, example - start] >= 0
        for j, horizon in enumerate(HORIZONS):
            accuracy_rows.append(accuracy_row(name, horizon, rollout_counts[j]))
            tp, fp, fn = counts[j].sum(axis=0)
            b, f, g = counts[j, 1]
            rows.append([name, horizon, samples[j], losses[j] / (400 * samples[j]),
                         tp / (tp + fp), tp / (tp + fn), 2 * tp / (2 * tp + fp + fn),
                         2 * b / (2 * b + f + g), exact[j] / samples[j]])
    with (out / "metrics.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "horizon", "windows", "bce", "precision", "recall",
                         "f1", "ball_f1", "exact_frame"])
        writer.writerows(rows)
    with (out / "object-accuracy.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "horizon", "frames", "pixel_accuracy", "positive_pixel_f1",
                         "ball_position", "paddle_position", "brick_occupancy"])
        writer.writerows(accuracy_rows)
    plot_example(examples, out / "comparison.png")


if __name__ == "__main__":
    main()
