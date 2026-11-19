"""Compare frozen Breakout dynamics on one complete rollout per held-out episode."""
import argparse
import csv
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from evaluate_rollouts import BATCH, CONTEXT, collect_episodes, make_env, predict
from train_dynamics import WorldModel, restore_tokenizer


HORIZONS = tuple(range(1, 21))
RUNS = {"Adam 2k": "breakout-frozen-0", "AdamW 60k": "breakout-frozen-adamw-100k-0"}
OUT = Path("runs/matched-breakout-1000")


def collect_windows():
    env, params = make_env()
    rng = np.random.default_rng(4020)
    observations, actions, starts = [], [], []
    seed = 20000
    steps = CONTEXT - 1 + len(HORIZONS)
    while len(starts) < 1000:
        episodes, _ = collect_episodes(env, params, jax.random.key(seed), 1000)
        for index, ep in enumerate(episodes):
            length = len(ep["actions"])
            if length < steps:
                continue
            start = rng.integers(length - steps + 1)
            observations.append(ep["observations"][start:start + steps + 1])
            actions.append(ep["actions"][start:start + steps])
            starts.append((seed, index, start, length))
            if len(starts) == 1000:
                break
        if seed % 20 == 0:
            print(f"Collected {len(starts)} eligible episodes", flush=True)
        seed += 1
    with (OUT / "starts.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["collection_seed", "episode_in_batch", "context_start", "episode_actions"])
        writer.writerows(starts)
    return np.stack(observations).astype(bool), np.stack(actions)


def main(qknorm_ablation=False):
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "windows.npz"
    if not archive.exists():
        if qknorm_ablation:
            raise FileNotFoundError("The QKNorm comparison requires the archived matched rollout set")
        obs, actions = collect_windows()
        np.savez_compressed(archive, observations=obs, actions=actions)
    with np.load(archive) as data:
        obs, actions = data["observations"], data["actions"]
    assert obs.shape == (1000, 36, 10, 10, 4) and actions.shape == (1000, 35)
    assert np.all(obs[:, CONTEXT:, :, :, 1].sum(axis=(2, 3)) == 1)
    rows = []
    result_dir = OUT / "qknorm" if qknorm_ablation else OUT
    result_dir.mkdir(exist_ok=True)
    runs = ({"Baseline": RUNS["AdamW 60k"], "QKNorm": "breakout-frozen-qknorm-100k-0"}
            if qknorm_ablation else RUNS)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, run in runs.items():
        model = WorldModel(nnx.Rngs(0), 4, 3, qknorm=(name == "QKNorm"))
        restore_tokenizer(model, Path("runs") / run / "best.msgpack")
        exact, distances = [], []
        for start in range(0, len(obs), BATCH):
            ids = np.minimum(np.arange(start, start + BATCH), len(obs) - 1)
            n = min(BATCH, len(obs) - start)
            logits = np.asarray(predict(model, obs[ids, :CONTEXT].astype(np.float32), actions[ids],
                                        ids, horizons=HORIZONS))[:, :n]
            target = obs[ids[:n], CONTEXT:].swapaxes(0, 1)
            exact.append(((logits >= 0) == target).all(axis=(2, 3, 4)))
            pos = logits[..., 1].reshape(20, n, 100).argmax(axis=-1)
            true = target[..., 1].reshape(20, n, 100).argmax(axis=-1)
            distances.append(np.hypot(pos // 10 - true // 10, pos % 10 - true % 10))
        exact = np.concatenate(exact, axis=1)
        distance = np.concatenate(distances, axis=1)
        perfect = np.logical_and.accumulate(exact, axis=0)
        assert np.all(perfect <= exact)
        if qknorm_ablation and name == "Baseline":
            with np.load(OUT / f"{run}-scores.npz") as previous:
                print("Baseline replay differences across 20,000 endpoints:",
                      np.count_nonzero(exact != previous["exact"]), "exact-frame flags,",
                      np.count_nonzero(distance != previous["ball_distance"]), "ball distances")
        np.savez_compressed(result_dir / f"{run}-scores.npz", exact=exact, ball_distance=distance)
        for j, h in enumerate(HORIZONS):
            rows.append((name, h, exact[j].sum(), perfect[j].sum(), len(obs),
                         exact[j].mean(), distance[j].mean(), perfect[j].mean()))
        ax.plot(HORIZONS, distance.mean(axis=1), marker="o", markersize=3, label=name)
    ax.set(xlabel="Frames ahead", ylabel="Mean ball position error (grid cells)",
           ylim=(0, None), xticks=[1, 5, 10, 15, 20])
    ax.legend()
    fig.tight_layout()
    fig.savefig(result_dir / "ball-error.png", dpi=180)
    with (result_dir / "metrics.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "horizon", "exact_count", "perfect_count", "episodes",
                         "exact_rate", "mean_ball_error", "perfect_rate"])
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qknorm-ablation", action="store_true")
    main(parser.parse_args().qknorm_ablation)
