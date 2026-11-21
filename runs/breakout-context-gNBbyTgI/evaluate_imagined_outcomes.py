"""Score transition events at rollout endpoints on recorded Breakout trajectories."""
import argparse
import csv
from itertools import groupby
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from collect import collect_episodes
from env import make_env
from model import WorldModel, denoise_next, restore_weights


CONTEXT = 16
HORIZONS = (1, 5, 10, 20)
LENGTH = CONTEXT - 1 + max(HORIZONS)
BATCH = 64


@nnx.jit
def score(model, observations, actions, ids):
    encoded = model.tokenizer.layers[0](observations.astype(jnp.float32))

    def step(history, index):
        real = jax.lax.dynamic_slice_in_dim(encoded, index, CONTEXT, axis=1)
        acts = jax.lax.dynamic_slice_in_dim(actions, index, CONTEXT, axis=1)
        imagined_logits = jnp.stack(model.dynamics.predict_outcomes(history, acts), axis=-1)
        real_logits = jnp.stack(model.dynamics.predict_outcomes(real, acts), axis=-1)
        key = jax.random.fold_in(jax.random.key(901), index)
        noise = jax.vmap(lambda i: jax.random.normal(jax.random.fold_in(key, i),
                                                     (history.shape[-1],)))(ids)
        latent = denoise_next(model.dynamics, history, acts, noise, 4)
        return jnp.concatenate((history[:, 1:], latent[:, None]), axis=1), (imagined_logits, real_logits)

    _, logits = jax.lax.scan(step, encoded[:, :CONTEXT], jnp.arange(max(HORIZONS)))
    return jnp.stack(logits)[:, jnp.array(HORIZONS) - 1]


def archived_starts(env, params, matched):
    with (matched / "starts.csv").open() as f:
        starts = list(csv.DictReader(f))
    with np.load(matched / "windows.npz") as data:
        obs, acts = data["observations"], data["actions"]
    assert len(starts) == len(acts) == 1000
    for seed, group in groupby(enumerate(starts), lambda item: int(item[1]["collection_seed"])):
        episodes, _ = collect_episodes(env, params, jax.random.key(seed), 1000)
        for i, row in group:
            n, s = int(row["episode_in_batch"]), int(row["context_start"])
            ep = episodes[n]
            np.testing.assert_array_equal(ep["observations"][s:s+LENGTH+1], obs[i])
            np.testing.assert_array_equal(ep["actions"][s:s+LENGTH], acts[i])
            yield seed, n, s, ep


def natural_starts(env, params):
    for seed in range(31001, 31011):
        episodes, _ = collect_episodes(env, params, jax.random.key(seed), 1000)
        for n, ep in enumerate(episodes):
            for s in range(len(ep["actions"]) - CONTEXT + 1):
                yield seed, n, s, ep


def evaluate(model, stream, name, output):
    from itertools import islice
    chunks = []
    offset = 0
    while group := list(islice(stream, BATCH)):
        size = len(group)
        frames, actions, labels, lengths, starts = [], [], [], [], []
        for seed, n, start, ep in group:
            a = ep["actions"][start:start+LENGTH]
            length = len(a)
            starts.append((seed, n, start))
            lengths.append(length - CONTEXT + 1)
            frames.append(ep["observations"][start + np.minimum(np.arange(LENGTH), length)])
            actions.append(np.pad(a, (0, LENGTH-length)))
            endpoints = start + np.minimum(CONTEXT - 2 + np.array(HORIZONS), length-1)
            labels.append(np.stack((ep["rewards"][endpoints] > 0, ep["terminals"][endpoints]), axis=-1))
        idx = np.minimum(np.arange(BATCH), size-1)
        logits = np.asarray(score(model, np.stack(frames)[idx], np.stack(actions)[idx], offset+idx))
        logits = logits[:, :, :size].transpose(2, 0, 1, 3)
        assert np.isfinite(logits).all()
        np.testing.assert_array_equal(logits[:, 0, 0] >= 0, logits[:, 1, 0] >= 0)
        chunks.append((np.array(starts), np.array(lengths), np.stack(labels), logits))
        offset += size
    starts, lengths, labels, logits = [np.concatenate(parts) for parts in zip(*chunks)]
    np.savez_compressed(output / f"{name}.npz", starts=starts, available_horizon=lengths,
                        labels=labels, logits=logits, horizons=HORIZONS)
    rows = []
    for j,h in enumerate(HORIZONS):
        valid = lengths >= h
        for mode, predictions in zip(("imagined", "real"), logits[valid, :, j].transpose(1, 0, 2)):
            for k,event in enumerate(("reward","game_over")):
                y, p = labels[valid, j, k], predictions[:, k] >= 0
                tp, fp, fn = (int(q.sum()) for q in (p & y, p & ~y, ~p & y))
                # Without a positive label, event-detection recall/F1 is uninformative.
                rows.append([name,mode,h,event,int(valid.sum()),tp+fn,tp,fp,fn,
                             tp/(tp+fp) if tp+fp else None, tp/(tp+fn) if tp+fn else None,
                             2*tp/(2*tp+fp+fn) if tp+fn else None])
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--matched", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env, params = make_env()
    model = WorldModel(nnx.Rngs(0),4,env.action_space(params).n)
    restore_weights(model,args.checkpoint)
    with (args.output / "metrics.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["cohort","history","horizon","event","windows","positives","tp","fp","fn","precision","recall","f1"])
        for name,stream in (("matched",archived_starts(env,params,args.matched)),
                            ("natural",natural_starts(env,params))):
            writer.writerows(evaluate(model,stream,name,args.output))
            f.flush()
            print(f"Completed {name} cohort",flush=True)
