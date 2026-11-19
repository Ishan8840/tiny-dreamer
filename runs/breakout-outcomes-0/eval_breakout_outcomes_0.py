"""Evaluate the stored best outcome heads on fresh, complete Breakout windows."""
import csv
from pathlib import Path

from flax.traverse_util import flatten_dict
from train import (WorldModel, nnx, restore_weights, np, make_env, collect_episodes,
                   jax, make_replay, cache_latents, outcome_validation, OUTCOME_COMPONENTS)

run = Path("runs/breakout-outcomes-0")
source = Path.home() / "tiny-dreamer-dynamics-RvLkR0Hs/runs/breakout-frozen-qknorm-100k-0/best.msgpack"
model = WorldModel(nnx.Rngs(0), 4, 3)
restore_weights(model, source)
def arrays():
    return flatten_dict(nnx.to_pure_dict(nnx.state(model, nnx.Param)))
before = arrays()
restore_weights(model, run / "best.msgpack")
after = arrays()
changed = {k for k in before if not np.array_equal(before[k], after[k])}
heads = {k for k in before if k[0] == "dynamics" and k[1] in OUTCOME_COMPONENTS}
assert changed == heads
print(f"{len(changed)} outcome arrays changed; {len(before)-len(changed)} pretrained arrays identical")
env, params = make_env()
episodes = []
for seed in (30101, 30102, 30103):
    batch, _ = collect_episodes(env, params, jax.random.key(seed), 1000)
    episodes.extend(batch)
replay = make_replay(episodes, 16, 0)
cache_latents(model.tokenizer.layers[0], replay)
losses, counts, windows = outcome_validation(model.dynamics, replay.episodes, 16, 64)
with (run / "fresh-metrics.csv").open("w") as f:
    writer = csv.writer(f)
    writer.writerow(["event","episodes_collected","eligible_episodes","windows","bce","tp","fp","fn","precision","recall","f1"])
    for event, loss, (tp, predicted, positives) in zip(("reward","terminal"), losses, counts.T):
        fp, fn = predicted-tp, positives-tp
        row = [event,len(episodes),len(replay.episodes),windows,loss,tp,fp,fn,
               tp/predicted if predicted else float("nan"),tp/positives if positives else float("nan"),
               2*tp/(predicted+positives) if predicted+positives else float("nan")]
        writer.writerow(row)
        print(row)
