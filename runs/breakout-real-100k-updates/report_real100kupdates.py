from pathlib import Path
import csv
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train import *

root = Path(__file__).resolve().parent
rows = re.findall(r"Update (\d+) actions (\d+) mean return ([\d.]+)", (root/"training.log").read_text())
assert len(rows) == 51 and rows[-1][:2] == ("100000", "16000000")
with (root/"metrics.csv").open("w") as f:
    writer = csv.writer(f)
    writer.writerow(("update","training_actions","validation_mean_return"))
    writer.writerows(rows)
fig, ax = plt.subplots(figsize=(7,3.5),layout="constrained")
ax.plot([int(x[0]) for x in rows],[float(x[2]) for x in rows],marker=".",color="#226591")
ax.set(xlabel="Optimizer updates",ylabel="Mean episode return",title="Breakout · 64 fixed validation episodes")
fig.savefig(root/"returns.png",dpi=160)
env, params = make_env("Breakout")
wm = WorldModel(nnx.Rngs(0),4,3)
restore_weights(wm, Path.home()/"tiny-dreamer-heads10k-BTdi973d/runs/dynamics-outcomes/best.msgpack")
actor_key = jax.random.split(jax.random.key(0),4)[-1]
rngs = nnx.Rngs(actor_key)
agent = nnx.Dict(actor=Actor(rngs,3,256),critic=Critic(rngs,256))
results = {}
for mode, path in (("real625",Path.home()/"tiny-dreamer-real100k-D9OC2Xuo/run/best.msgpack"),("real",root/"run/best.msgpack"),("imagined",Path.home()/"tiny-dreamer-ac500k-5EHfAamz/run/best.msgpack")):
    if path:
        restore_weights(agent,path)
    scores = np.concatenate([np.asarray(policy_returns(agent.actor,wm.tokenizer.layers[0],env,params,
                                jax.random.key(seed),64)) for seed in range(41001,41005)])
    results[mode]=scores
    print(mode, len(scores), scores.mean(), scores.std(ddof=1)/np.sqrt(len(scores)),flush=True)
with (root/"heldout.csv").open("w") as f:
    writer=csv.writer(f)
    writer.writerow(("seed","episode",*results))
    for i, values in enumerate(zip(*results.values())):
        writer.writerow((41001+i//64,i%64,*values))
