import argparse
import csv
from pathlib import Path

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from flax.traverse_util import flatten_dict
from evaluate_rollouts import predict, BATCH, CONTEXT
from model import WorldModel, restore_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("heads","dynamics-outcomes"), required=True)
    args = parser.parse_args()
    folder = Path("runs") / args.mode
    matched = Path.home() / "tiny-dreamer-dynamics-RvLkR0Hs/runs/matched-breakout-1000"
    m = WorldModel(nnx.Rngs(17),4,3)
    restore_weights(m, matched.parent / "breakout-frozen-qknorm-100k-0/best.msgpack")
    def state():
        return flatten_dict(nnx.to_pure_dict(nnx.state(m,nnx.Param)))
    initial = state()
    restore_weights(m, folder / "best.msgpack")
    final = state()
    assert initial.keys() == final.keys()
    changed = [k for k in initial if not np.array_equal(initial[k], final[k])]
    assert all(k[0]=="dynamics" for k in changed)
    if args.mode=="heads":
        assert set(changed)=={k for k in initial if k[0]=="dynamics" and k[1] in ("reward_head","terminal_head")}
    else:
        assert {"blocks","agent_token","reward_head","terminal_head","out_proj"} <= {k[1] for k in changed}
    print("Changed parameter arrays:",len(changed),"of",len(initial), "; tokenizer exactly preserved",flush=True)
    with np.load(matched / "windows.npz") as f:
        obs, actions = f["observations"],f["actions"]
    flags, distances = [],[]
    for start in range(0,len(obs),BATCH):
        ids=np.minimum(np.arange(start,start+BATCH),len(obs)-1)
        n=min(BATCH,len(obs)-start)
        logits=np.asarray(predict(m, obs[ids,:CONTEXT].astype(np.float32), actions[ids], ids,
                                  horizons=tuple(range(1,21))))[:,:n]
        truth=obs[ids[:n],CONTEXT:].swapaxes(0,1)
        flags.append(((logits>=0)==truth).all(axis=(2,3,4)))
        p=logits[...,1].reshape(20,n,100).argmax(-1)
        t=truth[...,1].reshape(20,n,100).argmax(-1)
        distances.append(np.hypot(p//10-t//10,p%10-t%10))
    exact=np.concatenate(flags,axis=1)
    distance=np.concatenate(distances,axis=1)
    perfect=np.logical_and.accumulate(exact,axis=0)
    np.savez_compressed(folder/"eval"/"visual-scores.npz",exact=exact,ball_error=distance)
    with (folder/"eval"/"visual.csv").open("w") as f:
        w=csv.writer(f);w.writerow(["horizon","exact","perfect","ball_error"])
        for i in range(20):
            w.writerow((i+1,exact[i].mean(),perfect[i].mean(),distance[i].mean()))

if __name__=="__main__":
    main()
