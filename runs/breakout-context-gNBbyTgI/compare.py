import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))

fig, axes = plt.subplots(1, 2, figsize=(10,4), layout='constrained')
policies = []
for c in (5,16):
    arm = Path(f'c{c}')
    evaluations = rows(arm/'seed-0/heldout.csv')
    assert len(evaluations)==256
    policies.append(evaluations)
    validation = [r for r in rows(arm/'seed-0/metrics.csv') if r['validation_return']]
    axes[0].plot([int(r['real_actions'])/1000 for r in validation],
                 [float(r['validation_return']) for r in validation],label=f'{c} frames')
    rollout = [r for r in rows(arm/'rollout-metrics.csv') if r['checkpoint']=='online']
    axes[1].plot([int(r['horizon']) for r in rollout],[100*float(r['exact_frame_rate']) for r in rollout],label=f'{c} frames')
axes[0].set(xlabel='Real training actions (thousands)',ylabel='Mean validation return (64 games)')
axes[1].set(xlabel='Frames ahead',ylabel='Exact frames (%)',xticks=[1,5,10,15,20])
for ax in axes:
    ax.legend()
fig.savefig('comparison.png',dpi=180)
assert [(r['seed'],r['lane']) for r in policies[0]]==[(r['seed'],r['lane']) for r in policies[1]]
scores=np.array([[float(r['return']) for r in e] for e in policies])
diff=scores[0]-scores[1]
rng=np.random.default_rng(81)
interval=np.quantile(diff[rng.integers(0,256,(10000,256))].mean(1),[.025,.975])
Path('comparison.json').write_text(json.dumps(dict(heldout_means=scores.mean(1).tolist(),paired_difference=diff.mean(),episode_bootstrap_95=interval.tolist()),indent=2))
