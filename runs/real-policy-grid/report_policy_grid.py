import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))
def save(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
groups = defaultdict(list)
all_rows = []
fig, axs = plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True,layout='constrained')
for ax,method in zip(axs.flat,('a2c','ppo','reinforce','pmpo')):
    for rate,color in (('3e-4','#2678a8'),('1e-4','#c0632c')):
        curves=[]
        for seed in (0,1,2):
            folder=root/f'{method}-{rate}-s{seed}'
            val=read(folder/'validation.csv')
            test=read(folder/'heldout.csv')
            assert [int(r['actions']) for r in val]==list(range(0,16000001,320000))
            assert [(int(r['seed']),int(r['lane'])) for r in test]==[(i,j) for i in range(41001,41005) for j in range(64)]
            curve=np.array([float(r['mean_return']) for r in val])
            best=val[int(curve.argmax())]
            hit=next((int(r['actions']) for r in val if float(r['mean_return'])>=8),None)
            row=dict(method=method,actor_lr=rate,seed=seed,heldout=np.mean([float(r['return']) for r in test]),
                     best_validation=curve.max(),selected_actions=int(best['actions']),final_validation=curve[-1],
                     last10_validation=np.mean(curve[-10:]),first_validation_ge8=hit,
                     optimizer_updates=int(val[-1]['optimizer_updates']))
            all_rows.append(row)
            groups[(method,rate)].append(row)
            curves.append(curve)
        ax.plot(np.arange(51)*.32,np.mean(curves,axis=0),color=color,label=rate)
        ax.fill_between(np.arange(51)*.32,np.min(curves,axis=0),np.max(curves,axis=0),color=color,alpha=.13)
    ax.set_title(method.upper())
    ax.grid(alpha=.2)
axs[0,0].legend(title='Actor LR')
fig.supxlabel('Real training actions (millions)')
fig.supylabel('Mean episode return on validation')
fig.savefig(root/'curves.png',dpi=160)
save(root/'per-seed.csv',all_rows)
summary=[]
for (method,rate),rs in groups.items():
    scores=[r['heldout'] for r in rs]
    summary.append(dict(method=method,actor_lr=rate,heldout_mean=np.mean(scores),heldout_seed_sd=np.std(scores,ddof=1),
                        seed0=scores[0],seed1=scores[1],seed2=scores[2],
                        mean_last10_validation=np.mean([r['last10_validation'] for r in rs]),
                        reached_8=sum(r['first_validation_ge8'] is not None for r in rs)))
save(root/'summary.csv',summary)
print((root/'summary.csv').read_text())
