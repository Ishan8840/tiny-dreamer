import csv,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
rows=[]
fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
for h,color in [(3,'#246c9b'),(5,'#ae5a2b')]:
    curves=[]
    for seed in range(3):
        name=f'h{h}-s{seed}'
        matches=re.findall(r'^Step (\d+)  actor loss (\S+)  critic MSE (\S+)  real mean return (\S+)',(root/f'{name}.log').read_text(),re.M)
        assert [int(r[0]) for r in matches]==list(range(8000,400001,8000))
        curve=np.array([float(r[3]) for r in matches])
        curves.append(curve)
        best=int(np.argmax(curve))
        with (root/name/'heldout.csv').open() as f:
            episodes=list(csv.DictReader(f))
        assert [(int(r['seed']),int(r['lane'])) for r in episodes]==[(s,i) for s in range(41001,41005) for i in range(64)]
        held=np.mean([float(r['return']) for r in episodes])
        rows.append((h,seed,int(matches[best][0]),curve[best],curve[-1],np.mean(curve[-10:]),held))
        with (root/name/'validation.csv').open('w') as f:
            w=csv.writer(f)
            w.writerow(('optimizer_updates','actor_loss','critic_mse','mean_return'))
            w.writerows(matches)
    ax.plot(np.arange(1,51)*8000,np.mean(curves,axis=0),label=f'{h}-step imagination',color=color)
    ax.fill_between(np.arange(1,51)*8000,np.min(curves,axis=0),np.max(curves,axis=0),color=color,alpha=.13)
ax.set(xlabel='PPO optimizer updates',ylabel='Real Breakout mean return',title='64 validation episodes per checkpoint; shaded range across training seeds')
ax.legend()
fig.savefig(root/'curves.png',dpi=150)
with (root/'per-seed.csv').open('w') as f:
    w=csv.writer(f)
    w.writerow(('horizon','seed','best_update','best_validation','final_validation','last10_validation','heldout'))
    w.writerows(rows)
print(rows)
