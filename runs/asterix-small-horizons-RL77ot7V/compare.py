from pathlib import Path
import csv,json
from statistics import mean
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flax.serialization import msgpack_restore

def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def count(x):
    return sum(map(count,x.values())) if isinstance(x,dict) else x.size if isinstance(x,np.ndarray) else 0
scores=[];result={}
fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
for h in [1,3]:
    d=Path(f'h{h}')
    held=rows(d/'seed-0/heldout.csv');assert len(held)==256
    scores.append(held)
    val=[r for r in rows(d/'seed-0/metrics.csv') if r['validation_return']]
    best=max(val,key=lambda r:float(r['validation_return']))
    world=msgpack_restore((d/'seed-0/world/best.msgpack').read_bytes())
    agent=msgpack_restore((d/'seed-0/best.msgpack').read_bytes())
    params={k:count(v) for obj in [world,agent] for k,v in obj.items() if isinstance(v,dict)}
    assert sum(params.values())==563488,params
    rollout=rows(d/'heldout-rollouts.csv')
    chosen=[r for r in rollout if r['checkpoint']=='online']
    result[h]=dict(policy_mean=mean(float(r['return']) for r in held),selected=int(best['real_actions']),params=params,online_rollouts=chosen)
    axes[0].plot([int(r['real_actions'])/1000 for r in val],[float(r['validation_return']) for r in val],label=f'Train {h}-step')
    axes[1].plot([int(r['horizon']) for r in chosen],[float(r['exact_frame'])*100 for r in chosen],marker='o',label=f'Train {h}-step')
axes[0].set(xlabel='Real actions (thousands)',ylabel='Mean validation return')
axes[1].set(xlabel='Frames ahead',ylabel='Exact frames (%)',xticks=[1,2,3,4,5])
for ax in axes:ax.legend()
fig.savefig('comparison.png',dpi=180)
assert [(r['seed'],r['lane']) for r in scores[0]]==[(r['seed'],r['lane']) for r in scores[1]]
diff=np.array([float(y['return'])-float(x['return']) for x,y in zip(*scores)])
result['paired']=dict(mean=diff.mean(),interval=np.quantile(np.random.default_rng(885).choice(diff,(10000,256)).mean(1),[.025,.975]).tolist())
Path('comparison.json').write_text(json.dumps(result,indent=2))
