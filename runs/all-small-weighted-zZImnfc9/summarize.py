from pathlib import Path
import json,csv,sys
import numpy as np
from flax.serialization import msgpack_restore
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def read(path):
    with path.open() as f:return list(csv.DictReader(f))
def count(x):
    return sum(map(count,x.values())) if isinstance(x,dict) else x.size if isinstance(x,np.ndarray) else 0
g=sys.argv[1];d=Path(g);policy=d/'seed-0';refs=Path('references')
ref,new=read(refs/f'{g}.csv'),read(policy/'heldout.csv')
assert len(new)==len(ref)==256 and [(r['seed'],r['lane']) for r in new]==[(r['seed'],r['lane']) for r in ref]
scores=np.array([[float(r['return']) for r in vals] for vals in [ref,new]])
diff=scores[1]-scores[0]
records=read(policy/'metrics.csv')
assert len(records)==100 and int(records[-1]['ppo_updates'])==400000
m=[r for r in records if r['validation_return']];best=max(m,key=lambda r:float(r['validation_return']))
params=sum(count(msgpack_restore(p.read_bytes())) for p in (policy/'world/best.msgpack',policy/'best.msgpack'))
rand=read(refs/f'{g}-random.csv')
result=dict(game=g,score=scores[1].mean(),reference=scores[0].mean(),random=float(np.mean([float(r['return']) for r in rand])),diff=diff.mean(),ci=np.quantile(np.random.default_rng(387).choice(diff,(10000,256)).mean(1),[.025,.975]).tolist(),best=int(best['real_actions']),params=params)
fig,ax=plt.subplots(figsize=(6.5,3.7),layout='constrained')
ax.plot([int(r['real_actions'])/1000 for r in m],[float(r['validation_return']) for r in m])
ax.set(title=g,xlabel='Real actions (thousands)',ylabel='Mean validation score (64 games)')
fig.savefig(d/'validation.png',dpi=160);(d/'summary.json').write_text(json.dumps(result,indent=2))
