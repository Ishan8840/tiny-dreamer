import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flax.serialization import msgpack_restore
def rows(p):
    with p.open() as f:return list(csv.DictReader(f))
def count(x):
    return sum(map(count,x.values())) if isinstance(x,dict) else x.size if isinstance(x,np.ndarray) else 0
summary={}
fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
for game,ax in zip(['Asterix','Breakout'],axes):
    d=Path(game);out=d/'seed-0'
    a,b=rows(d/'reference.csv'),rows(out/'heldout.csv')
    assert len(b)==len(a)==256 and [(r['seed'],r['lane']) for r in a]==[(r['seed'],r['lane']) for r in b]
    returns=np.array([[float(r['return']) for r in group] for group in (a,b)])
    diff=returns[1]-returns[0]
    m=[r for r in rows(out/'metrics.csv') if r['validation_return']]
    best=max(m,key=lambda r:float(r['validation_return']))
    param_count=sum(count(msgpack_restore(p.read_bytes())) for p in (out/'best.msgpack',out/'world/best.msgpack'))
    summary[game]=dict(reference=returns[0].mean(),score=returns[1].mean(),delta=diff.mean(),paired95=np.quantile(np.random.default_rng(892).choice(diff,(10000,256)).mean(1),[.025,.975]).tolist(),best_step=int(best['real_actions']),params=param_count)
    ax.plot([int(r['real_actions'])/1000 for r in m],[float(r['validation_return']) for r in m])
    ax.set(title=game,xlabel='Real actions (thousands)',ylabel='Mean validation return (64 games)')
fig.savefig('validation.png',dpi=180)
def best_recall(y,p,floor):
    ix=np.argsort(-p,kind='stable')
    tp=np.cumsum(y[ix]);ends=np.r_[np.flatnonzero(np.diff(p[ix])),len(p)-1]
    pr=tp[ends]/(ends+1)
    return float(np.max(tp[ends][pr>=floor],initial=0)/sum(y))
d=Path('Asterix')
with np.load(d/'coin-eval/paired.npz') as new,np.load(d/'reference-eval/paired.npz') as old:
    for k in ['reward','action','episode','frame']:np.testing.assert_array_equal(new[k],old[k])
    matched={}
    for i in (0,4):
        mask=np.isfinite(old['reward'][:,i]);y=old['reward'][mask,i]>0
        matched[i+1]={}
        for label,archive in [('natural',old),('weighted',new)]:
            p=archive['p_event'][mask,i]
            matched[i+1][label]=dict(recall_at_50pct_precision=best_recall(y,p,.5),actual_points=int(y.sum()),predicted_points=float(archive['expected'][mask,i].sum()))
    summary['matched_reward']=matched
Path('comparison.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
