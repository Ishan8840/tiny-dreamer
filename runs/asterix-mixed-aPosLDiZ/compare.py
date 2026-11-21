import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
base=Path('/workspace/asterix-natural5-esIDhLRY/seed-0')
a,b=[rows(p/'heldout.csv') for p in [base,Path('seed-0')]]
assert len(a)==len(b)==256 and [(x['seed'],x['lane']) for x in a]==[(x['seed'],x['lane']) for x in b]
d=np.array([float(y['return'])-float(x['return']) for x,y in zip(a,b)])
fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
for directory,label in [(base,'Imagined critic'),(Path('seed-0'),'50/50 critic')]:
    m=[r for r in rows(directory/'metrics.csv') if r['validation_return']]
    ax.plot([int(r['real_actions'])/1000 for r in m],[float(r['validation_return']) for r in m],label=label)
best=max(m,key=lambda r:float(r['validation_return']))
ax.set(xlabel='Real actions (thousands)',ylabel='Mean return on 64 validation games');ax.legend()
fig.savefig('validation.png',dpi=170)
out=dict(n=len(d),baseline=float(np.mean([float(x['return']) for x in a])),mixed=float(np.mean([float(x['return']) for x in b])),difference=d.mean(),paired95=np.quantile(np.random.default_rng(780).choice(d,(10000,len(d))).mean(1),[.025,.975]).tolist(),selected=int(best['real_actions']))
Path('comparison.json').write_text(json.dumps(out,indent=2))
