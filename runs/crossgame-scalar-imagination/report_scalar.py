from pathlib import Path
import csv,json,hashlib
from importlib.metadata import version
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flax import serialization

root=Path.cwd()
real=Path.home()/'tiny-dreamer-crossgame-ppo-yDkAlBWH'
baseline={r['game']:r for r in csv.DictReader((real/'summary.csv').open())}
def read(p):
    with p.open() as f: return list(csv.DictReader(f))
def save(name,rows):
    with open(name,'w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
def arrays_match(a,b):
    if isinstance(a,dict):return set(a)==set(b) and all(arrays_match(a[k],b[k]) for k in a)
    return np.array_equal(a,b)
games=('Asterix','Freeway','Seaquest','SpaceInvaders')
rows=[]; metrics=[]
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for ax,g in zip(axes.flat,games):
    checkpoint=serialization.msgpack_restore((root/'world'/g/'best.msgpack').read_bytes())
    original=serialization.msgpack_restore((real/'tokenizers'/g/'best.msgpack').read_bytes())
    assert checkpoint['_reward_target']=='points' and arrays_match(original,checkpoint['tokenizer'])
    scores=json.loads((root/'world'/g/'reward-eval.json').read_text())
    curves=[];held=[]
    for s in range(3):
        folder=root/'policies'/f'{g}-s{s}'
        val=read(folder/'validation.csv')
        assert [int(v['update']) for v in val]==list(range(8000,400001,8000))
        outcomes=read(folder/'heldout.csv')
        reference=read(real/f'{g}-s{s}'/'heldout.csv')
        assert len(outcomes)==256 and [(r['seed'],r['lane']) for r in outcomes]==[(r['seed'],r['lane']) for r in reference]
        curve=np.array([float(v['score']) for v in val])
        best=val[curve.argmax()]
        score=np.mean([float(r['return']) for r in outcomes])
        curves.append(curve);held.append(score)
        metrics.append(dict(game=g,seed=s,heldout=score,best_validation=curve.max(),best_update=best['update'],final_validation=curve[-1],final_critic_mse=val[-1]['critic_mse']))
    r=baseline[g]
    rows.append(dict(game=g,imagined=np.mean(held),sd=np.std(held,ddof=1),seed0=held[0],seed1=held[1],seed2=held[2],real=r['ppo_mean'],random=r['random'],single_mse=scores['1']['reward_mse'],single_constant=scores['1']['baseline_mse'],three_mse=scores['3']['reward_mse'],three_constant=scores['3']['baseline_mse'],three_scoring_mse=scores['3']['scoring_mse'],three_zero_mse=scores['3']['zero_mse']))
    x=np.arange(1,51)*8000
    ax.plot(x,np.mean(curves,axis=0),color='#2675a5',label='Imagined PPO, validation')
    ax.fill_between(x,np.min(curves,axis=0),np.max(curves,axis=0),color='#2675a5',alpha=.2)
    ax.axhline(float(r['random']),linestyle=':',color='#a85a24',label='Held-out random')
    ax.set(title=g,xlabel='PPO optimizer updates',ylabel='Real game return')
    ax.grid(alpha=.2)
axes.flat[0].legend()
fig.savefig('scalar-curves.png',dpi=160)
save('scalar-summary.csv',rows);save('scalar-seeds.csv',metrics)
print(Path('scalar-summary.csv').read_text())
