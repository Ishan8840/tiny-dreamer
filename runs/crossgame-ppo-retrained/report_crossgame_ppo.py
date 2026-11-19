from pathlib import Path
import csv,json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
games=['Asterix','Freeway','Seaquest','SpaceInvaders']
def read(p):
    with p.open() as f: return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
def score(p):
    rows=read(p)
    assert [(int(r['seed']),int(r['lane'])) for r in rows]==[(s,i) for s in range(41001,41005) for i in range(64)]
    nums=[float(r['return']) for r in rows]
    assert np.isfinite(nums).all()
    return np.mean(nums)
per=[]
summary=[]
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for game,ax in zip(games,axes.flat):
    tok=json.loads((root/'tokenizers'/game/'metrics.json').read_text())
    random=score(root/f'{game}-random.csv')
    held=[]
    curves=[]
    for seed in (0,1,2):
        run=root/f'{game}-s{seed}'
        val=read(run/'validation.csv')
        assert [int(r['actions']) for r in val]==list(range(0,16000001,320000))
        assert int(val[-1]['optimizer_updates'])==400000 and all(None not in r for r in val)
        curve=np.array([float(r['mean_return']) for r in val])
        best=val[int(curve.argmax())]
        test=score(run/'heldout.csv')
        held.append(test)
        curves.append(curve)
        per.append(dict(game=game,seed=seed,heldout=test,selected_actions=best['actions'],selected_update=best['optimizer_updates'],best_validation=curve.max(),final_validation=curve[-1]))
    ax.plot(np.arange(51)*.32,np.mean(curves,axis=0),color='#245b88',label='PPO mean')
    ax.fill_between(np.arange(51)*.32,np.min(curves,axis=0),np.max(curves,axis=0),color='#245b88',alpha=.16,label='Training seed range')
    ax.axhline(random,color='#92601d',linestyle='--',label='Held-out random')
    ax.set_title(game)
    ax.set_xlabel('Real training actions (millions)')
    ax.set_ylabel('Mean episode return')
    ax.grid(alpha=.2)
    summary.append(dict(game=game,random=random,ppo_mean=np.mean(held),seed_sd=np.std(held,ddof=1),seed0=held[0],seed1=held[1],seed2=held[2],tokenizer_exact=tok['exact_frame_accuracy'],tokenizer_f1=tok['positive_pixel_f1'],tokenizer_bce=tok['bce']))
axes.flat[0].legend()
fig.savefig(root/'curves.png',dpi=160)
write(root/'summary.csv',summary)
write(root/'per-seed.csv',per)
print((root/'summary.csv').read_text())
