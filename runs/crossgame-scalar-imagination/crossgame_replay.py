from pathlib import Path
import json
import sys

import jax
import numpy as np
from flax import nnx
from collect import collect_episodes
from train import encode_frames, make_env, make_tokenizer, restore_weights

def build(game):
    root=Path('data')/game
    root.mkdir(parents=True,exist_ok=True)
    env,params=make_env(game)
    tok=make_tokenizer(nnx.Rngs(17),env.observation_space(params).shape[-1])
    restore_weights(tok,Path.home()/f'tiny-dreamer-crossgame-ppo-yDkAlBWH/tokenizers/{game}/best.msgpack')
    width=128
    for split,total,key in [('train',10000,71017),('val',1111,81017)]:
        lengths=[]
        positive=terminals=0
        max_reward=0.
        with (root/f'{split}.z').open('wb') as zf,(root/f'{split}.a').open('wb') as af,(root/f'{split}.r').open('wb') as rf,(root/f'{split}.t').open('wb') as tf:
            for start in range(0,total,128):
                episodes,_=collect_episodes(env,params,jax.random.fold_in(jax.random.key(key),start),min(128,total-start))
                obs=np.concatenate([e['observations'] for e in episodes])
                for off in range(0,len(obs),512):
                    chunk=obs[off:off+512]
                    np.asarray(encode_frames(tok.layers[0],chunk),dtype=np.float16).tofile(zf)
                a=np.concatenate([e['actions'] for e in episodes]).astype(np.uint8)
                r=np.concatenate([e['rewards'] for e in episodes]).astype(np.float32)
                t=np.concatenate([e['terminals'] for e in episodes]).astype(bool)
                assert np.all(np.isfinite(r)) and r.min()>=0
                positive+=int((r>0).sum());terminals+=int(t.sum());max_reward=max(max_reward,float(r.max()))
                a.tofile(af);r.tofile(rf);t.tofile(tf)
                lengths.extend(len(e['actions']) for e in episodes)
                if start%2048==0: print(game,split,start+len(episodes),sum(lengths),flush=True)
        np.save(root/f'{split}.lengths.npy',np.array(lengths))
        stats=dict(episodes=total,transitions=sum(lengths),eligible_episodes=sum(n>=16 for n in lengths),
                   positive_rewards=positive,max_reward=max_reward,natural_terminals=terminals,collection_seed=key)
        (root/f'{split}.json').write_text(json.dumps(stats,indent=2))
        print(game,split,stats,flush=True)

class Replay:
    def __init__(self,game,split,seed):
        root=Path('data')/game
        self.lengths=np.load(root/f'{split}.lengths.npy')
        self.offsets=np.r_[0,np.cumsum(self.lengths[:-1])]
        self.z=np.memmap(root/f'{split}.z',dtype=np.float16,mode='r').reshape(-1,128)
        self.a=np.memmap(root/f'{split}.a',dtype=np.uint8,mode='r')
        self.r=np.memmap(root/f'{split}.r',dtype=np.float32,mode='r')
        self.t=np.memmap(root/f'{split}.t',dtype=bool,mode='r')
        assert len(self.z)==len(self.a)+len(self.lengths) and len(self.a)==len(self.r)==len(self.t)==self.lengths.sum()
        self.eligible=np.flatnonzero(self.lengths>=16)
        self.rng=np.random.default_rng(seed)
    def batch(self,n):
        ep=self.rng.choice(self.eligible,n)
        s=self.rng.integers(0,self.lengths[ep]-16+1)+self.offsets[ep]
        ix=s[:,None]+np.arange(16)
        iz=(s+ep)[:,None]+np.arange(17)
        return self.z[iz].astype(np.float32), self.a[ix].astype(np.int32), self.r[ix[:,-1]], self.t[ix[:,-1]]
if __name__=='__main__': build(sys.argv[1])
