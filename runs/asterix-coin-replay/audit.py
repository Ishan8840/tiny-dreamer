from pathlib import Path
import json,sys
import numpy as np
def count(lengths,rewards):
    starts=np.r_[0,np.cumsum(lengths)[:-1]]
    ids=np.repeat(np.arange(len(lengths)),lengths)
    assert len(ids)==len(rewards) and set(np.unique(rewards))<={0.,1.}
    ends=np.flatnonzero(np.arange(len(ids))-starts[ids]>=15)
    pos=ends[rewards[ends]>0];neg=ends[rewards[ends]==0]
    counts=np.bincount(ids[pos],minlength=len(lengths))
    eligible=lengths>=16
    cumulative=np.r_[0,np.cumsum(rewards)]
    return dict(episodes=len(lengths),scoring_episodes=len(np.unique(ids[rewards>0])),transitions=len(ids),pickups=int(rewards.sum()),eligible_windows=len(ends),endpoint_pickups=len(pos),windows_containing_pickup=int((cumulative[ends+1]-cumulative[ends-15]>0).sum()),episode_uniform_probability=float((counts[eligible]/(lengths[eligible]-15)).mean())),ids,pos,neg
if __name__=='__main__':
    if sys.argv[1]=='source':
        base=Path.home()/'tiny-dreamer-crossgame-imagination-CWLEBVog/data/Asterix'
        L=np.load(base/'train.lengths.npy');r=np.fromfile(base/'train.r',np.float32)
        report,ids,pos,neg=count(L,r)
        print(json.dumps(report),flush=True)
        if '--filter' in sys.argv:
            rng=np.random.default_rng(903)
            ix=rng.permutation(np.r_[pos,rng.choice(neg,3*len(pos),replace=False)])
            a=np.memmap(base/'train.a',dtype=np.uint8,mode='r');t=np.memmap(base/'train.t',dtype=bool,mode='r')
            z=np.memmap(base/'train.z',dtype=np.float16,mode='r').reshape(-1,128)
            window=ix[:,None]+np.arange(-15,1)
            assert np.all(ids[window]==ids[ix,None]) and 4*r[ix].sum()==len(ix) and len(np.unique(ix))==len(ix)
            np.savez_compressed('pickup-windows.npz',history=z[window+ids[ix,None]],actions=a[window],points=r[ix],terminal=t[ix],episode=ids[ix],endpoint=ix)
            with np.load('pickup-windows.npz') as data:
                np.testing.assert_array_equal(data['points'],r[data['endpoint']])
                assert data['history'].shape==(len(ix),16,128)
            report.update(filtered_windows=len(ix),filtered_positive=len(pos),filter_seed=903)
    else:
        import jax
        from train import make_env,collect_episodes
        env,p=make_env('Asterix')
        key,*_=jax.random.split(jax.random.key(0),4)
        episodes,_=collect_episodes(env,p,key,10000)
        report,*_=count(np.array([len(e['actions']) for e in episodes]),np.concatenate([e['rewards'] for e in episodes]))
    Path(sys.argv[1]+'.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
