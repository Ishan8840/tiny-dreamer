from tempfile import TemporaryFile
import jax
import numpy as np
from train import collect_episodes,encode_frames

def dataset(env, params, encoder, key, count):
    # File-backed latents keep long Freeway episodes out of device memory.
    storage=TemporaryFile(dir='.')
    aa,tt,lengths=[],[],[]
    for i in range(0,count,250):
        episodes,_=collect_episodes(env,params,jax.random.fold_in(key,i),min(count-i,250))
        raw=np.concatenate([ep['observations'] for ep in episodes])
        for offset in range(0,len(raw),512):
            part=raw[offset:offset+512]
            padded=np.resize(part,(512,*part.shape[1:]))
            np.asarray(encode_frames(encoder,padded))[:len(part)].astype(np.float32).tofile(storage)
        for ep in episodes:
            aa.append(ep['actions'].astype(np.int32))
            tt.append(np.stack([ep['rewards'],ep['terminals']],axis=1))
            lengths.append(len(ep['actions']))
    storage.flush()
    z=np.memmap(storage,dtype=np.float32,mode='r',shape=(sum(lengths)+len(lengths),64))
    return z,np.concatenate(aa),np.concatenate(tt),np.asarray(lengths)

def episodes_from(data):
    z,a,t,lengths=data
    episodes=[]
    start=0
    for i,n in enumerate(lengths):
        if n>=5:
            episodes.append(dict(observations=z[start+i:start+i+n+1],actions=a[start:start+n],rewards=t[start:start+n,0],terminals=t[start:start+n,1]))
        start+=n
    return episodes
