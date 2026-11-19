import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from train import WorldModel, make_env, restore_weights, frozen_update_step, save_weights, denoise_next
from crossgame_replay import Replay

@nnx.jit
def next_mse(dynamics, z, a, key):
    noise=jax.random.normal(key,z[:,-1].shape)
    prediction=denoise_next(dynamics,z[:,:-1],a,noise,4)
    return jnp.mean(jnp.square(prediction-z[:,-1]))

def pretrained(game):
    env, params=make_env(game)
    model=WorldModel(nnx.Rngs(17),env.observation_space(params).shape[-1],env.action_space(params).n)
    restore_weights(model.tokenizer,Path.home()/f'tiny-dreamer-crossgame-ppo-yDkAlBWH/tokenizers/{game}/best.msgpack')
    return model

def main(game):
    out=Path('dynamics')/game
    out.mkdir(parents=True)
    model=pretrained(game)
    optim=nnx.Optimizer(model.dynamics,optax.adamw(3e-4,weight_decay=1e-4),wrt=nnx.Param)
    train=Replay(game,'train',17)
    val=Replay(game,'val',37)
    valid=[val.batch(64)[:2] for i in range(64)]
    best=np.inf
    stale=0
    with (out/'validation.csv').open('w') as f:
        w=csv.writer(f);w.writerow(('update','training_shortcut','validation_next_latent_mse'))
        for epoch in range(50):
            losses=[]
            for i in range(2000):
                z,a,_,_=train.batch(64)
                key=jax.random.fold_in(jax.random.key(17),epoch*2000+i)
                losses.append(frozen_update_step(model.dynamics,optim,z,a,key,1.))
            score=np.mean([float(next_mse(model.dynamics,z,a,jax.random.fold_in(jax.random.key(37),i))) for i,(z,a) in enumerate(valid)])
            loss=np.mean(losses)
            assert np.isfinite([score,loss]).all()
            step=(epoch+1)*2000
            w.writerow((step,loss,score));f.flush()
            print('update',step,'val MSE',score,flush=True)
            if score<best:
                save_weights(model,out)
                best=score;stale=0
            else:
                stale+=1
                if stale>=5: break
    (out/'complete').touch()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('game')
    main(parser.parse_args().game)
