from pathlib import Path
import csv
import hashlib
import json
import sys
from importlib.metadata import version
import jax
import numpy as np
from flax import nnx
from train import train, make_tokenizer, restore_weights, encode_frames
from env import make_env

game = sys.argv[1]
root = Path.cwd()
data = Path.home() / 'tiny-dreamer-size101-C1J0WKJV/data' / game
folder = root / 'tokenizers' / game
folder.mkdir(parents=True)
assert jax.default_backend() == 'gpu' and version('gymnax') == '1.0.1'
frames = np.load(data/'train.npy', mmap_mode='r')
val = np.load(data/'val.npy', mmap_mode='r')
assert (len(frames),len(val)) == (101404,11068)
env, params = make_env(game)
assert frames.shape[1:] == env.observation_space(params).shape
model = train(frames, seed=17, validation_observations=val, epochs=20, batch_size=64, output=folder, learning_rate=3e-4)
restore_weights(model,folder/'best.msgpack')
exact = tp = fp = fn = 0
bce = 0.
predict = nnx.jit(lambda net, x: net(x))
for offset in range(0,len(val),256):
    x = np.asarray(val[offset:offset+256],dtype=bool)
    logits = np.asarray(predict(model,x.astype(np.float32)),dtype=np.float64)
    pred = logits >= 0
    exact += int((pred == x).all(axis=(1,2,3)).sum())
    tp += int((pred & x).sum())
    fp += int((pred & ~x).sum())
    fn += int((~pred & x).sum())
    bce += np.logaddexp(0,np.where(x,-logits,logits)).sum()
assert np.isfinite(bce)
summary=dict(game=game,seed=17,train_frames=len(frames),validation_frames=len(val),exact_frames=exact,
             exact_frame_accuracy=exact/len(val),positive_pixel_f1=2*tp/(2*tp+fp+fn),bce=bce/val.size,
             sha256=hashlib.sha256((folder/'best.msgpack').read_bytes()).hexdigest())
(folder/'metrics.json').write_text(json.dumps(summary,indent=2))
print(summary,flush=True)
