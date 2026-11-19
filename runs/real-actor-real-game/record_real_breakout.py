from pathlib import Path
import csv
import subprocess
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from PIL import Image, ImageDraw, ImageFont
from train import WorldModel, Actor, Critic, restore_weights, make_env

env, params = make_env()
wm = WorldModel(nnx.Rngs(0),4,3)
home = Path.home()
restore_weights(wm, home/'tiny-dreamer-heads10k-BTdi973d/runs/dynamics-outcomes/best.msgpack')
rngs = nnx.Rngs(0)
agent = nnx.Dict(actor=Actor(rngs,3,256), critic=Critic(rngs,256))
restore_weights(agent, home/'tiny-dreamer-real100kupdates-AHiSiZMf/run/best.msgpack')
encoder = wm.tokenizer.layers[0]
seed = 41001
reset_key, key = jax.random.split(jax.random.key(seed))
# Replay the first lane of the batched evaluation with the original RNG shapes.
obs, state = env.reset(jax.random.split(reset_key,64)[0], params)
current = encoder(obs[None].astype(jnp.float32))
previous = current
step_env = jax.jit(env.step_env)
score = 0.
frames, trace = [np.asarray(obs)], [(0,0.,0.)]
while True:
    key, ak, sk = jax.random.split(key,3)
    logits = agent.actor(jnp.concatenate((previous,current),axis=-1))
    action = jax.random.categorical(ak,jnp.broadcast_to(logits,(64,3)))[0]
    obs,state,reward,done,_ = step_env(jax.random.split(sk,64)[0],state,action,params)
    previous,current = current,encoder(obs[None].astype(jnp.float32))
    score += float(reward)
    frames.append(np.asarray(obs))
    trace.append((int(action),float(reward),score))
    if bool(done | env.is_truncated(state,params)):
        break
assert score == 8, score
font = ImageFont.truetype('DejaVuSans.ttf',20)
small = ImageFont.truetype('DejaVuSans.ttf',16)
images = []
for i, (frame, (_, _, points)) in enumerate(zip(frames,trace)):
    im = Image.new('RGB',(384,448),'#101a27')
    draw = ImageDraw.Draw(im)
    draw.text((20,10),'Breakout   /   real-trained actor',font=small,fill='#eeeeee')
    draw.text((20,42),f'Score  {points:.0f}',font=font,fill='#ffffff')
    draw.text((234,45),f'Step  {i}' if i<len(frames)-1 else 'Game over',font=small,fill='#cbd6df')
    for row,col in np.ndindex(10,10):
        x,y = 22+34*col,82+34*row
        paddle,ball,trail,brick=frame[row,col]
        draw.rectangle((x,y,x+32,y+32),fill='#1b2d40')
        if brick: draw.rounded_rectangle((x+1,y+2,x+31,y+30),radius=3,fill='#80bc92')
        if trail: draw.line((x+8,y+8,x+24,y+24),width=3,fill='#cb9fe1')
        if paddle: draw.rectangle((x,y+22,x+32,y+30),fill='#5abbff')
        if ball: draw.ellipse((x+6,y+5,x+26,y+25),fill='#ffc562')
    images.append(im)
with open('trajectory.csv','w') as f:
    w=csv.writer(f)
    w.writerow(['step','action','reward','score'])
    for i,t in enumerate(trace[1:],1): w.writerow((i,*t))
with subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','384x448','-r','8','-i','-','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart','episode.mp4'],stdin=subprocess.PIPE) as p:
    for im in images + [images[-1]]*8: p.stdin.write(im.tobytes())
    p.stdin.close()
    assert p.wait()==0
images[0].save('episode.gif',save_all=True,append_images=images[1:],duration=[125]*(len(images)-1)+[1125],loop=0)
print(f'seed {seed}, lane 0, {len(trace)-1} actions, return {score}',flush=True)
