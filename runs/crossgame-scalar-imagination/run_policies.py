from pathlib import Path
import os
import subprocess
import time
games=('Asterix','Freeway','Seaquest','SpaceInvaders')
jobs=[(g,s) for s in range(3) for g in games]
running={}
while jobs or running:
    for gpu,(p,g,s,log) in list(running.items()):
        if p.poll() is None: continue
        log.close()
        print(g,s,'exit',p.returncode,flush=True)
        del running[gpu]
        if p.returncode: raise RuntimeError(f'{g}-s{s} failed')
    available=[j for j in jobs if (Path('world')/j[0]/'complete').is_file()]
    for gpu in range(8):
        if gpu in running or not available: continue
        if gpu<4 and not (Path('world')/games[gpu]/'complete').is_file(): continue
        if gpu==7 and not Path('dynamics/SpaceInvaders/complete').is_file(): continue
        g,s=available.pop(0);jobs.remove((g,s))
        log=open(f'{g}-s{s}.log','w')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        p=subprocess.Popen([str(Path.home()/'tiny-dreamer/.venv/bin/python'),'-u','crossgame_policy.py',g,str(s)],env=env,stdout=log,stderr=subprocess.STDOUT)
        running[gpu]=(p,g,s,log)
        print('started',g,s,'gpu',gpu,flush=True)
    time.sleep(5)
