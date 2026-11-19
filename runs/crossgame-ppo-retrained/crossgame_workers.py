from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue, Empty
import json
import subprocess

root = Path.cwd()
games = ('Seaquest','Freeway','Asterix','SpaceInvaders')
for game in games:
    assert json.loads((root/'tokenizers'/game/'metrics.json').read_text())['game'] == game
jobs = Queue()
for seed in range(3):
    for game in games:
        jobs.put((game,seed))
def worker(gpu):
    while True:
        try:
            game,seed=jobs.get_nowait()
        except Empty:
            return
        proc=subprocess.run(['bash','run_crossgame_ppo.sh',game,str(gpu),str(seed)])
        print(game,seed,'gpu',gpu,'exit',proc.returncode,flush=True)
        if proc.returncode:
            raise RuntimeError(f'{game} seed {seed} failed')
with ThreadPoolExecutor(8) as pool:
    list(pool.map(worker,range(8)))
