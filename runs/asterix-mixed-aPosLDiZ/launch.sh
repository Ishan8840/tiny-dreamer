set -eu
root=$PWD
export PYTHONPATH="$root" JAX_PLATFORMS=cuda OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_COMPILATION_CACHE_DIR=/workspace/online-upgrades-b7nIUbAw/jax-cache
py=/workspace/online-upgrades-b7nIUbAw/.venv/bin/python
"$py" -u online.py 0 >online.log 2>&1
"$py" evaluate_horizon_policy.py seed-0
"$py" diagnostic.py seed-0
for GAME in Asterix Seaquest; do
 export GAME
 if [ "$GAME" = Asterix ]; then CKPT="$root/seed-0"; else CKPT=/workspace/minatar-natural5-bUODJhRX/Seaquest/seed-0; fi
 export CKPT
 mkdir -p "$root/$GAME"; cd "$root/$GAME"
 "$py" "$root/mc.py" >mc.log 2>&1
 "$py" "$root/targets.py" >targets.log 2>&1
done
echo complete
