set -eu
cd /workspace/minatar-natural5-bUODJhRX
root=$PWD
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda PYTHONPATH="$root" JAX_COMPILATION_CACHE_DIR=/workspace/online-upgrades-b7nIUbAw/jax-cache
py=/workspace/online-upgrades-b7nIUbAw/.venv/bin/python
for MINATAR_GAME in Seaquest SpaceInvaders Freeway; do
 export MINATAR_GAME
 cd "$root/$MINATAR_GAME"
 "$py" -u ../pretrain.py 5 >pretrain.log 2>&1
 "$py" -u ../online.py 0 >online.log 2>&1
 "$py" ../evaluate_horizon_policy.py seed-0 >heldout.log 2>&1
 "$py" ../diagnostic.py seed-0 >diagnostic.log 2>&1
 "$py" ../report.py >report.log 2>&1
done
echo complete
