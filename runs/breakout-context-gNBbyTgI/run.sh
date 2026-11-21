set -eu
cd /workspace/breakout-context-gNBbyTgI
root=$PWD
py=/workspace/online-upgrades-b7nIUbAw/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda PYTHONPATH="$root" JAX_COMPILATION_CACHE_DIR=/workspace/online-upgrades-b7nIUbAw/jax-cache
for c in 5 16; do
  mkdir -p "$root/c$c"
  cd "$root/c$c"
  "$py" -u ../pretrain.py "$c" >pretrain.log 2>&1
  "$py" -u ../online.py 0 "$c" >online.log 2>&1
  "$py" ../evaluate_horizon_policy.py seed-0 >eval.log 2>&1
  "$py" ../diagnostic.py seed-0 >diagnostic.log 2>&1
done
echo complete
