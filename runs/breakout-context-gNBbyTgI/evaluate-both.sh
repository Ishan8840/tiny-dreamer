set -eu
cd /workspace/breakout-context-gNBbyTgI
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda PYTHONPATH="$PWD" JAX_COMPILATION_CACHE_DIR=/workspace/online-upgrades-b7nIUbAw/jax-cache
while ! grep -qx complete launcher.log; do sleep 30; done
for c in 5 16; do
  /workspace/online-upgrades-b7nIUbAw/.venv/bin/python -u matched.py "$c"
done
touch evaluated
