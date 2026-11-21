set -eu
cd /workspace/asterix-natural5-esIDhLRY
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda PYTHONPATH="$PWD" JAX_COMPILATION_CACHE_DIR=/workspace/online-upgrades-b7nIUbAw/jax-cache
py=/workspace/online-upgrades-b7nIUbAw/.venv/bin/python
"$py" -u pretrain.py 5 >pretrain.log 2>&1
"$py" -u online.py 0 >online.log 2>&1
"$py" evaluate_horizon_policy.py seed-0
"$py" diagnostic.py seed-0
"$py" report.py
mkdir -p coin-eval
cd coin-eval
"$py" -u ../coins.py >eval.log 2>&1
echo complete
