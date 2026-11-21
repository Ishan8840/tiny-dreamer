#!/bin/bash
set -eu
cd /workspace/asterix-online-Z7n3PzzZ
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda
export JAX_COMPILATION_CACHE_DIR="$PWD/jax-cache"
py=/workspace/online-upgrades-b7nIUbAw/.venv/bin/python
for s in 0; do
    echo "seed $s training $(date -u)"
    "$py" -u online.py "$s" > "seed-$s.log" 2>&1
    "$py" -u evaluate_horizon_policy.py "seed-$s" >> "seed-$s.log" 2>&1
    echo "seed $s evaluated $(date -u)"
done
for s in 0; do
    "$py" -u diagnostic.py "seed-$s" >> "seed-$s.log" 2>&1
done
$py report.py
echo "complete $(date -u)"
