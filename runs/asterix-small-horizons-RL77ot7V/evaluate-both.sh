set -eu
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda
py=/home/ubuntu/tiny-dreamer/.venv/bin/python
arm() (
  while [ ! -f "h$1/complete" ]; do sleep 30; done
  CUDA_VISIBLE_DEVICES="$2" "$py" -u rollout_eval.py "h$1" >"h$1/rollout-eval.log" 2>&1
)
arm 1 2 & a=$!
arm 3 3 & b=$!
wait "$a";wait "$b"
"$py" compare.py
