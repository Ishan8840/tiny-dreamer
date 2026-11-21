set -eu
root=$(cd "$(dirname "$0")" && pwd)
cd "$root"
py=/home/ubuntu/tiny-dreamer/.venv/bin/python
export PYTHONPATH="$root" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda
until test -s tokenizers/Asterix/metrics.json; do sleep 5; done
CUDA_VISIBLE_DEVICES=0 "$py" -u pretrain.py 0 >prepare.log 2>&1
arm() (
  export CUDA_VISIBLE_DEVICES=$1
  h=$2
  mkdir "h$h"; cd "h$h"
  "$py" -u ../pretrain.py "$h" >pretrain.log 2>&1
  "$py" -u ../online.py 0 "$h" >online.log 2>&1
  "$py" ../evaluate_horizon_policy.py seed-0 >evaluate.log 2>&1
  "$py" ../diagnostic.py seed-0 >diagnostic.log 2>&1
  touch complete
)
arm 0 1 & p1=$!
arm 1 3 & p3=$!
status=0
wait "$p1" || status=1
wait "$p3" || status=1
exit "$status"
