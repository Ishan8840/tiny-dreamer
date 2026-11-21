set -eu
root=$(cd "$(dirname "$0")" && pwd)
py=/home/ubuntu/tiny-dreamer/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda
arm() (
  cd "$root/$1"; export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=$2
  if [ "$1" = Breakout ]; then
    "$py" -u retrain_crossgame.py Breakout >tokenizer.log 2>&1
  fi
  "$py" -u pretrain.py 5 >pretrain.log 2>&1
  "$py" -u online.py 0 5 >online.log 2>&1
  "$py" evaluate_horizon_policy.py seed-0 >heldout.log 2>&1
  "$py" diagnostic.py seed-0 >diagnostic.log 2>&1
  if [ "$1" = Asterix ]; then
    mkdir coin-eval; (cd coin-eval && "$py" -u ../coins.py >eval.log 2>&1)
  else
    "$py" matched.py 5 >rollout.log 2>&1
  fi
  touch complete
)
arm Asterix 0 & a=$!
arm Breakout 1 & b=$!
status=0;wait "$a" || status=1;wait "$b" || status=1
exit "$status"
