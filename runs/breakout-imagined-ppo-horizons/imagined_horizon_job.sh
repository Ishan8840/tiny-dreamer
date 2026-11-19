#!/bin/bash
set -eu
cd /home/ubuntu/tiny-dreamer-imagination-horizons-q9thnzAA
h=$1
seed=$2
export CUDA_VISIBLE_DEVICES=$3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
dir=h$h-s$seed
py=~/tiny-dreamer/.venv/bin/python
"$py" -u train.py --train actor-critic --policy-method ppo --batch-size 32 --imagination-horizon "$h" --seed "$seed" --model-checkpoint ~/tiny-dreamer-heads10k-BTdi973d/runs/dynamics-outcomes/best.msgpack --output "$dir" --episodes 10000 --epochs 50 --steps-per-epoch 2000 --patience 0 > "$dir.log" 2>&1
"$py" -u evaluate_horizon_policy.py "$dir" >> "$dir.log" 2>&1
