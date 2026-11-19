#!/bin/bash
set -eu
cd /home/ubuntu/tiny-dreamer-policy-ablation-Cypp0xXu
method=$1
rate=$2
export CUDA_VISIBLE_DEVICES=$3
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for seed in 0 1 2; do
  name="${method}-${rate}-s${seed}"
  ~/tiny-dreamer/.venv/bin/python -u train.py --train real-actor-critic --policy-method "$method" --actor-learning-rate "$rate" --model-checkpoint ~/tiny-dreamer-heads10k-BTdi973d/runs/dynamics-outcomes/best.msgpack --output "$name" --batch-size 32 --real-horizon 5 --epochs 50 --steps-per-epoch 2000 --eval-episodes 64 --seed "$seed" > "$name.log" 2>&1
done
touch "${method}-${rate}.complete"
