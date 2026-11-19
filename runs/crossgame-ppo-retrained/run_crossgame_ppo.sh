#!/bin/bash
set -eu
cd /home/ubuntu/tiny-dreamer-crossgame-ppo-yDkAlBWH
game=$1
export CUDA_VISIBLE_DEVICES=$2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
shift 2
for seed in "$@"; do
    output="$game-s$seed"
    ~/tiny-dreamer/.venv/bin/python -u train.py --train real-actor-critic --game "$game" --tokenizer-checkpoint "tokenizers/$game/best.msgpack" --seed "$seed" --output "$output" > "$output.log" 2>&1
done
