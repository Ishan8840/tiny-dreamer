#!/bin/bash
set -eu
game=$1
export CUDA_VISIBLE_DEVICES=$2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
while test ! -f "dynamics/$game/complete"; do sleep 5; done
~/tiny-dreamer/.venv/bin/python -u scalar_world.py "$game" > "$game-world.log" 2>&1
