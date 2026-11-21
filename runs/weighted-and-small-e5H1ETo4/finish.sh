set -eu
cd "$(dirname "$0")"; root=$PWD
py=/home/ubuntu/tiny-dreamer/.venv/bin/python
export CUDA_VISIBLE_DEVICES=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda
while [ ! -f Asterix/complete ]; do sleep 15; done
mkdir -p Asterix/reference-eval
(cd Asterix/reference-eval && PYTHONPATH="$root/Asterix" "$py" ../reference-head.py >eval.log 2>&1)
while [ ! -f Breakout/complete ]; do sleep 15; done
"$py" report.py
