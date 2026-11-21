set -eu
cd "$(dirname "$0")"
py=/home/ubuntu/tiny-dreamer/.venv/bin/python
for g in Asterix Breakout SpaceInvaders Seaquest Freeway; do (
  until [ -f "$g/complete" ]; do sleep 15; done
  OPENBLAS_NUM_THREADS=1 "$py" summarize.py "$g"
) & done
wait
