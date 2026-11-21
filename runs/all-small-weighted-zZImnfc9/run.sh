set -eu
root=$(cd "$(dirname "$0")" && pwd)
py=/home/ubuntu/tiny-dreamer/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda PYTHONPATH="$root"
game() (
 export MINATAR_GAME="$1" CUDA_VISIBLE_DEVICES="$2"
 mkdir -p "$root/$1"; cd "$root/$1"
 case "$1" in
 Asterix) cp ~/asterix-small-horizons-RL77ot7V/tokenizers/Asterix/best.msgpack tokenizer.msgpack;;
 Breakout) cp ~/weighted-and-small-e5H1ETo4/Breakout/tokenizers/Breakout/best.msgpack tokenizer.msgpack;;
 *) "$py" -u ../retrain_crossgame.py "$1" >tokenizer.log 2>&1
    cp "tokenizers/$1/best.msgpack" tokenizer.msgpack;;
 esac
 "$py" -u ../pretrain.py 5 >pretrain.log 2>&1
 "$py" -u ../online.py 0 >online.log 2>&1
 "$py" ../evaluate_horizon_policy.py seed-0 >heldout.log 2>&1
 "$py" ../diagnostic.py seed-0 >diagnostic.log 2>&1
 touch complete
)
games=(Asterix Breakout SpaceInvaders Seaquest Freeway)
pids=()
for i in "${!games[@]}"; do game "${games[i]}" "$i" & pids+=("$!"); done
rc=0;for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
exit "$rc"
