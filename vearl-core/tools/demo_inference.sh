#!/usr/bin/env bash
# End-to-end demonstration of USEFULNESS: a client has real images (handwritten digits) classified by an int8 CNN
# executed by a GPU worker of the network; each operation is verified (Freivalds + recomputation), the worker is only paid afterwards.
# Usage (from vearl-core, after `cargo build --release --features gpu --bins`): bash tools/demo_inference.sh [NB_SAMPLES]
set -euo pipefail
B=${B:-target/release}; N=${1:-4}; D=$(mktemp -d); MODEL=models/digits-cnn.json
$B/node --listen 127.0.0.1:9300 --mine-seed 102 --worker --gpu --api 127.0.0.1:8700 --data-dir "$D" > "$D/node.log" 2>&1 &
NODE=$!; trap 'kill $NODE 2>/dev/null; rm -rf "$D"' EXIT
sleep 6
$B/vearl-cli status 127.0.0.1:9300 | head -1
for i in $(seq 0 $((N-1))); do
  out=$($B/vearl-cli prog-submit 127.0.0.1:9300 127.0.0.1:9300 seed:100 20 $i model:$MODEL)
  id=$(echo "$out" | sed -n 's/^job_id = //p')
  for t in $(seq 1 60); do
    st=$($B/vearl-cli prog-status 127.0.0.1:9300 $id); echo "$st" | grep -q "posted: 8" && break; sleep 1
  done
  echo "--- sample $i (job ${id:0:8}) ---"
  $B/vearl-cli prog-finish 127.0.0.1:9300 127.0.0.1:9300 seed:100 $id $i model:$MODEL
done
