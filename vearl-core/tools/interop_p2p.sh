#!/usr/bin/env bash
# Network interoperability test: a REAL Rust node mines, transfers signed by the wallet are sent, then the
# PYTHON light client (tools/light_client.py) connects over Noise XX, downloads the whole chain and re-validates it.
# Usage: tools/interop_p2p.sh   (from vearl-core, after `cargo build --release --bins` ; pip install noiseprotocol blake3 numpy cryptography)
set -euo pipefail
B=${B:-target/release}; D=$(mktemp -d); export VEARL_WALLET_PASSWORD=motdepasse-test-123
$B/vearl-wallet new --file "$D/f.json" >/dev/null 2>&1; $B/vearl-wallet new --file "$D/g.json" >/dev/null 2>&1
$B/vearl-genesis --network interop --faucet "$($B/vearl-wallet address --file "$D/f.json")" --retarget Fixed --target-divisor 64 > "$D/genesis.json"
$B/node --genesis "$D/genesis.json" --listen 127.0.0.1:9201 --mine-seed 1 --insecure-devnet-keys --data-dir "$D/data" > "$D/node.log" 2>&1 &
NODE=$!; trap 'kill $NODE 2>/dev/null; rm -rf "$D"' EXIT
sleep 12
for i in 1 2 3; do $B/vearl-wallet send --file "$D/f.json" --node 127.0.0.1:9201 --to "$($B/vearl-wallet address --file "$D/g.json")" --amount 1.5 >/dev/null; sleep 3; done
sleep 8
python3 tools/light_client.py "$D/genesis.json" 127.0.0.1:9201 --min-height 20
# header-based light client: stateless tile proofs + balance of the recipient proven against accounts_root
python3 tools/light_client.py "$D/genesis.json" 127.0.0.1:9201 --headers "$($B/vearl-wallet address --file "$D/g.json")" --min-height 20
