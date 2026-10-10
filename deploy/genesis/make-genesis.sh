#!/bin/bash
# Builds the genesis file of the PUBLIC testnet. Run it on a trusted machine, with the release tools built
# (cargo build --release --bin vearl-genesis --bin vearl-cli).
#
#   FAUCET_ADDRESS=vrl1... bash deploy/genesis/make-genesis.sh
#
# FAUCET_ADDRESS is the address of a DEDICATED wallet created OFFLINE (vearl-wallet new --file faucet.json; write the 24 words on paper).
# Only the address goes into the genesis, the phrase never leaves the machine that created it.
# Optional variables (defaults in brackets):
#   NETWORK [vearl-testnet-1]   LAUNCH_UTC [2026-10-23T16:00:00Z, i.e. 18:00 in Paris]   HASHRATE [100000000 tile attempts/s]
#   FAUCET_AMOUNT [1000000]     EXTRA_ALLOCS ["vrl1...=100 vrl1...=50"]   BIN_DIR [vearl-core/target/release]
# Output (deploy/genesis/out/, not committed): genesis.json, summary.txt, check.txt, genesis.sha256.
# The launch time is when the network STARTS: the difficulty rule (ASERT) is anchored on it. If the launch slips, build a new file
# (a new timestamp is a new genesis hash, and every node must use the same file).
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=${BIN_DIR:-vearl-core/target/release}
: "${FAUCET_ADDRESS:?set FAUCET_ADDRESS to the address (vrl1...) of the faucet wallet created offline}"
NETWORK=${NETWORK:-vearl-testnet-1}
LAUNCH_UTC=${LAUNCH_UTC:-2026-10-23T16:00:00Z}
HASHRATE=${HASHRATE:-100000000}
FAUCET_AMOUNT=${FAUCET_AMOUNT:-1000000}
EXTRA_ALLOCS=${EXTRA_ALLOCS:-}
OUT=deploy/genesis/out
mkdir -p "$OUT"

# 1. no address of the genesis may come from a PUBLIC devnet seed (seed:0 .. seed:255): anyone could spend it
ALL="$FAUCET_ADDRESS"
for e in $EXTRA_ALLOCS; do ALL="$ALL ${e%%=*}"; done
for n in $(seq 0 255); do
  PUB=$("$BIN/vearl-cli" address "seed:$n")
  for a in $ALL; do
    if [ "$a" = "$PUB" ]; then echo "REFUSED: $a is the address of the public devnet seed:$n, anybody can spend it" >&2; exit 1; fi
  done
done

# 2. the start must be in the future
NOW=$(date -u +%s)
START=$(date -u -d "$LAUNCH_UTC" +%s)
if [ "$START" -le "$NOW" ]; then echo "REFUSED: LAUNCH_UTC=$LAUNCH_UTC is not in the future" >&2; exit 1; fi

ARGS=(--network "$NETWORK" --preset public --faucet "$FAUCET_ADDRESS" --faucet-amount "$FAUCET_AMOUNT" --launch "$LAUNCH_UTC" --hashrate "$HASHRATE")
for e in $EXTRA_ALLOCS; do ARGS+=(--alloc "$e"); done
"$BIN/vearl-genesis" "${ARGS[@]}" > "$OUT/genesis.json" 2> "$OUT/summary.txt"
"$BIN/vearl-genesis" --check "$OUT/genesis.json" > "$OUT/check.txt"
( cd "$OUT" && sha256sum genesis.json > genesis.sha256 )

cat "$OUT/summary.txt"
echo "file SHA-256     : $(cut -d' ' -f1 "$OUT/genesis.sha256")"
echo "starts in        : $(( (START - NOW) / 3600 )) h $(( (START - NOW) % 3600 / 60 )) min"
echo
echo "Next: read the summary twice (faucet address, launch time, n = 1024), then copy genesis.json to every node and publish it with its SHA-256."
