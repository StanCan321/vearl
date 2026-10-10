#!/bin/bash
# Refreshes the public leaderboard of the testnet contributors program (docs/19) from the local node. Runs every hour (vearl-rewards.timer).
# Writes /var/www/vearl-rewards/{leaderboard.json,leaderboard.csv,blocks.json}, served by nginx at https://vearl.org/rewards/.
# Optional files: /etc/vearl/rewards-registered.json (made by tools/verify_registration.py) and /etc/vearl/rewards-excluded.txt.
set -euo pipefail
START=${VEARL_TESTNET_START:-2026-10-23T16:00:00Z}
OUT=/var/www/vearl-rewards
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
ARGS=(--node 127.0.0.1:8545 --start "$START" --out "$TMP" --dump-blocks)
[ -s /etc/vearl/rewards-registered.json ] && ARGS+=(--registered /etc/vearl/rewards-registered.json)
[ -s /etc/vearl/rewards-excluded.txt ] && ARGS+=(--exclude /etc/vearl/rewards-excluded.txt)
python3 /opt/vearl/tools/testnet_rewards.py "${ARGS[@]}"
for f in leaderboard.json leaderboard.csv blocks.json; do
  install -m 0644 "$TMP/$f" "$OUT/$f.new"
  mv "$OUT/$f.new" "$OUT/$f"
done
