#!/usr/bin/env bash
# Shared by the Hive scripts of the Vearl package (sourced, never run by itself).
#
# The package folder (/hive/miners/custom/vearl) holds only the programs and is replaced at every update. Everything that must survive an
# update (the wallet, the settings, the chain data, the logs) lives in VEARL_DATA, a persistent folder: /hive-config/vearl on HiveOS.
# The programs are linked into it, because vearl-miner looks for its node and its network files in its --home folder.

VEARL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${VEARL_DATA:-}" ]]; then
  if [[ -d /hive-config ]]; then
    VEARL_DATA=/hive-config/vearl
  else
    VEARL_DATA="$VEARL_DIR/home"   # not on HiveOS (tests, other Linux): next to the programs
  fi
fi
export VEARL_DIR VEARL_DATA

vearl_log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') [hive] $*"; }

# Creates VEARL_DATA and links the programs and the network files of the package into it (refreshed at every start, so an update of the
# package is picked up). Files that belong to the package only: no wallet, no settings.
vearl_prepare_home() {
  mkdir -p "$VEARL_DATA"
  chmod 700 "$VEARL_DATA" 2>/dev/null || true
  local p b
  for p in "$VEARL_DIR"/*; do
    b="$(basename "$p")"
    case "$b" in
      h-*|common.sh|setup.sh|README*|home|*.md) continue ;;
    esac
    ln -sfn "$p" "$VEARL_DATA/$b"
  done
}

# The Vearl address written in a wallet file (it is stored in clear, no password needed), or nothing.
vearl_wallet_address() {
  local f="${1:-$VEARL_DATA/wallet.json}"
  [[ -f "$f" ]] || return 0
  if command -v jq >/dev/null 2>&1; then
    jq -r '.address // empty' "$f" 2>/dev/null
  else
    sed -n 's/.*"address"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$f" | head -n 1
  fi
}
