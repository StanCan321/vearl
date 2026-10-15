#!/bin/bash
# Installs the testnet faucet on the seed server, with the DEDICATED faucet wallet made offline (the one whose address is in genesis.json).
#   bash deploy/faucet/install-faucet.sh [path/to/faucet.json]        (default: ~/vearl-faucet/faucet.json)
#   bash deploy/faucet/install-faucet.sh --check [path/to/faucet.json]  checks the wallet file and the password on this PC only, installs nothing
#
# What it does, in this order, and stops at the first problem:
#   1. the address inside the wallet file must be the faucet address of deploy/genesis/out/genesis.json (the 1,000,000 VRL of the genesis)
#   2. it asks the wallet password (hidden, never in a command line nor in a log) and proves it opens the file, on this PC
#   3. it copies the ENCRYPTED wallet file to /etc/vearl/faucet-wallet.json (root:vearl, 640) and sends the password through ssh's standard
#      input to /etc/vearl/faucet-password (root only, 600): the service reads it through systemd's LoadCredential, never as an argument
#   4. it installs deploy/systemd/vearl-faucet.service, starts it, and shows its first log lines
# The 24 recovery words are never needed and never touched. Nobody but you should ever type this password.
# Afterwards the faucet page is published with  bash deploy/deploy-site.sh --nginx --with-explorer --with-compute --with-leaderboard --with-participate --with-faucet
set -euo pipefail
cd "$(dirname "$0")/../.."
CHECK=0; [ "${1:-}" = "--check" ] && { CHECK=1; shift; }
WALLET=${1:-$HOME/vearl-faucet/faucet.json}
KEY="-i ${VEARL_SSH_KEY:-$HOME/.ssh/id_ed25519} -o IdentitiesOnly=yes"
HOST=${VEARL_SSH_HOST:-root@your-server}
GENESIS=${GENESIS:-deploy/genesis/out/genesis.json}
WTOOL=${VEARL_WALLET_BIN:-}
if [ -z "$WTOOL" ]; then   # the wallet tool of this PC: a build of the repository, or an unpacked release (vearl-wallet.exe / vearl-wallet)
  for c in vearl-core/target/release/vearl-wallet.exe vearl-core/target/release/vearl-wallet "$HOME"/Desktop/vearl-*/vearl-wallet.exe "$HOME"/vearl-*/vearl-wallet; do
    [ -f "$c" ] && { WTOOL=$c; break; }
  done
fi
[ -f "$WALLET" ] || { echo "no wallet file at $WALLET" >&2; exit 1; }
PY=; for c in python3 python py; do "$c" -c "import json" >/dev/null 2>&1 && { PY=$c; break; }; done   # (a Windows shim that only opens the Store does not count)
[ -n "$PY" ] || { echo "python not found" >&2; exit 1; }
# 1. the wallet is the genesis faucet
ADDR=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['address'])" "$WALLET")
GADDR=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['alloc'][0]['address'])" "$GENESIS")
echo "wallet address : $ADDR"
echo "genesis faucet : $GADDR"
[ "$ADDR" = "$GADDR" ] || { echo "REFUSED: this wallet is not the faucet of the genesis" >&2; exit 1; }
echo "ok: same address."
# 2. the password opens the wallet (checked here, with the wallet tool)
[ -n "$WTOOL" ] && [ -f "$WTOOL" ] || { echo "vearl-wallet not found: set VEARL_WALLET_BIN" >&2; exit 1; }
printf 'Faucet wallet password (hidden): '
read -rs PW; echo
[ -n "$PW" ] || { echo "empty password" >&2; exit 1; }
# (pq-address unlocks the wallet: a wrong password makes it fail; `address` alone only reads the public field)
GOT=$(VEARL_WALLET_PASSWORD="$PW" "$WTOOL" pq-address --file "$WALLET" 2>/dev/null | grep -o 'vrlq1[0-9a-z]*' | head -1 || true)
[ -n "$GOT" ] || { echo "REFUSED: that password does not open this wallet" >&2; exit 1; }
echo "ok: the password opens the wallet."
[ $CHECK = 1 ] && { echo "check only: nothing was installed."; exit 0; }
# 3 + 4. the server
scp -q $KEY "$WALLET" $HOST:/root/faucet-wallet.json.new
scp -q $KEY deploy/systemd/vearl-faucet.service $HOST:/root/vearl-faucet.service.new
printf '%s\n' "$PW" | ssh $KEY $HOST 'set -e
  umask 077; cat > /root/faucet-password.new; chmod 600 /root/faucet-password.new
  install -m 600 -o root -g root /root/faucet-password.new /etc/vearl/faucet-password
  install -m 640 -o root -g vearl /root/faucet-wallet.json.new /etc/vearl/faucet-wallet.json
  install -m 644 /root/vearl-faucet.service.new /etc/systemd/system/vearl-faucet.service
  rm -f /root/faucet-password.new /root/faucet-wallet.json.new /root/vearl-faucet.service.new
  systemctl daemon-reload; systemctl enable vearl-faucet >/dev/null 2>&1; systemctl restart vearl-faucet; sleep 4
  systemctl is-active vearl-faucet; journalctl -u vearl-faucet -n 6 --no-pager | cut -c1-200'
unset PW
echo "Faucet installed. Test:  curl -s -X POST https://vearl.org/drip -d '{}'   (needs --with-faucet and --nginx once, see the header of this file)"
