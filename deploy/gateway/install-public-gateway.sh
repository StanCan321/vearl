#!/bin/bash
# Installs the PUBLIC job gateway on the seed server (the service behind https://vearl.org/compute/ once the testnet runs).
#   bash deploy/gateway/install-public-gateway.sh [--cli PATH_TO_LINUX_VEARL_CLI]
# Needs the Linux (musl) build of vearl-cli that has `l2-who` (rebuilt with the recipe of docs/06 §3 ter; default path of the last build:
# /c/vt/musl-cli-target/x86_64-unknown-linux-musl/release/vearl-cli).
#
# What it does (nothing starts, nothing public changes): creates the service user, copies the gateway and the CLI, creates the SERVICE WALLET on the
# server (a throw-away testnet wallet: its random password stays in /etc/vearl-gateway/service-password, root only; the 24 words are not kept: the
# wallet only ever holds test tokens that you can top up), installs the systemd units (gateway + peer-list timer), enables the timer.
# It prints the service wallet's address. At the launch, once blocks exist:
#   1. fund it from the faucet wallet (you type the faucet password, on your PC):
#        vearl-wallet send --file ~/vearl-faucet/faucet.json --node seed.vearl.org:9000 --to <ADDRESS> --amount 100000
#   2. on the server:  systemctl enable --now vearl-gateway
#   3. open the page:  bash deploy/deploy-site.sh --nginx --gw-live --api-live --with-explorer --with-compute --with-leaderboard --with-participate --with-faucet
# Each free job locks 2 VRL of escrow (10 % burned, the rest paid to the miner): 100,000 VRL is about 50,000 jobs.
set -euo pipefail
cd "$(dirname "$0")/../.."
KEY="-i ${VEARL_SSH_KEY:-$HOME/.ssh/id_ed25519} -o IdentitiesOnly=yes"
HOST=${VEARL_SSH_HOST:-root@your-server}
CLI=/c/vt/musl-cli-target/x86_64-unknown-linux-musl/release/vearl-cli
[ "${1:-}" = "--cli" ] && CLI=$2
[ -f "$CLI" ] || { echo "no Linux vearl-cli at $CLI" >&2; exit 1; }
scp -q $KEY "$CLI" $HOST:/root/vearl-cli.new
scp -q $KEY deploy/gateway/vearl-gateway.py deploy/systemd/vearl-gateway.service deploy/systemd/vearl-gateway-peers.service deploy/systemd/vearl-gateway-peers.timer $HOST:/root/
# the new CLI must know l2-who: it is run once on the server against the seed node with a job id that does not exist ("unknown job" = the command exists)
ssh $KEY $HOST 'chmod +x /root/vearl-cli.new; /root/vearl-cli.new l2-who 127.0.0.1:9000 0000000000000000000000000000000000000000000000000000000000000000 2>&1 | grep -q -E "unknown job|worker = "'   || { echo "REFUSED: this vearl-cli does not know l2-who (rebuild it from the current source)" >&2; exit 1; }
ssh $KEY $HOST 'set -e
  id vearlgw >/dev/null 2>&1 || useradd --system --home /var/lib/vearl-gateway-public --shell /usr/sbin/nologin vearlgw
  install -d -o vearlgw -g vearlgw -m 0750 /var/lib/vearl-gateway-public
  install -d -o root -g vearlgw -m 0750 /etc/vearl-gateway
  install -d -o root -g root -m 0755 /opt/vearl-gateway-public
  install -m 755 /root/vearl-cli.new /opt/vearl-gateway-public/vearl-cli
  install -m 644 /root/vearl-gateway.py /opt/vearl-gateway-public/vearl-gateway.py
  install -m 644 /root/vearl-gateway.service /root/vearl-gateway-peers.service /root/vearl-gateway-peers.timer /etc/systemd/system/
  rm -f /root/vearl-cli.new /root/vearl-gateway.py /root/vearl-gateway*.service /root/vearl-gateway-peers.timer
  if [ ! -f /etc/vearl-gateway/service-wallet.json ]; then
    umask 077
    head -c 24 /dev/urandom | od -An -tx1 | tr -d " \n" > /etc/vearl-gateway/service-password
    VEARL_WALLET_PASSWORD="$(cat /etc/vearl-gateway/service-password)" /opt/vearl/bin/vearl-wallet new --file /etc/vearl-gateway/service-wallet.json >/dev/null
    chown root:vearlgw /etc/vearl-gateway/service-wallet.json; chmod 640 /etc/vearl-gateway/service-wallet.json
    echo "service wallet created"
  else echo "service wallet already there (kept)"; fi
  systemctl daemon-reload
  systemctl enable --now vearl-gateway-peers.timer >/dev/null 2>&1
  echo "service wallet address: $(/opt/vearl/bin/vearl-wallet address --file /etc/vearl-gateway/service-wallet.json)"
  systemctl is-enabled vearl-gateway >/dev/null 2>&1 && echo "(vearl-gateway is already enabled)" || echo "vearl-gateway installed, NOT started (start it after funding the wallet)"'
