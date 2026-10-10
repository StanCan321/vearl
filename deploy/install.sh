#!/usr/bin/env bash
# Installs a VEARL testnet node on Ubuntu/Debian. Run as root from the repository root.
#   ./deploy/install.sh                 builds (cargo) and installs the binaries + systemd units (starts nothing)
#   ./deploy/install.sh --bin DIR       installs ALREADY BUILT binaries (folder containing node, vearl-cli, vearl-wallet,
#                                       vearl-genesis, vearl-faucet: see deploy/release/make_release.py): no cargo on the server
#   --ports P2P,API,FAUCET              ports instead of 9000,8545,8081 (to be used with --prefix)
#   --prefix NAME                       installs under another name (/opt/NAME, /etc/NAME, /var/lib/NAME, units NAME-node/NAME-faucet):
#                                       useful for a REHEARSAL next to an existing network; ports to be changed afterwards in the unit
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "must be run as root"; exit 1; }
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR=""
NAME=vearl
P2P=9000; API=8545; FAUCET=8081
while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN_DIR="$2"; shift 2 ;;
    --prefix) NAME="$2"; shift 2 ;;
    --ports) IFS=, read -r P2P API FAUCET <<< "$2"; shift 2 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done
id "$NAME" >/dev/null 2>&1 || useradd --system --home "/var/lib/$NAME" --shell /usr/sbin/nologin "$NAME"
install -d -o "$NAME" -g "$NAME" -m 0750 "/var/lib/$NAME"
install -d -o root  -g "$NAME" -m 0750 "/etc/$NAME" "/opt/$NAME/bin" "/opt/$NAME/docs"
if [ -z "$BIN_DIR" ]; then
  ( cd "$REPO/vearl-core" && sudo -u "${SUDO_USER:-root}" env PATH="$PATH:$HOME/.cargo/bin" cargo build --release )
  BIN_DIR="$REPO/vearl-core/target/release"
fi
for b in node vearl-cli vearl-wallet vearl-genesis vearl-faucet; do
  [ -f "$BIN_DIR/$b" ] || { echo "missing binary: $BIN_DIR/$b"; exit 1; }
  install -m 0755 "$BIN_DIR/$b" "/opt/$NAME/bin/"
done
cp -r "$REPO/docs/." "/opt/$NAME/docs/"
for u in node faucet; do
  sed "s#/opt/vearl#/opt/$NAME#g; s#/etc/vearl#/etc/$NAME#g; s#/var/lib/vearl#/var/lib/$NAME#g; s#User=vearl#User=$NAME#; s#Group=vearl#Group=$NAME#; s#vearl-node.service#$NAME-node.service#g; s#0.0.0.0:9000#0.0.0.0:$P2P#; s#127.0.0.1:8545#127.0.0.1:$API#; s#127.0.0.1:9000#127.0.0.1:$P2P#; s#127.0.0.1:8081#127.0.0.1:$FAUCET#" \
    "$REPO/deploy/systemd/vearl-$u.service" > "/etc/systemd/system/$NAME-$u.service"
done
install -m 0755 "$REPO/deploy/monitor/vearl-watch.py" "/opt/$NAME/bin/"
sed "s#/opt/vearl#/opt/$NAME#g; s#/etc/vearl#/etc/$NAME#g; s#/var/lib/vearl#/var/lib/$NAME#g; s#User=vearl#User=$NAME#; s#127.0.0.1:8545#127.0.0.1:$API#; s#--name seed#--name $NAME#"   "$REPO/deploy/monitor/vearl-watch.service" > "/etc/systemd/system/$NAME-watch.service"
sed "s#vearl-watch#$NAME-watch#g" "$REPO/deploy/monitor/vearl-watch.timer" > "/etc/systemd/system/$NAME-watch.timer"
systemctl daemon-reload
cat <<MSG

Installed under the name "$NAME". Next steps (see docs/06-testnet-runbook.md):
  1. faucet wallet, created OFFLINE on a trusted machine:  vearl-wallet new --file faucet-wallet.json
     (only faucet-wallet.json is copied into /etc/$NAME/, the 24 words stay on paper)
  2. genesis:  FAUCET_ADDRESS=vrl1... bash deploy/genesis/make-genesis.sh   then copy genesis.json into /etc/$NAME/genesis.json
  3. echo -n 'password' > /etc/$NAME/faucet-password ; chmod 600 /etc/$NAME/faucet-password
  4. systemctl enable --now $NAME-node $NAME-faucet $NAME-watch.timer
  5. (optional) put a Discord webhook URL in /etc/$NAME/alert-webhook (chmod 640, group $NAME): the health check posts there when the node stops, has no peer or the height is stuck
MSG
