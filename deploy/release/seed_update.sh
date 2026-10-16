#!/bin/bash
# Upgrades the seed server to a new release (run as root on the server, from the unpacked package made by make_seed_package.py).
#   tar xzf seedpkg-rcNN.tgz && cd seedpkg-rcNN && DRY=1 bash seed_update.sh     # shows what it would do, changes nothing
#   bash seed_update.sh                                                         # does it
# Keeps the network key (/var/lib/vearl/node.key), the chain, the wallets and every configuration file. Before touching anything it copies the
# programs, the tools and the settings it will replace to /root/seed-backup-DATE, and it tells you how to go back.
# Order matters: the seed must run the new release BEFORE the miners do (the relay and the fee beacon are new messages that an older seed does not
# understand), so run this first and publish the release afterwards.
set -euo pipefail
cd "$(dirname "$0")"
DRY=${DRY:-0}
run() { if [ "$DRY" = 1 ]; then echo "[dry] $*"; else "$@"; fi }
if [ "$DRY" != 1 ] && [ "$(id -u)" != 0 ]; then echo "run as root (or DRY=1 to look)"; exit 1; fi

echo "== 1. the package is intact"
sha256sum -c SHA256SUMS --quiet && echo "ok"

B=/root/seed-backup-$(date +%Y%m%d-%H%M%S)
echo "== 2. backup in $B"
run mkdir -p "$B"
run cp -a /opt/vearl/bin "$B/bin"
for f in /etc/vearl/node.env /etc/vearl/known-hashes.json /etc/systemd/system/vearl-feeaudit.service /etc/systemd/system/vearl-feeaudit.timer; do
  [ -e "$f" ] && run cp -a "$f" "$B/" || true
done
[ -d /opt/vearl/tools ] && run cp -a /opt/vearl/tools "$B/tools" || true
[ -d /opt/vearl/feeaudit ] && run cp -a /opt/vearl/feeaudit "$B/feeaudit" || true
[ -d /opt/vearl-gateway-public ] && run cp -a /opt/vearl-gateway-public "$B/gateway-public" || true
[ -f /var/lib/vearl/node.key ] && echo "network key before: $(sha256sum /var/lib/vearl/node.key | cut -c1-16)"

echo "== 3. which services depend on the node and are running now"
RESTART=""
for u in vearl-faucet vearl-gateway; do
  if systemctl is-active --quiet "$u" 2>/dev/null; then RESTART="$RESTART $u"; fi
done
echo "will be started again after the node:${RESTART:- none}"

echo "== 4. stop the node (the services that require it stop with it)"
run systemctl stop vearl-node

echo "== 5. programs"
for f in bin/*; do run install -m 0755 -o root -g root "$f" "/opt/vearl/bin/$(basename "$f")"; done
if [ -d /opt/vearl-gateway-public ]; then
  run install -m 0755 bin/vearl-cli /opt/vearl-gateway-public/vearl-cli
  run install -m 0644 gateway/vearl-gateway.py /opt/vearl-gateway-public/vearl-gateway.py
fi

echo "== 6. settings: the seed collects fee beacons AND relays (keeps every other option of node.env)"
args=" $( (grep -h '^VEARL_EXTRA_ARGS=' /etc/vearl/node.env 2>/dev/null || true) | tail -1 | cut -d= -f2-) "
for f in --collect-fee-beacons --relay; do
  case "$args" in *" $f "*) ;; *) args="$args$f " ;; esac
done
args=$(echo $args)
others=$(grep -v '^VEARL_EXTRA_ARGS=' /etc/vearl/node.env 2>/dev/null || true)
echo "VEARL_EXTRA_ARGS=$args"
if [ "$DRY" != 1 ]; then { [ -n "$others" ] && echo "$others"; echo "VEARL_EXTRA_ARGS=$args"; } > /etc/vearl/node.env.new && cat /etc/vearl/node.env.new > /etc/vearl/node.env && rm -f /etc/vearl/node.env.new; fi
run install -m 0640 -o root -g vearl known-hashes.json /etc/vearl/known-hashes.json

echo "== 7. fee auditor and rewards tools"
run install -d -m 0755 /opt/vearl/tools /opt/vearl/feeaudit
for f in tools/*.py; do run install -m 0755 "$f" "/opt/vearl/tools/$(basename "$f")"; done
run install -m 0755 feeaudit/feeaudit.py /opt/vearl/feeaudit/feeaudit.py
run install -m 0644 feeaudit/vearl-feeaudit.service feeaudit/vearl-feeaudit.timer /etc/systemd/system/
run install -m 0755 rewards/vearl-rewards.sh /opt/vearl/bin/vearl-rewards.sh
run systemctl daemon-reload

echo "== 8. start"
run systemctl start vearl-node
[ "$DRY" = 1 ] || sleep 8
for u in $RESTART; do run systemctl start "$u"; done

echo "== 9. check"
if [ "$DRY" != 1 ]; then
  systemctl is-active vearl-node $RESTART vearl-feeaudit.timer || true
  journalctl -u vearl-node --no-pager -n 6 | cut -c1-200
  curl -s -m 5 http://127.0.0.1:8545/api/status | head -c 300; echo
  [ -f /var/lib/vearl/node.key ] && echo "network key after:  $(sha256sum /var/lib/vearl/node.key | cut -c1-16)  (must be the same as before)"
  grep -h VEARL_EXTRA_ARGS /etc/vearl/node.env
  ss -ltn | grep ":9000" || echo "WARNING: nothing listens on 9000"
fi
echo
echo "Done. To go back: systemctl stop vearl-node; cp -a $B/bin/. /opt/vearl/bin/; cp $B/node.env /etc/vearl/node.env; systemctl start vearl-node"
echo "Then publish the release (miners of this release talk to this seed). Watch the relay: journalctl -u vearl-node | grep '\[relay\]'"
