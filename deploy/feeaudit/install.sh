#!/usr/bin/env bash
# Installs the fee auditor on the seed node (run as root from the repository root): see docs/17-fee-protection.md
#   ./deploy/feeaudit/install.sh [--known known-hashes.json]
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }
D="$(cd "$(dirname "$0")" && pwd)"
KNOWN=""
[ "${1:-}" = "--known" ] && KNOWN="$2"
install -d -o vearl -g vearl -m 0750 /var/lib/vearl/feeaudit
install -d -m 0755 /opt/vearl/feeaudit
install -m 0755 "$D/feeaudit.py" /opt/vearl/feeaudit/feeaudit.py
install -m 0644 "$D/vearl-feeaudit.service" "$D/vearl-feeaudit.timer" /etc/systemd/system/
if [ -n "$KNOWN" ]; then install -m 0640 -o root -g vearl "$KNOWN" /etc/vearl/known-hashes.json; fi
[ -f /etc/vearl/known-hashes.json ] || echo '{"node": [], "quantus": []}' | install -m 0640 -o root -g vearl /dev/stdin /etc/vearl/known-hashes.json
# the seed collects the beacons
touch /etc/vearl/node.env
grep -q -- '--collect-fee-beacons' /etc/vearl/node.env 2>/dev/null || echo 'VEARL_EXTRA_ARGS=--collect-fee-beacons' >> /etc/vearl/node.env
systemctl daemon-reload
systemctl enable --now vearl-feeaudit.timer
echo "installed. Restart the node to start collecting:  systemctl restart vearl-node"
echo "run once now:  systemctl start vearl-feeaudit && cat /var/lib/vearl/feeaudit/compliance.json"
