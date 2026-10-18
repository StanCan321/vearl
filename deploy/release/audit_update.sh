#!/bin/bash
# Replaces ONLY the pool-audit program of the seed (the hourly rewards job, vearl-rewards.sh, runs it): no service is restarted, the node is not touched.
#   tar xzf auditpkg-rcNN.tgz && cd auditpkg-rcNN && DRY=1 bash audit_update.sh     # shows what it would do, changes nothing
#   bash audit_update.sh                                                          # does it
# The old program is kept in /root/vearl-pool-audit.bak-DATE (copy it back to go back). Run as root on the seed.
set -euo pipefail
cd "$(dirname "$0")"
DRY=${DRY:-0}
run() { if [ "$DRY" = 1 ]; then echo "[dry] $*"; else "$@"; fi }
if [ "$DRY" != 1 ] && [ "$(id -u)" != 0 ]; then echo "run as root (or DRY=1 to look)"; exit 1; fi

echo "== 1. the package is intact"
sha256sum -c SHA256SUMS --quiet && echo "ok"
[ "$(head -c 4 vearl-pool-audit | od -An -c | tr -d ' ')" = '177ELF' ] && echo "the program is a Linux executable"

echo "== 2. the new program starts on this server"
./vearl-pool-audit --help > /dev/null 2>&1 || [ $? -le 2 ]
echo "ok (exit status acceptable)"
echo "new program: $(sha256sum vearl-pool-audit | cut -c1-16)    current: $(sha256sum /opt/vearl/bin/vearl-pool-audit | cut -c1-16)"

echo "== 3. backup"
B=/root/vearl-pool-audit.bak-$(date +%Y%m%d-%H%M%S)
run cp -a /opt/vearl/bin/vearl-pool-audit "$B"
echo "old program kept as $B"

echo "== 4. install (atomic: the hourly job never sees a half-written file)"
run install -m 0755 -o root -g root vearl-pool-audit /opt/vearl/bin/vearl-pool-audit.new
run mv /opt/vearl/bin/vearl-pool-audit.new /opt/vearl/bin/vearl-pool-audit

echo "== 5. what to look at next"
echo "  systemctl start vearl-rewards.service ; journalctl -u vearl-rewards.service --since '-2 minutes' --no-pager | tail"
echo "  to go back: cp -a $B /opt/vearl/bin/vearl-pool-audit"
