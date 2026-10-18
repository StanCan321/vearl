#!/bin/bash
# Installs the automatic launch on the seed server (docs/06 section 3 sexies): nothing public changes until the launch time.
#   bash deploy/launch/install-launch.sh
# It uploads vearl-launch.sh, the two live nginx snippets and the faucet page (with the same content fingerprints as deploy-site.sh), installs the
# systemd timer (every minute, plus one tick at the launch time), and runs the read-only check on the server.
set -euo pipefail
cd "$(dirname "$0")/../.."
KEY="-i ${VEARL_SSH_KEY:-$HOME/.ssh/id_ed25519_vps_new} -o IdentitiesOnly=yes"
HOST=${VEARL_SSH_HOST:-root@103.89.12.219}
OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT
mkdir -p "$OUT/faucet"
cp site/faucet/index.html "$OUT/faucet/index.html"
for f in assets/site.css assets/explorer.css assets/compute.css assets/faucet.js assets/nav.js assets/site.js fonts/fonts.css; do
  h=$(md5sum "site/$f" | cut -c1-10)
  sed -i "s#/$f\"#/$f?v=$h\"#g" "$OUT/faucet/index.html"
done
# the fingerprints must be those of the files that are on the server (the page is published at the launch, the assets are already there)
for f in $(grep -o '/\(assets\|fonts\)/[a-z.]*?v=[0-9a-f]*' "$OUT/faucet/index.html"); do
  path=${f%%\?*}; want=${f##*v=}
  have=$(ssh $KEY $HOST "md5sum /var/www/vearl$path | cut -c1-10")
  [ "$have" = "$want" ] || { echo "the page expects $path v=$want but the server holds v=$have: deploy the site first (deploy-site.sh)" >&2; exit 1; }
done
cp deploy/nginx-vearl-api-live.conf deploy/nginx-vearl-gw-live.conf deploy/launch/vearl-launch.sh deploy/launch/vearl-launch.service deploy/launch/vearl-launch.timer "$OUT"/
tar czf - -C "$OUT" . | ssh $KEY $HOST 'set -e
  rm -rf /root/launch-new && mkdir -p /root/launch-new && tar xzf - -C /root/launch-new
  install -d -m 755 /opt/vearl/launch/faucet /var/lib/vearl-launch
  install -m 755 /root/launch-new/vearl-launch.sh /opt/vearl/launch/vearl-launch.sh
  install -m 644 /root/launch-new/nginx-vearl-api-live.conf /root/launch-new/nginx-vearl-gw-live.conf /opt/vearl/launch/
  install -m 644 /root/launch-new/faucet/index.html /opt/vearl/launch/faucet/index.html
  install -m 644 /root/launch-new/vearl-launch.service /root/launch-new/vearl-launch.timer /etc/systemd/system/
  rm -rf /root/launch-new
  systemctl daemon-reload
  systemctl enable --now vearl-launch.timer >/dev/null 2>&1
  systemctl list-timers vearl-launch.timer --no-pager | head -3
  /opt/vearl/launch/vearl-launch.sh --check || true'
