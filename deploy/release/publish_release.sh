#!/bin/bash
# Uploads the archives of a release (built by make_release.py) to the web server, where /download/files/ serves them (nginx location, see
# deploy/nginx-vearl-locations.conf). They live outside /var/www/vearl, so republishing the site never deletes them.
#   bash deploy/release/publish_release.sh DIST_DIR
# Only run it once the release is final and you want it downloadable: the files are public as soon as they are uploaded and nginx
# has the location (deploy-site.sh --nginx). The page that links to them is published with deploy-site.sh --with-download.
set -euo pipefail
DIST=${1:?usage: publish_release.sh DIST_DIR}
KEY="-i ${VEARL_SSH_KEY:-$HOME/.ssh/id_ed25519_vps4} -o IdentitiesOnly=yes"   # override with VEARL_SSH_KEY
HOST=${VEARL_SSH_HOST:-root@vearl.org}   # the web/seed server (override with VEARL_SSH_HOST)
[ -s "$DIST/SHA256SUMS.txt" ] || { echo "no SHA256SUMS.txt in $DIST"; exit 1; }
( cd "$DIST" && sha256sum -c SHA256SUMS.txt )
ssh $KEY $HOST 'mkdir -p /var/www/vearl-downloads'
for f in "$DIST"/*; do
  n=$(basename "$f")
  for try in 1 2 3; do
    ssh $KEY $HOST "cat > /var/www/vearl-downloads/$n.part" < "$f" && ssh $KEY $HOST "mv /var/www/vearl-downloads/$n.part /var/www/vearl-downloads/$n" && break
    sleep 5
  done
done
ssh $KEY $HOST 'chown -R root:root /var/www/vearl-downloads && chmod 755 /var/www/vearl-downloads && chmod 644 /var/www/vearl-downloads/* && cd /var/www/vearl-downloads && sha256sum -c SHA256SUMS.txt'
