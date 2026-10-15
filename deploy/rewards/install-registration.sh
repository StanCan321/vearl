#!/bin/bash
# Installs the registration by e-mail for the testnet rewards (docs/19, section 5) on the web/mail server. Run it from your PC:
#   bash deploy/rewards/install-registration.sh
# It needs the leaderboard install first (bash deploy/rewards/install-rewards.sh) and the mail server of deploy/mail/.
#  1. mail: register@ (receives) and registrar@ (public send-only account: can write to register@ only), port 465 reserved for it
#     (deploy/rewards/setup-registration-mail.sh, with backups of every file it changes);
#  2. the processor (tools/process_registration_mail.py) as a systemd timer every two minutes: it reads register@, checks the signatures and
#     publishes https://vearl.org/rewards/registered.json (addresses only);
#  3. the leaderboard then takes the list of registered addresses (vearl-rewards.sh).
# Test afterwards:  python3 deploy/rewards/test-registration.py
set -euo pipefail
cd "$(dirname "$0")/../.."
KEY="-i ${VEARL_SSH_KEY:-$HOME/.ssh/id_ed25519} -o IdentitiesOnly=yes"   # override with VEARL_SSH_KEY
HOST=${VEARL_SSH_HOST:-root@vearl.org}                                        # the web/mail server (override with VEARL_SSH_HOST)
PW=$(grep -o 'SMTP_PASSWORD: &str = "[^"]*"' vearl-core/src/regmail.rs | sed 's/.*= "//; s/"$//')
[ -n "$PW" ] || { echo "cannot read SMTP_PASSWORD from vearl-core/src/regmail.rs"; exit 1; }
scp -q $KEY vearl-core/tools/process_registration_mail.py vearl-core/tools/verify_registration.py deploy/rewards/setup-registration-mail.sh \
  deploy/rewards/vearl-register.service deploy/rewards/vearl-register.timer deploy/rewards/vearl-rewards.sh $HOST:/root/
ssh $KEY $HOST "set -e
bash /root/setup-registration-mail.sh '$PW'
install -d -m 0755 /opt/vearl/tools
install -m 0755 /root/process_registration_mail.py /opt/vearl/tools/process_registration_mail.py
install -m 0755 /root/verify_registration.py /opt/vearl/tools/verify_registration.py
install -m 0755 /root/vearl-rewards.sh /opt/vearl/bin/vearl-rewards.sh
install -d -o vearl -g vearl -m 2775 /var/www/vearl-rewards
chmod 2775 /var/www/vearl-rewards
install -m 0644 /root/vearl-register.service /etc/systemd/system/vearl-register.service
install -m 0644 /root/vearl-register.timer /etc/systemd/system/vearl-register.timer
rm -f /root/process_registration_mail.py /root/verify_registration.py /root/vearl-register.service /root/vearl-register.timer /root/vearl-rewards.sh /root/setup-registration-mail.sh
systemctl daemon-reload
systemctl enable --now vearl-register.timer
systemctl start vearl-register.service
sleep 2
systemctl is-active vearl-register.timer
journalctl -u vearl-register.service -n 3 --no-pager | cut -c1-200
ls -la /var/www/vearl-rewards"
echo
curl -s https://vearl.org/rewards/registered.json || true
