#!/bin/bash
# Runs ON THE MAIL SERVER, as root (called by deploy/rewards/install-registration.sh). Idempotent, keeps backups in /root.
#   bash setup-registration-mail.sh 'PUBLIC_PASSWORD'
# PUBLIC_PASSWORD is the value of SMTP_PASSWORD in vearl-core/src/regmail.rs: the password of the PUBLIC send-only account used by
# `vearl-wallet register` (public on purpose, see docs/19 and the header of regmail.rs).
#
# What it sets up (docs/19, section 5):
#   register@vearl.org   a mailbox that receives the registrations (read by the processor, nobody logs in: random password)
#   registrar@vearl.org  a send-only account. Port 465 (smtps) is now RESERVED for it:
#                          * it can send to register@vearl.org and to nobody else;
#                          * it can only use the address registrar@vearl.org as sender;
#                          * 16 KB per message, 10 messages per minute per client IP, 20 connections per minute;
#                          * it cannot receive mail (no mailbox on Postfix) and its IMAP mailbox cannot be created (mail location set to /nonexistent).
#                        Port 587 (submission) is NOT changed: contact@vearl.org keeps sending from the mail apps there.
set -euo pipefail
PW="${1:?usage: setup-registration-mail.sh PUBLIC_PASSWORD}"
TS=$(date +%s)
for f in /etc/postfix/main.cf /etc/postfix/master.cf /etc/postfix/vmailbox /etc/dovecot/users; do
  cp -a "$f" "/root/$(basename "$f").before-registration-$TS"
done

# 1. Dovecot users (passwd-file): register@ (random password, nobody logs in) and registrar@ (public password, SASL only)
add_user() { # name hash home extra
  if grep -q "^$1:" /etc/dovecot/users; then
    echo "dovecot user $1 exists: kept"
  else
    printf '%s:%s:5000:5000::%s::%s\n' "$1" "$2" "$3" "$4" >> /etc/dovecot/users
    echo "dovecot user $1 added"
  fi
}
RANDOM_PW=$(head -c 32 /dev/urandom | base64 | tr -d '=+/' | head -c 40)
add_user register@vearl.org "$(doveadm pw -s SHA512-CRYPT -p "$RANDOM_PW")" /var/vmail/vearl.org/register ""
add_user registrar@vearl.org "$(doveadm pw -s SHA512-CRYPT -p "$PW")" /nonexistent "userdb_mail=maildir:/nonexistent/Maildir"

# 2. the mailbox of register@ (Postfix accepts it, Dovecot stores it)
grep -q '^register@vearl.org' /etc/postfix/vmailbox || echo 'register@vearl.org OK' >> /etc/postfix/vmailbox
install -d -o vmail -g vmail -m 0700 /var/vmail/vearl.org/register /var/vmail/vearl.org/register/Maildir
for d in new cur tmp; do install -d -o vmail -g vmail -m 0700 "/var/vmail/vearl.org/register/Maildir/$d"; done

# 3. Postfix: who may send what on port 465
printf 'registrar@vearl.org only_register\n' > /etc/postfix/sasl_access
printf 'register@vearl.org OK\n' > /etc/postfix/register_only
printf 'registrar@vearl.org registrar@vearl.org\n' > /etc/postfix/sender_login
for m in vmailbox sasl_access register_only sender_login; do postmap "hash:/etc/postfix/$m"; done
postconf -e "smtpd_restriction_classes = only_register" "only_register = check_recipient_access hash:/etc/postfix/register_only, reject"

python3 - <<'PY'
import re
p = "/etc/postfix/master.cf"
s = open(p).read()
# remove the current smtps service (its line and the indented option lines that follow)
s = re.sub(r"(?m)^smtps[ \t]+inet.*\n(?:[ \t]+.*\n)*", "", s)
s = s.rstrip("\n") + "\n" + """smtps     inet n       -       y       -       -       smtpd
  -o syslog_name=postfix/smtps
  -o smtpd_tls_wrappermode=yes
  -o smtpd_sasl_auth_enable=yes
  -o smtpd_sasl_type=dovecot
  -o smtpd_sasl_path=private/auth
  -o smtpd_sasl_security_options=noanonymous
  -o smtpd_sender_login_maps=hash:/etc/postfix/sender_login
  -o smtpd_client_restrictions=permit_sasl_authenticated,reject
  -o smtpd_sender_restrictions=reject_sender_login_mismatch
  -o smtpd_relay_restrictions=permit_sasl_authenticated,reject
  -o smtpd_recipient_restrictions=check_sasl_access hash:/etc/postfix/sasl_access,reject
  -o message_size_limit=16384
  -o smtpd_client_message_rate_limit=10
  -o smtpd_client_recipient_rate_limit=10
  -o smtpd_client_connection_rate_limit=20
"""
open(p, "w").write(s)
PY

# 4. apply
postfix check
systemctl reload dovecot
systemctl reload postfix
echo
echo "--- ports:"; ss -ltn | awk 'NR==1||/:(465|587|993|25)[ \t]/'
echo "--- the smtps service now:"; grep -A18 '^smtps' /etc/postfix/master.cf
echo "OK: registration mail is set up. Next: test with the throw-away wallet (see deploy/rewards/README or docs/19)."
