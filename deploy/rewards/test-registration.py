#!/usr/bin/env python3
"""Checks, from any computer, that the registration mail account is locked down as intended (docs/19, deploy/rewards/).

    python3 deploy/rewards/test-registration.py            (reads the public password from vearl-core/src/regmail.rs)

Each check connects to mail.vearl.org:465 with the PUBLIC account registrar@vearl.org and must be REFUSED, except the last one, which sends
one real (empty-signature) mail to register@ and must be ACCEPTED by the mail server (the processor then drops it: the signature is not valid).
"""
import os
import re
import smtplib
import ssl
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "..", "..", "vearl-core", "src", "regmail.rs"), encoding="utf-8").read()
PW = re.search(r'SMTP_PASSWORD: &str = "([^"]+)"', SRC).group(1)
HOST, PORT, USER = "mail.vearl.org", 465, "registrar@vearl.org"
ok = True


def session(user=USER, pw=PW):
    s = smtplib.SMTP_SSL(HOST, PORT, context=ssl.create_default_context(), timeout=30)
    s.ehlo("test.local")
    s.login(user, pw)
    return s


def expect_refused(name, fn):
    global ok
    try:
        fn()
    except (smtplib.SMTPRecipientsRefused, smtplib.SMTPSenderRefused, smtplib.SMTPDataError, smtplib.SMTPAuthenticationError,
            smtplib.SMTPResponseException, smtplib.SMTPServerDisconnected) as e:
        print(f"ok      refused as it must be: {name}  ({str(e)[:90]})")
        return
    print(f"FAILED  accepted but it must be refused: {name}")
    ok = False


def to_other_recipient():
    s = session()
    s.sendmail(USER, ["contact@vearl.org"], b"From: registrar@vearl.org\r\nTo: contact@vearl.org\r\nSubject: x\r\n\r\nx\r\n")


def to_an_external_recipient():
    s = session()
    s.sendmail(USER, ["someone@example.org"], b"From: registrar@vearl.org\r\nTo: someone@example.org\r\nSubject: x\r\n\r\nx\r\n")


def with_another_sender():
    s = session()
    s.sendmail("contact@vearl.org", ["register@vearl.org"], b"From: contact@vearl.org\r\nTo: register@vearl.org\r\nSubject: x\r\n\r\nx\r\n")


def too_large():
    s = session()
    s.sendmail(USER, ["register@vearl.org"], b"From: registrar@vearl.org\r\nTo: register@vearl.org\r\nSubject: big\r\n\r\n" + b"x" * 40000)


def wrong_password():
    session(pw="not-the-password")


def other_account_on_465():
    # the contact@ mailbox must keep using 587: on 465 only the registrar account may send
    s = smtplib.SMTP_SSL(HOST, PORT, context=ssl.create_default_context(), timeout=30)
    s.ehlo("test.local")
    pw = os.environ.get("CONTACT_PASSWORD")
    if not pw:
        raise smtplib.SMTPAuthenticationError(535, b"skipped: set CONTACT_PASSWORD to test it")
    s.login("contact@vearl.org", pw)
    s.sendmail("contact@vearl.org", ["register@vearl.org"], b"From: contact@vearl.org\r\nTo: register@vearl.org\r\nSubject: x\r\n\r\nx\r\n")




def imap_storage():
    """The public account must not be usable as free storage: it can log in at most, but it cannot store a message."""
    import imaplib
    m = imaplib.IMAP4_SSL(HOST, 993, ssl_context=ssl.create_default_context())
    try:
        m.login(USER, PW)
    except imaplib.IMAP4.error as e:
        raise smtplib.SMTPAuthenticationError(535, str(e).encode())
    typ, _ = m.append("INBOX", None, None, b"Subject: storage test\r\n\r\nx\r\n")
    if typ == "OK":
        return
    raise smtplib.SMTPAuthenticationError(535, b"IMAP append refused")

expect_refused("a recipient other than register@ (contact@vearl.org)", to_other_recipient)
expect_refused("an outside recipient", to_an_external_recipient)
expect_refused("another sender address", with_another_sender)
expect_refused("a 40 KB message", too_large)
expect_refused("a wrong password", wrong_password)
expect_refused("storing a message in the IMAP mailbox of registrar@", imap_storage)
expect_refused("any other account on port 465 (needs CONTACT_PASSWORD to be meaningful)", other_account_on_465)

try:
    s = session()
    s.sendmail(USER, ["register@vearl.org"], b"From: registrar@vearl.org\r\nTo: register@vearl.org\r\nSubject: Vearl testnet registration\r\n\r\n"
                                              b"address: test\r\nmessage: test\r\nsignature: 00\r\n")
    print("ok      accepted as it must be: a small mail from registrar@ to register@ (the processor will drop it: invalid signature)")
except Exception as e:  # noqa: BLE001
    print(f"FAILED  the normal registration mail was refused: {e}")
    ok = False
print("\nALL GOOD" if ok else "\nSOMETHING IS WRONG: do not announce the registration before it is fixed")
sys.exit(0 if ok else 1)
