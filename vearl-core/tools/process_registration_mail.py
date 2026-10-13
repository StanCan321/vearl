#!/usr/bin/env python3
"""Reads the mailbox of register@vearl.org, checks every registration for the testnet rewards (docs/19) and publishes the list.

    python3 process_registration_mail.py --maildir /var/vmail/vearl.org/register/Maildir --state /var/lib/vearl-register --public /var/www/vearl-rewards

`vearl-wallet register` e-mails three labelled lines from registrar@vearl.org (the body of the mail):
    address: vrl1...
    message: Vearl testnet rewards registration. Address: vrl1.... Contact: HANDLE. I accept the terms in docs/19.
    signature: HEX
For each new mail the script checks the Bech32m address, the form of the message (it must name the same address), and the Ed25519 signature
(same code as tools/verify_registration.py). A registration is accepted only if the signature is valid, so nobody can register an address that
they do not own, whatever the sender of the mail says. Mail that is too large, malformed or invalid is counted and dropped.

Files written (all atomically, so a reader never sees a half-written file):
  STATE/registrations.json          every accepted registration WITH the contact (private: never published)
  STATE/rewards-registered.json     {"addresses": [...], "rejected": [...], "shared_contacts": [...]} for testnet_rewards.py --registered
  PUBLIC/registered.json            {"updated": ..., "count": N, "addresses": [...]}  addresses only, no contact details (public)
Processed mails are moved to the Maildir folder .Processed, so the inbox is empty and a mail is never read twice.
The same address registered twice keeps the first registration. A contact that registers several addresses is listed in shared_contacts for a
manual look (the rules allow one participant per person). Needs `cryptography` (apt install python3-cryptography).
"""
import argparse
import datetime
import email
import email.policy
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_registration import MESSAGE_RE, verify_signature  # noqa: E402

MAX_BYTES = 8192            # a real registration is about 600 bytes
MAX_PER_RUN = 500           # never loop forever on a flood
LABEL = re.compile(r"^(address|message|signature):[ \t]*(.*?)[ \t]*$", re.I)


def atomic_write(path, data):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o644)
    except OSError:
        pass


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def body_text(raw):
    """The plain text of the mail (first text/plain part), or None."""
    try:
        msg = email.message_from_bytes(raw, policy=email.policy.default)
    except Exception:
        return None
    part = msg.get_body(preferencelist=("plain",))
    if part is None:
        return None
    try:
        return part.get_content()
    except Exception:
        return None


def parse_registration(raw):
    """(address, message, signature) from the raw mail, or (None, reason)."""
    if len(raw) > MAX_BYTES:
        return None, "mail too large"
    text = body_text(raw)
    if text is None:
        return None, "no text body"
    fields = {}
    for line in text.splitlines():
        m = LABEL.match(line.strip())
        if m and m.group(1).lower() not in fields:
            fields[m.group(1).lower()] = m.group(2)
    if not all(k in fields for k in ("address", "message", "signature")):
        return None, "the three lines address / message / signature are not all there"
    return (fields["address"], fields["message"], fields["signature"]), None


def check(address, message, signature):
    """(contact, None) for a valid registration, (None, reason) otherwise."""
    m = MESSAGE_RE.match(message)
    if not m:
        return None, "message is not in the required form"
    if m.group(1) != address:
        return None, "the message names another address"
    reason = verify_signature(address, message, signature)
    if reason:
        return None, reason
    return m.group(2).strip(), None


def maildir_files(maildir):
    for sub in ("new", "cur"):
        d = os.path.join(maildir, sub)
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        for n in names:
            if not n.startswith("."):
                yield os.path.join(d, n)


def archive(maildir, path):
    """Moves a processed mail to the folder .Processed (Maildir++), so it is never read again."""
    dest = os.path.join(maildir, ".Processed", "cur")
    for sub in ("cur", "new", "tmp"):
        os.makedirs(os.path.join(maildir, ".Processed", sub), exist_ok=True)
    flags = ":2,S" if os.name != "nt" else ".S"   # the Maildir flag separator is a colon, which Windows file names cannot contain (tests only)
    name = os.path.basename(path).split(":")[0] + flags
    os.replace(path, os.path.join(dest, name))


def process(maildir, state, public, now=None, archive_mail=True):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    os.makedirs(state, exist_ok=True)
    regs_path = os.path.join(state, "registrations.json")
    rej_path = os.path.join(state, "rejected.json")
    regs = load_json(regs_path, [])
    rejected = load_json(rej_path, [])
    known = {r["address"] for r in regs}
    added = dropped = dup = 0
    for i, path in enumerate(maildir_files(maildir)):
        if i >= MAX_PER_RUN:
            break
        try:
            raw = open(path, "rb").read(MAX_BYTES + 1)
        except OSError:
            continue
        parsed, why = parse_registration(raw)
        if parsed is None:
            rejected.append({"time": now.isoformat(timespec="seconds"), "address": "", "reason": why})
            dropped += 1
        else:
            address, message, signature = parsed
            contact, why = check(address, message, signature)
            if why:
                rejected.append({"time": now.isoformat(timespec="seconds"), "address": address[:80], "reason": why})
                dropped += 1
            elif address in known:
                dup += 1
            else:
                regs.append({"address": address, "contact": contact, "message": message, "signature": signature,
                             "registered_at": now.isoformat(timespec="seconds")})
                known.add(address)
                added += 1
        if archive_mail:
            try:
                archive(maildir, path)
            except OSError:
                pass
    rejected = rejected[-500:]
    by_contact = {}
    for r in regs:
        by_contact.setdefault(r["contact"].lower(), []).append(r["address"])
    scoring = {"addresses": sorted(r["address"] for r in regs),
               "rejected": [{"address": x["address"], "reason": x["reason"]} for x in rejected if x["address"]],
               "shared_contacts": [{"contact": c, "addresses": sorted(v)} for c, v in sorted(by_contact.items()) if len(v) > 1]}
    atomic_write(regs_path, regs)
    atomic_write(rej_path, rejected)
    atomic_write(os.path.join(state, "rewards-registered.json"), scoring)
    if public:
        os.makedirs(public, exist_ok=True)
        atomic_write(os.path.join(public, "registered.json"),
                     {"updated": now.isoformat(timespec="seconds"), "count": len(regs), "addresses": scoring["addresses"]})
    return {"added": added, "dropped": dropped, "duplicates": dup, "total": len(regs), "shared_contacts": len(scoring["shared_contacts"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maildir", required=True, help="Maildir of register@vearl.org")
    ap.add_argument("--state", required=True, help="private folder for the registrations")
    ap.add_argument("--public", help="folder served at https://vearl.org/rewards/ (registered.json is written there)")
    ap.add_argument("--keep", action="store_true", help="do not move the processed mails (for tests)")
    a = ap.parse_args()
    r = process(a.maildir, a.state, a.public, archive_mail=not a.keep)
    print(f"{r['added']} new, {r['duplicates']} already known, {r['dropped']} dropped; {r['total']} registered in total, "
          f"{r['shared_contacts']} contacts with several addresses ({time.strftime('%Y-%m-%d %H:%M:%S')})")


if __name__ == "__main__":
    sys.exit(main())
