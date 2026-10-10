#!/usr/bin/env python3
"""Checks the registrations of the testnet rewards program (docs/19): proof that the person owns the address they registered.

    python3 tools/verify_registration.py registrations.json --out registered.json

registrations.json is a list of {"address": "vrl1...", "contact": "discord handle or e-mail", "message": "...", "signature": "HEX"}.
A contributor makes the signature with
    vearl-wallet sign --file wallet.json --message "Vearl testnet rewards registration. Address: vrl1.... Contact: HANDLE. I accept the terms in docs/19."
The tool checks the Bech32m address, that the message has the required form and names the same address, and the Ed25519 signature (made over
the domain-separated message, see vearl-core/src/wallet.rs). It writes {"addresses": [...], "rejected": [...], "shared_contacts": [...]}:
pass the file to testnet_rewards.py --registered. A contact that registers several addresses is listed in shared_contacts for a
manual look (the rules allow one participant per person). Needs `pip install cryptography`.
"""
import argparse
import json
import re
import sys

CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
BECH32M_CONST = 0x2BC830A3
DOMAIN = b"vearl/message/v1\0"


def _polymod(values):
    gen = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for v in values:
        top = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ v
        for i in range(5):
            chk ^= gen[i] if (top >> i) & 1 else 0
    return chk


def bech32m_decode(s):
    """(hrp, bytes) of a Bech32m string, or raises ValueError."""
    if s != s.lower() and s != s.upper():
        raise ValueError("mixed case")
    s = s.lower()
    pos = s.rfind("1")
    if pos < 1 or pos + 7 > len(s):
        raise ValueError("bad separator")
    hrp, data = s[:pos], s[pos + 1:]
    if any(c not in CHARSET for c in data):
        raise ValueError("bad character")
    vals = [CHARSET.index(c) for c in data]
    exp = [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]
    if _polymod(exp + vals) != BECH32M_CONST:
        raise ValueError("bad checksum")
    acc, bits, out = 0, 0, []
    for v in vals[:-6]:
        acc = (acc << 5) | v
        bits += 5
        while bits >= 8:
            bits -= 8
            out.append((acc >> bits) & 255)
    if bits >= 5 or (acc << (8 - bits)) & 255:
        raise ValueError("bad padding")
    return hrp, bytes(out)


def verify_signature(address, message, signature_hex):
    """None if the signature is valid, otherwise the reason."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    try:
        hrp, pk = bech32m_decode(address)
    except ValueError as e:
        return f"invalid address ({e})"
    if hrp != "vrl" or len(pk) != 32:
        return "not a classic vrl1 address"
    try:
        sig = bytes.fromhex(signature_hex)
    except ValueError:
        return "signature is not hexadecimal"
    if len(sig) != 64:
        return "signature has the wrong length"
    try:
        Ed25519PublicKey.from_public_bytes(pk).verify(sig, DOMAIN + message.encode())
    except (InvalidSignature, ValueError):
        return "signature does not match the address and the message"
    return None


MESSAGE_RE = re.compile(r"^Vearl testnet rewards registration\. Address: (vrl1[a-z0-9]+)\. Contact: (.+)\. I accept the terms in docs/19\.$")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("registrations")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = json.load(open(a.registrations, encoding="utf-8"))
    ok, rejected, by_contact = {}, [], {}
    for it in items:
        addr = (it.get("address") or "").strip()
        msg = it.get("message") or ""
        m = MESSAGE_RE.match(msg)
        reason = None
        if not m:
            reason = "message is not in the required form"
        elif m.group(1) != addr:
            reason = "the message names another address"
        else:
            reason = verify_signature(addr, msg, (it.get("signature") or "").strip())
        if reason:
            rejected.append({"address": addr, "reason": reason})
            continue
        if addr in ok:
            continue   # the same registration sent twice
        ok[addr] = m.group(2).strip()
        by_contact.setdefault(m.group(2).strip().lower(), []).append(addr)
    result = {"addresses": sorted(ok), "rejected": rejected,
              "shared_contacts": [{"contact": c, "addresses": sorted(v)} for c, v in sorted(by_contact.items()) if len(v) > 1]}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1)
    print(f"{len(ok)} valid, {len(rejected)} rejected, {len(result['shared_contacts'])} contacts with several addresses; written to {a.out}")


if __name__ == "__main__":
    sys.exit(main())
