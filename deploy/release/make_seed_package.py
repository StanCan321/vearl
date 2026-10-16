#!/usr/bin/env python3
"""Builds the upgrade package of the seed server for a release (used with deploy/release/seed_update.sh).

    python3 deploy/release/make_seed_package.py --linux DIR_LINUX_RELEASE --known known-hashes.json --out seedpkg-rc14.tgz

DIR_LINUX_RELEASE holds the Linux programs of the release (node, vearl-cli, vearl-wallet, vearl-faucet, vearl-genesis, vearl-pool-audit). The package also
carries the fee auditor, the rewards tools and the gateway of this source tree. Nothing secret goes into it: files named *.password, *.token, token.txt
or *.env are refused, and so is a program that is not an ELF executable.
"""
import argparse
import hashlib
import io
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
NEXUS = os.path.abspath(os.path.join(HERE, "..", ".."))
BINS = ["node", "vearl-cli", "vearl-wallet", "vearl-faucet", "vearl-genesis", "vearl-pool-audit"]
TOOLS = ["testnet_rewards.py", "pool_points.py", "audit_pools.py", "fetch_pool_bundle.py", "verify_registration.py"]
REFUSED = (".password", ".token", ".env", "token.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--linux", required=True)
    ap.add_argument("--known", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = {}
    for b in BINS:
        p = os.path.join(a.linux, b)
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
        if open(p, "rb").read(4) != b"\x7fELF":
            sys.exit(f"{p} is not a Linux executable")
        files[f"bin/{b}"] = p
    for t in TOOLS:
        files[f"tools/{t}"] = os.path.join(NEXUS, "vearl-core", "tools", t)
    for f in ("feeaudit.py", "vearl-feeaudit.service", "vearl-feeaudit.timer"):
        files[f"feeaudit/{f}"] = os.path.join(NEXUS, "deploy", "feeaudit", f)
    files["rewards/vearl-rewards.sh"] = os.path.join(NEXUS, "deploy", "rewards", "vearl-rewards.sh")
    files["gateway/vearl-gateway.py"] = os.path.join(NEXUS, "deploy", "gateway", "vearl-gateway.py")
    files["known-hashes.json"] = a.known
    files["seed_update.sh"] = os.path.join(HERE, "seed_update.sh")
    for name, p in files.items():
        if not os.path.isfile(p):
            sys.exit(f"missing {p}")
        if name.lower().endswith(REFUSED) or os.path.basename(p).lower() in REFUSED:
            sys.exit(f"refused: {name} looks like a secret")
    sums = io.StringIO()
    data = {}
    for name, p in sorted(files.items()):
        raw = open(p, "rb").read()
        if name.endswith((".sh", ".py", ".service", ".timer")):
            raw = raw.replace(b"\r\n", b"\n")      # these run on Linux
        data[name] = raw
        if name != "seed_update.sh":
            sums.write(f"{hashlib.sha256(raw).hexdigest()}  {name}\n")
    root = os.path.splitext(os.path.splitext(os.path.basename(a.out))[0])[0]
    with tarfile.open(a.out, "w:gz") as t:
        def put(name, raw, mode):
            ti = tarfile.TarInfo(f"{root}/{name}")
            ti.size, ti.mode, ti.mtime = len(raw), mode, int(os.path.getmtime(files.get(name, a.out)))
            t.addfile(ti, io.BytesIO(raw))
        for name, raw in data.items():
            put(name, raw, 0o755 if name.startswith("bin/") or name.endswith((".sh", ".py")) else 0o644)
        put("SHA256SUMS", sums.getvalue().encode(), 0o644)
    print(f"{a.out}: {len(data)} files, {os.path.getsize(a.out) // 1024} KiB")


if __name__ == "__main__":
    main()
