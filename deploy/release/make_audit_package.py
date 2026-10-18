#!/usr/bin/env python3
"""Builds the small package that replaces only the pool-audit program of the seed (used with deploy/release/audit_update.sh).

    python3 deploy/release/make_audit_package.py --audit DIR_LINUX_RELEASE/vearl-pool-audit --out auditpkg-rc20.tgz

The package holds the program (checked to be a Linux ELF executable), audit_update.sh and SHA256SUMS. Nothing secret goes into it."""
import argparse
import hashlib
import io
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prog = open(a.audit, "rb").read()
    if prog[:4] != b"\x7fELF":
        sys.exit(f"{a.audit} is not a Linux executable")
    script = open(os.path.join(HERE, "audit_update.sh"), "rb").read()
    sums = f"{hashlib.sha256(prog).hexdigest()}  vearl-pool-audit\n{hashlib.sha256(script).hexdigest()}  audit_update.sh\n".encode()
    top = os.path.splitext(os.path.basename(a.out))[0]
    if top.endswith(".tar"):
        top = top[:-4]
    with tarfile.open(a.out, "w:gz") as t:
        for name, data, mode in (("vearl-pool-audit", prog, 0o755), ("audit_update.sh", script, 0o755), ("SHA256SUMS", sums, 0o644)):
            ti = tarfile.TarInfo(f"{top}/{name}")
            ti.size, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname, ti.mtime = len(data), mode, 0, 0, "root", "root", 1_700_000_000
            t.addfile(ti, io.BytesIO(data))
    print(f"{a.out}: {hashlib.sha256(open(a.out, 'rb').read()).hexdigest()}")


if __name__ == "__main__":
    main()
