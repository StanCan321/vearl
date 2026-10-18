#!/usr/bin/env python3
"""Builds the HiveOS custom-miner package of Vearl (README-HIVEOS.txt explains how it is used).

    python deploy/hiveos/make_package.py --release-dir DIR --version 0.1.0-rc16.2 [--out dist]

DIR is the folder of the LINUX release (the extracted vearl-<version>-linux-x86_64.tar.gz): the package takes from it vearl-miner, node,
vearl-poolminer, vearl-wallet, vearl-cli, network.json and genesis.json (the same programs as the release, byte for byte) and adds the Hive scripts of this
folder. Result: OUT/vearl-hiveos-<version>.tar.gz containing one folder `vearl/` (HiveOS unpacks it into /hive/miners/custom/), and
OUT/vearl-hiveos-<version>.tar.gz.sha256.
"""
import argparse
import hashlib
import io
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["h-manifest.conf", "h-config.sh", "h-run.sh", "h-stats.sh", "common.sh", "setup.sh", "README-HIVEOS.txt"]
PROGRAMS = ["vearl-miner", "vearl-poolminer", "node", "vearl-wallet", "vearl-cli"]
DATA = ["network.json", "genesis.json"]
EXEC = {"h-config.sh", "h-run.sh", "h-stats.sh", "setup.sh", *PROGRAMS}


def add(tar, name, data, mode):
    ti = tarfile.TarInfo(f"vearl/{name}")
    ti.size = len(data)
    ti.mode = mode
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = "root"
    ti.mtime = 1_700_000_000   # fixed: the same inputs give the same archive
    tar.addfile(ti, io.BytesIO(data))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-dir", required=True)
    ap.add_argument("--version", required=True, help="for example 0.1.0-rc16.2")
    ap.add_argument("--out", default="dist")
    a = ap.parse_args()
    for f in PROGRAMS + DATA:
        if not os.path.isfile(os.path.join(a.release_dir, f)):
            sys.exit(f"{f} is missing from {a.release_dir}")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"vearl-hiveos-{a.version}.tar.gz")
    import gzip
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for f in SCRIPTS:
            data = open(os.path.join(HERE, f), "rb").read().replace(b"\r\n", b"\n")   # shell scripts need LF endings
            if f == "h-manifest.conf":
                data = data.replace(b"@VERSION@", a.version.encode())
            add(tar, f, data, 0o755 if f in EXEC else 0o644)
        for f in PROGRAMS + DATA:
            add(tar, f, open(os.path.join(a.release_dir, f), "rb").read(), 0o755 if f in EXEC else 0o644)
    with open(path, "wb") as out:
        with gzip.GzipFile(fileobj=out, mode="wb", mtime=0, compresslevel=9) as gz:
            gz.write(raw.getvalue())
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with open(path + ".sha256", "w", newline="\n") as f:
        f.write(f"{digest}  {os.path.basename(path)}\n")
    print(f"{path}  {os.path.getsize(path) // 1024} KiB  sha256 {digest}")


if __name__ == "__main__":
    main()
