#!/usr/bin/env python3
"""Assembles the downloadable miner packages of a testnet release and their checksums.

    python deploy/release/make_release.py --version testnet-1 --linux DIR --windows DIR --genesis genesis.json \
        --peer HOST:9000@PUBLIC_KEY [--network vearl-testnet-1] [--out dist]

DIR holds the files of one platform (the same names on both): vearl-miner (THE program users open), node, vearl-wallet, vearl-cli,
vearl-genesis, vearl-faucet, optionally vearl-pool / vearl-poolminer (mining pools, docs/16). The Linux files are glibc builds made on
Ubuntu 22.04 with the CUDA runtime linked in, the Windows ones the multi-architecture CUDA build: both need only the NVIDIA driver.

The MAIN package mines Vearl only and contains no third-party mining program. If DIR also holds quantus-miner plus the NVIDIA NVRTC
libraries it loads (Windows: nvrtc64_120_0.dll and nvrtc-builtins64_*.dll; Linux: libnvrtc.so.12 and libnvrtc-builtins.so.*), a separate
OPTIONAL ADD-ON archive is built from them (docs/15 section 4 bis): extracted into the folder of the main package, it lets the miner
also mine Quantus when the user asks for it. Nothing of the add-on is started unless the user enables it.

Every user must run the SAME release: the genesis hash only fixes the network, not the version of the code, so the version is
printed in README.txt and in SHA256SUMS.txt. network.json (name, genesis file, seed nodes) is what vearl-miner reads.

Output (out/): vearl-<version>-linux-x86_64.tar.gz, vearl-<version>-windows-x86_64.zip, the same two with "quantus-addon-" in the name
(when quantus-miner is present), SHA256SUMS.txt, and genesis.json when given.
"""
import argparse
import glob
import hashlib
import io
import json
import os
import sys
import tarfile
import time
import zipfile

BINS = ["vearl-miner", "node", "vearl-cli", "vearl-wallet", "vearl-genesis", "vearl-faucet"]
# pools (docs/16): included when present in the platform directory
OPTIONAL_BINS = ["vearl-poolminer", "vearl-pool"]
# optional Quantus add-on (docs/15 section 4 bis): a separate archive
QUANTUS_BIN = "quantus-miner"
LIB_GLOBS = {True: ["nvrtc64_*.dll", "nvrtc-builtins64_*.dll"], False: ["libnvrtc.so*", "libnvrtc-builtins.so*"]}

README = """VEARL {version}
================

Vearl is a TESTNET. Its tokens have no value and are not for sale, the code has not been audited by anybody, and the network can be
reset: read https://vearl.org before running anything.

START HERE
----------
{start_here}

What the miner does
-------------------
 * It mines Vearl blocks with your GPU and computes the paid jobs that clients submit to the network (AI inference, rendering, linear
   algebra). It mines Vearl only: no other program runs on your machine.
 * You need an NVIDIA GPU of the Ampere generation or newer (RTX 30xx, 40xx, 50xx, A-series, L-series, H100...) and the NVIDIA driver
   ({driver}). Without a GPU the miner runs on the CPU, slowly.
 * The first start downloads the chain from the seed node and can take a few minutes. A status line is printed every 10 seconds
   and a copy of the log (with date and time) is kept in miner.log.
 * Optional: the separate "quantus-addon" download lets the same miner also mine Quantus (QTC) with your GPU. It is off unless you
   install it AND answer yes in the assistant. Its README (QUANTUS-ADDON.txt) explains what it does, including the development fee.

Your wallet
-----------
The assistant creates a wallet.json protected by a password and shows you 24 recovery words ONCE: write them on paper. Whoever has
the words has the funds, and without them a lost password or a lost file cannot be recovered. Do not put them in a file, a photo
or a message, and never give them to anybody (nobody from the project will ever ask for them).

Command-line tools
------------------
vearl-wallet{exe}, vearl-cli{exe} and the other programs of this folder, except vearl-miner{exe}, are command-line tools: they have no
window, so a double-click only shows their help and waits for Enter. Open a terminal in this folder (Windows: click the address bar of
the folder, type cmd and press Enter; Linux: any terminal) and type a command. Run a tool without arguments to see its help, for example:
   vearl-wallet{exe}
   vearl-wallet{exe} address --file wallet.json     (shows your address; the password is asked)
   vearl-wallet{exe} register --file wallet.json --contact "YOUR_DISCORD_NAME_OR_EMAIL"   (testnet rewards: signs your registration and, after you confirm, e-mails it to register@vearl.org)

Files in this folder
--------------------
 vearl-miner{exe}        the program to open (everything else is started by it)
 node{exe}               the Vearl node
 vearl-wallet{exe}, vearl-cli{exe}   wallet and command line tools
 vearl-pool{exe}, vearl-poolminer{exe}   for pool operators / mining in a pool (docs/16 in the source repository)
 network.json, genesis.json   the network (do not edit: every miner must use the same files)
 miner.json, wallet.json, data/, miner.log   created on first start (your settings, wallet, chain, log)

Advanced: {advanced}

Release {version}: SHA-256 of every file is in SHA256SUMS.txt. Build date: {date}.
"""

README_ADDON = """VEARL {version}: QUANTUS ADD-ON (optional)
=============================================

This archive is NOT needed to mine Vearl. It adds the official open-source Quantus miner to an existing Vearl miner folder, so that the
same GPU can also mine Quantus (QTC).

Install: extract this archive INTO the folder of the Vearl miner (the one that contains vearl-miner{exe}); the files are merged. Then run
"vearl-miner setup" and answer yes to "Also mine Quantus with this miner?". Until you do, the add-on is never started.

What it does
------------
 * While Vearl has no paid job for you, the GPU mines Quantus with the official Quantus miner included here. You mine to YOUR OWN
   Quantus address and the Quantus pool pays you directly: Vearl never holds your funds. Default split: 80 % of the time Quantus,
   20 % Vearl (miner.json, partner_share). When a paid Vearl job is assigned to you, the whole GPU goes to Vearl at once, even in the
   middle of a Quantus window, and Quantus resumes when the job is done.
 * Development fee: 10 % of your Quantus mining time is spent mining to the Vearl development address
   {dev_address} (shown at every start). Quantus rewards are small and change every day: do not count on them.
 * Fee check: every 5 minutes the miner sends the Vearl seed nodes a signed message (your Vearl account, your Quantus address, program
   versions and hashes; no key, no secret) so the project can verify, with the public API of the Quantus pool, that the fee worker runs.
 * The fee address and percentage cannot be changed in this program and the Quantus miner shipped here is the only one it will start.
   Details and limits: docs/17-fee-protection.md of the source repository.
 * To stop: run "vearl-miner setup" and answer no, or delete the add-on files (quantus-miner{exe} and the nvrtc libraries).

Files: quantus-miner{exe}, nvrtc libraries, licenses/ (Quantus miner: Apache-2.0; NVIDIA NVRTC: CUDA Toolkit EULA, redistributable).
Release {version}: SHA-256 of every file is in SHA256SUMS.txt. Build date: {date}.
"""

DEV_ADDRESS = "qzm7tR2aEJEdMFajwDspjiWefz2F4iDzsys32gE1DK8Dxc59P"


def sha256(path_or_bytes):
    h = hashlib.sha256()
    if isinstance(path_or_bytes, bytes):
        h.update(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
    return h.hexdigest()


def collect(d, win):
    """((name in the archive, path) of the main package, same for the Quantus add-on) of one platform directory."""
    main, addon, ext = [], [], (".exe" if win else "")
    for b in BINS:
        p = os.path.join(d, b + ext)
        if not os.path.isfile(p):
            sys.exit(f"missing {p}")
        main.append((b + ext, p))
    for b in OPTIONAL_BINS:
        p = os.path.join(d, b + ext)
        if os.path.isfile(p):
            main.append((b + ext, p))
    q = os.path.join(d, QUANTUS_BIN + ext)
    if os.path.isfile(q):
        libs = []
        for g in LIB_GLOBS[win]:
            libs += sorted(glob.glob(os.path.join(d, g)))
        if not libs:
            sys.exit(f"{d}: {QUANTUS_BIN} is present but no NVRTC library (its CUDA engine would not start)")
        addon = [(QUANTUS_BIN + ext, q)] + [(os.path.basename(p), p) for p in libs]
    return main, addon


def licenses(dirpath):
    out = []
    for p in sorted(glob.glob(os.path.join(dirpath, "*"))):
        if os.path.isfile(p):
            out.append((f"licenses/{os.path.basename(p)}", open(p, "rb").read()))
    return out


START_BAT = r"""@echo off
rem Starts the Vearl miner. First time: an assistant creates your Vearl wallet.
cd /d "%~dp0"
vearl-miner.exe %*
if errorlevel 1 pause
"""

START_SH = """#!/bin/sh
# Starts the Vearl miner. First time: an assistant creates your Vearl wallet.
cd "$(dirname "$0")"
exec ./vearl-miner "$@"
"""

START_NODE_SH = """#!/bin/sh
# Runs a Vearl node WITHOUT mining (a follower / seed node). Run it from this folder.
cd "$(dirname "$0")"
exec ./node --genesis genesis.json --peer {peer} --listen 0.0.0.0:9000 --data-dir data --log-file node.log "$@"
"""


def is_lib(name):
    return name.endswith(".so") or ".so." in name


def write_tar(path, top, files, extra):
    """files: (name, path) from disk; extra: (name, bytes, mode) generated."""
    with tarfile.open(path, "w:gz") as t:
        for name, p in files:
            ti = t.gettarinfo(p, arcname=f"{top}/{name}")
            ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = (0o644 if is_lib(name) else 0o755), 0, 0, "", ""
            with open(p, "rb") as f:
                t.addfile(ti, f)
        for name, data, mode in extra:
            ti = tarfile.TarInfo(f"{top}/{name}")
            ti.size, ti.mode = len(data), mode
            t.addfile(ti, io.BytesIO(data))


def write_zip(path, top, files, extra):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, p in files:
            z.write(p, f"{top}/{name}")
        for name, data, _mode in extra:
            z.writestr(f"{top}/{name}", data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--linux", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--genesis")
    ap.add_argument("--peer", action="append", default=[], help="HOST:PORT@PUBLIC_KEY of a seed node, written into network.json (repeat for several: e.g. the domain name, then the IP)")
    ap.add_argument("--network", default="vearl-testnet-1")
    ap.add_argument("--licenses", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "licenses"))
    ap.add_argument("--out", default="dist")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    top = f"vearl-{a.version}"
    sums = []
    gen = open(a.genesis, "rb").read() if a.genesis else None
    peer = a.peer[0] if a.peer else "SEED_HOST:9000@SEED_PUBLIC_KEY"
    network = {"name": a.network, "genesis": "genesis.json" if gen else None, "peers": list(a.peer), "listen": "0.0.0.0:9000"}
    net_json = (json.dumps(network, indent=2) + "\n").encode()
    lic = licenses(a.licenses)
    date = time.strftime("%Y-%m-%d", time.gmtime())

    def readme(win):
        start = "Double-click vearl-miner.exe (or start-miner.bat)." if win else "Run ./vearl-miner  (or ./start-miner.sh) in a terminal."
        start += "\n The first time, an assistant creates your Vearl wallet."
        driver = "Windows: 580 or newer" if win else "Linux: 575 or newer; `nvidia-smi` must work"
        adv = "vearl-miner setup (run the assistant again), vearl-miner check (test GPU and files), miner.json (settings)."
        return README.format(version=a.version, date=date, exe=".exe" if win else "", start_here=start, driver=driver, advanced=adv).encode()

    def readme_addon(win):
        return README_ADDON.format(version=a.version, date=date, exe=".exe" if win else "", dev_address=DEV_ADDRESS).encode()

    built = {}
    for plat, d, win, ext in (("linux", a.linux, False, "tar.gz"), ("windows", a.windows, True, "zip")):
        main_files, addon_files = collect(d, win)
        write = write_zip if win else write_tar
        path = os.path.join(a.out, f"{top}-{plat}-x86_64.{ext}")
        extra = [("README.txt", readme(win), 0o644), ("network.json", net_json, 0o644)]
        if win:
            extra.append(("start-miner.bat", START_BAT.replace("\n", "\r\n").encode(), 0o755))
        else:
            extra += [("start-miner.sh", START_SH.encode(), 0o755), ("start-node.sh", START_NODE_SH.format(peer=peer).encode(), 0o755)]
        if gen:
            extra.append(("genesis.json", gen, 0o644))
        write(path, top, main_files, extra)
        sums.append((sha256(path), os.path.basename(path)))
        built[plat] = False
        if addon_files:
            apath = os.path.join(a.out, f"{top}-quantus-addon-{plat}-x86_64.{ext}")
            write(apath, top, addon_files, [("QUANTUS-ADDON.txt", readme_addon(win), 0o644)] + [(n, dd, 0o644) for n, dd in lic])
            sums.append((sha256(apath), os.path.basename(apath)))
            built[plat] = True

    if gen:
        with open(os.path.join(a.out, "genesis.json"), "wb") as f:
            f.write(gen)
        sums.append((sha256(gen), "genesis.json"))
    with open(os.path.join(a.out, "SHA256SUMS.txt"), "w", newline="\n") as f:
        for h, n in sums:
            f.write(f"{h}  {n}\n")
    for h, n in sums:
        print(f"{h}  {n}")
    print(f"quantus add-on archive built: linux={built['linux']} windows={built['windows']}; genesis={'yes' if gen else 'NO (pre-release build, not usable on the real network)'}")


if __name__ == "__main__":
    main()
