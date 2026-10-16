#!/usr/bin/env python3
"""Creates the build secrets of an official release: the salt that hides the development-fee constants in the executables (docs/17).

    python3 deploy/release/fee_secrets.py --dir /path/OUTSIDE/the/repository --version rc14 [--address qz...]

Writes DIR/fee-secrets-<version>.env with VEARL_FEE_SALT (32 random bytes), VEARL_FEE_ADDRESS and VEARL_FEE_PCT, and prints how to build with it.
Every release must use a NEW salt: the constants of the executable (masked address, percentage, digest) are then different, so that a
patch written for one release does not apply to the next. Keep the file private and backed up (it is needed to rebuild a release bit for bit),
never commit it, never put it in the public repository or in a release archive. Without these variables a build uses the public defaults: it
is a development build, which the auditor does not recognise as official (its hashes are not in the list of release hashes).
"""
import argparse
import os
import secrets
import sys

PUBLIC_ADDRESS = "qzm7tR2aEJEdMFajwDspjiWefz2F4iDzsys32gE1DK8Dxc59P"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="private folder, outside the repository")
    ap.add_argument("--version", required=True, help="release name, for example rc14")
    ap.add_argument("--address", default=PUBLIC_ADDRESS, help="Quantus address of the development fee")
    ap.add_argument("--pct", type=int, default=10)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    target = os.path.abspath(a.dir)
    if os.path.commonpath([target, repo]) == repo:
        sys.exit("refused: the secrets folder must be OUTSIDE the repository (it would end up in git or in an archive)")
    if not (40 <= len(a.address) <= 60 and a.address.startswith("qz") and a.address.isalnum()):
        sys.exit("--address: a Quantus address (qz...)")
    if not 0 <= a.pct <= 100:
        sys.exit("--pct: 0..100")
    os.makedirs(target, exist_ok=True)
    path = os.path.join(target, f"fee-secrets-{a.version}.env")
    if os.path.exists(path):
        sys.exit(f"refused: {path} already exists (a release keeps its salt; use another --version for a new release)")
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(f"VEARL_FEE_SALT={secrets.token_hex(32)}\nVEARL_FEE_ADDRESS={a.address}\nVEARL_FEE_PCT={a.pct}\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    print(f"written: {path}")
    print("Build the release with these variables set, for example:")
    print("  Linux/macOS:  set -a; . " + path + "; set +a; cargo build --release ...")
    print("  Windows (cmd): for /f \"tokens=1,* delims==\" %a in (" + path + ") do set %a=%b   then cargo build --release ...")
    print("Then regenerate the list of release hashes (known_hashes.py) and install it on the seed.")


if __name__ == "__main__":
    main()
