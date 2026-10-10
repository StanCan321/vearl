#!/usr/bin/env python3
"""Builds a CLEAN git repository of the project, ready to be pushed to a public host (nothing is pushed, nothing is uploaded).

    python deploy/release/make_public_repo.py --out ../vearl-public [--name "Vearl"] [--email contact@vearl.org] [--single-commit | --no-commit]

Why not push the working repository: its history carries old names, French messages and a personal e-mail address. This script copies
the files that git would publish (tracked or new, minus everything in .gitignore and the private tooling listed in EXCLUDE), refuses to
go on if it finds something that looks like a secret, a personal path or an oversized file, and creates ONE commit ("Initial public
release", or (default) a short series of "Import ..." commits, one per part of the project, so the repository is easy to browse. The
history is honest: it is an import of the current tree, with the real date and the identity given by --name / --email, nothing is
backdated and nobody is invented. Then, from the new folder:

    git remote add origin git@github.com:ORG/REPO.git
    git push -u origin main
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

# private tooling, operations of the project's own servers and heavy media that are not part of the public project (the promo video sources
# and renders weigh about 60 MB). The mail and statistics set-ups describe how one particular server is run: they help nobody else and
# only tell an attacker where to look.
EXCLUDE = ("deploy/pc-worker/", "deploy/private-explorer/", "deploy/mail/", "deploy/stats/", "video/")
# text removed or replaced in the copied files (path, regular expression, replacement): details of the project's own server
REDACT = [
    ("deploy/nginx-vearl.conf", r"[ \t]*# unlisted preview pages.*?include /etc/nginx/vearl-private/\*\.conf;\n\n?", ""),
    ("deploy/nginx-vearl.conf", r"[ \t]*# visitor statistics:.*?location \^~ /stats/ \{.*?\n    \}\n", ""),
    ("docs/06-testnet-runbook.md", r"103\.90\.160\.22", "SERVER_IP"),
    ("docs/06-testnet-runbook.md", r", no firewall\)", ")"),
    ("docs/06-testnet-runbook.md", r"at Porkbun", "at your DNS provider"),
    ("deploy/deploy-site.sh", r"\$\{VEARL_SSH_HOST:-root@vearl\.org\}", "${VEARL_SSH_HOST:?set VEARL_SSH_HOST, for example root@your-server}"),
    ("deploy/deploy-site.sh", r"id_ed25519_vps4", "id_ed25519"),
    ("deploy/release/publish_release.sh", r"\$\{VEARL_SSH_HOST:-root@vearl\.org\}", "${VEARL_SSH_HOST:?set VEARL_SSH_HOST, for example root@your-server}"),
    ("deploy/release/publish_release.sh", r"id_ed25519_vps4", "id_ed25519"),
]
# generated model files above this size are not stored (vearl-core/models/README.md says how to rebuild them)
BIG_MODEL = 10_000_000
BAD_NAMES = re.compile(r"(^|/)(wallet[^/]*\.json|node\.key|id_[a-z0-9_]+|[^/]*\.pem|[^/]*\.password|[^/]*\.pw|token\.txt|\.env)$", re.I)
BAD_TEXT = [
    (re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(rb"discord(?:app)?\.com/api/webhooks/\d+/[\w-]+"), "Discord webhook URL"),
    (re.compile(rb"\b[MN][A-Za-z0-9_-]{23}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,}\b"), "Discord bot token"),
    (re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b"), "GitHub token"),
    (re.compile(rb"(?i)C:[\\/]Users[\\/][A-Za-z0-9_.-]+[\\/]"), "personal Windows path"),
    (re.compile(rb"/home/[a-z0-9_-]+/\.ssh/"), "personal ssh path"),
]
SKIP_TEXT_SUFFIX = (".png", ".jpg", ".jpeg", ".webp", ".woff2", ".ttf", ".wav", ".mp4", ".npz", ".pt", ".ico", ".gz", ".zip", ".webm")
MAX_SCAN = 8_000_000   # the big model and vector files are data, not text to scan line by line
WARN = 50_000_000
FAIL = 100_000_000

# the import is split by part of the project; first matching group wins, leftovers go to the last commit
GROUPS = [
    ("Add license, notices and project documents", ("LICENSE", "NOTICE", "THIRD-PARTY-NOTICES.md", "SECURITY.md", "CONTRIBUTING.md", "CHANGELOG.md", "CODE_OF_CONDUCT.md", "README.md", ".gitignore", ".gitattributes", ".editorconfig", ".github/")),
    ("Import design, specification and threat model", ("docs/",)),
    ("Import the Rust core: consensus, market, network, wallet and miner", ("vearl-core/Cargo.toml", "vearl-core/Cargo.lock", "vearl-core/build.rs", "vearl-core/quantus-miner.pins", "vearl-core/src/")),
    ("Import the CUDA kernels", ("vearl-core/gpu/",)),
    ("Import the test suite and the frozen consensus vectors", ("vearl-core/tests/",)),
    ("Import the independent Python implementations, model tools and demos", ("vearl-core/tools/", "vearl-core/models/")),
    ("Import the website, the explorer and the faucet page", ("site/",)),
    ("Import deployment, release packaging and monitoring", ("deploy/",)),
    ("Import the first prototypes and the brand files", ("proto0_econ/", "proto1_gpu/", "brand/")),
]


def git(root, *args):
    return subprocess.run(["git", "-C", root, *args], capture_output=True, check=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="Vearl")
    ap.add_argument("--email", default="contact@vearl.org")
    ap.add_argument("--no-commit", action="store_true")
    ap.add_argument("--single-commit", action="store_true", help="one commit instead of one per part of the project")
    a = ap.parse_args()
    src = os.path.abspath(a.src)
    out = os.path.abspath(a.out)
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f"{out} exists and is not empty: choose a new folder")
    files = sorted(set(git(src, "ls-files", "-z", "--cached", "--others", "--exclude-standard").decode("utf-8").split("\0")) - {""})
    keep, problems = [], []
    for f in files:
        if f.startswith(EXCLUDE) or not os.path.isfile(os.path.join(src, f)):
            continue
        p = os.path.join(src, f)
        size = os.path.getsize(p)
        if f.startswith("vearl-core/models/") and size > BIG_MODEL and not f.endswith("README.md"):
            print(f"not stored (generated, {size / 1e6:.0f} MB): {f}")
            continue
        if BAD_NAMES.search(f):
            problems.append(f"{f}: file name looks like a secret")
        if size > FAIL:
            problems.append(f"{f}: {size / 1e6:.0f} MB is over the 100 MB limit of GitHub")
        elif size > WARN:
            print(f"warning: {f} is {size / 1e6:.0f} MB (GitHub warns above 50 MB)")
        if size <= MAX_SCAN and not f.lower().endswith(SKIP_TEXT_SUFFIX):
            data = open(p, "rb").read()
            for rx, what in BAD_TEXT:
                if rx.search(data):
                    problems.append(f"{f}: contains a {what}")
        keep.append(f)
    if problems:
        print("REFUSED: fix these first:")
        for x in problems:
            print("  -", x)
        sys.exit(1)
    for f in keep:
        d = os.path.join(out, f)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(os.path.join(src, f), d)
        for rp, rx, repl in REDACT:
            if f == rp:
                text = open(d, encoding="utf-8", newline="").read()
                new = re.sub(rx, repl, text, flags=re.S)
                if new == text:
                    sys.exit(f"redaction did not match in {f}: {rx}")
                open(d, "w", encoding="utf-8", newline="").write(new)
    total = sum(os.path.getsize(os.path.join(out, f)) for f in keep)
    print(f"{len(keep)} files, {total / 1e6:.1f} MB copied to {out}")
    if a.no_commit:
        return
    env = dict(os.environ, GIT_AUTHOR_NAME=a.name, GIT_AUTHOR_EMAIL=a.email, GIT_COMMITTER_NAME=a.name, GIT_COMMITTER_EMAIL=a.email)
    run = lambda *c, **k: subprocess.run(["git", "-C", out, *c], env=env, check=True, capture_output=True, **k)
    run("init", "-q", "-b", "main")
    if a.single_commit:
        run("add", "-A")
        run("commit", "-q", "-m", "Initial public release")
    else:
        left = set(keep)
        for msg, prefixes in GROUPS:
            part = sorted(f for f in left if f.startswith(prefixes))
            if msg == GROUPS[-1][0]:
                part = sorted(left)   # the last commit also takes anything that matched no group
            if not part:
                continue
            run("add", "--pathspec-from-file=-", "--pathspec-file-nul", input="\0".join(part).encode())
            run("commit", "-q", "-m", msg)
            left -= set(part)
    print(subprocess.run(["git", "-C", out, "log", "--reverse", "--format=%h %an <%ae> %s"], capture_output=True, text=True).stdout.strip())
    print("Done. Nothing was pushed. Next: git remote add origin ... ; git push -u origin main")


if __name__ == "__main__":
    main()
