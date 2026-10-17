#!/usr/bin/env python3
"""REPLAYABLE AUDIT FILE: runs, in one command, all the independent verifications of the repository and prints an OK / FAIL table.

  - re-validation of the frozen vectors by the INDEPENDENT PYTHON IMPLEMENTATIONS (work, chain, market, programs, block-wise products, level 2);
  - blocks / transactions WRITTEN by these implementations and replayed by the RUST validator (the .ext files);
  - DIFFERENTIAL FUZZ Rust against Python (market, tensor programs, block-wise products): same decisions, same final state;
  - independent operation engine compared tensor by tensor with the Rust on random programs;
  - (option --cargo) the whole Rust test suite.

    python3 tools/audit_all.py [--quick] [--cargo] [--features gpu]          # from vearl-core, after cargo build --release --bins
Prerequisites: Rust (binaries in target/release), Python 3 with blake3 numpy cryptography; no GPU needed (except --features gpu).
"""
import os, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUICK = "--quick" in sys.argv
FEAT = sys.argv[sys.argv.index("--features") + 1] if "--features" in sys.argv else ""
os.chdir(ROOT)
BIN = "target/release"
V = "tests/vectors"
TMP = tempfile.mkdtemp(prefix="vearl-audit-")
for f in os.listdir(V):
    if os.path.isfile(f"{V}/{f}"):      # (the fixtures of the pool points are a directory, replayed by the Rust tests)
        shutil.copy(f"{V}/{f}", TMP)          # the Python validators write their .ext files next to the input: we work on a copy
T = lambda f: f"{TMP}/{f}"
n_fuzz = 300 if QUICK else 1500
py = sys.executable

checks = [
    ("work layer (spec_ref)", [py, "tools/spec_ref.py", T("consensus-work-v1.json")]),
    ("Native chain + transfers (chain_ref)", [py, "tools/chain_ref.py", T("chain-native-v1.json")]),
    ("compute market (market_ref)", [py, "tools/market_ref.py", T("market-v1.json")]),
    ("tensor programs (prog_ref)", [py, "tools/prog_ref.py", T("prog-v1.json")]),
    ("block-wise products (tiled_ref)", [py, "tools/tiled_ref.py", T("tiled-v1.json")]),
    ("level 2: Monte Carlo / rendering kernels (l2_ref)", [py, "tools/l2_ref.py", T("l2-kernels-v1.json")]),
    ("block-wise products: extension written by the Python", [py, "tools/tiled_ref.py", T("tiled-v1.json"), "--extend", T("tiled-v1.ext.json")]),
    ("Rust replays the Native chain extended by the Python", [f"{BIN}/vearl-chain-vectors", "verify", T("chain-native-v1.ext.json")]),
    ("Rust replays the market extended by the Python", [f"{BIN}/vearl-market-vectors", "verify", T("market-v1.ext.json")]),
    ("Rust replays the programs written by the Python", [f"{BIN}/vearl-prog-vectors", "verify", T("prog-v1.ext.json")]),
    ("Rust replays the block-wise products written by the Python", [f"{BIN}/vearl-tiled-vectors", "verify", T("tiled-v1.ext.json")]),
    (f"differential fuzz of the market ({n_fuzz} blocks)", [py, "tools/diff_fuzz.py", str(n_fuzz), "11", "--rust", f"{BIN}/vearl-market-vectors"]),
    (f"differential fuzz of the programs ({n_fuzz} blocks)", [py, "tools/diff_fuzz_prog.py", str(n_fuzz), "11", "--rust", f"{BIN}/vearl-prog-vectors"]),
    (f"differential fuzz of the block-wise products ({n_fuzz} blocks)", [py, "tools/diff_fuzz_tiled.py", str(n_fuzz), "11", "--rust", f"{BIN}/vearl-tiled-vectors"]),
    ("ASERT difficulty: official aserti3-2d vectors + Rust vs Python (20,000 cases)", [py, "tools/asert_ref.py", "--rust", f"{BIN}/vearl-chain-vectors", "20000"]),
    ("post-quantum SLH-DSA signatures: Rust crate vs Python verifier (FIPS 205)", [py, "tools/pq_ref.py", "--rust", f"{BIN}/vearl-pq-vectors"]),
    (f"independent operation engine vs Rust ({n_fuzz * 2} programs)", [py, "tools/vearl_ref.py", "fuzz", str(n_fuzz * 2), f"{BIN}/vearl-model"]),
]
if "--cargo" in sys.argv:
    checks.append(("Rust test suite", ["cargo", "test", "--release", "--no-fail-fast"] + (["--features", FEAT] if FEAT else [])))

env = dict(os.environ, PROG_VECTORS=T("prog-v1.json"), TILED_VECTORS=T("tiled-v1.json"), MARKET_VECTORS=T("market-v1.json"))
rows, bad = [], 0
for name, cmd in checks:
    t0 = time.time()
    if os.name == "nt":      # the programs of target/release have the .exe suffix
        cmd = [os.path.abspath(c + ".exe") if isinstance(c, str) and c.startswith(BIN + "/") and not c.endswith(".exe") else c for c in cmd]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
        ok, out = r.returncode == 0, (r.stdout + r.stderr).strip()
    except Exception as e:  # missing binary, timeout...
        ok, out = False, str(e)
    bad += not ok
    last = out.splitlines()[-1][:110] if out else ""
    rows.append((name, ok, time.time() - t0, last))
    print(f"{'OK    ' if ok else 'FAIL  '} {name:64s} {time.time() - t0:6.1f} s  {last}", flush=True)
    if not ok:
        print("   ...", "\n   ".join(out.splitlines()[-6:])[:900], flush=True)
print(f"\n{len(rows) - bad}/{len(rows)} checks passed" + ("" if not bad else f" ; {bad} FAILURE(S)"))
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if bad else 0)
