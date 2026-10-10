#!/usr/bin/env python3
"""Partition then MERGE: two groups of REAL nodes each mine on their own side (two competing chains, with their
own jobs and transfers), then we link them. Verdict: (1) the 4 nodes converge to the chain of greatest cumulative
work (deep reorganisation by replay), (2) the money supply invariant holds everywhere, (3) no node crashed,
(4) the merged network works: an honest job is paid and a cheating job is penalised AFTER the reorganisation.

    python3 tools/fork_test.py [PARTITION_DURATION_SECONDS=90]       # after cargo build --release --bins
"""
import os, re, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from netlab import Lab, run

DUR = int(sys.argv[1]) if len(sys.argv) > 1 else 90
NODES = {
    "A": dict(port=9501, seed=1, api=8811, extra=[], peers=["B"]),                                  # group 1
    "B": dict(port=9502, seed=2, extra=["--worker", "--ask", "2"], peers=["A"]),                      # honest worker
    "C": dict(port=9503, seed=3, extra=["--worker", "--cheat"], peers=["D"]),                         # group 2, cheating worker
    "D": dict(port=9504, seed=6, api=8812, extra=[], peers=["C"]),
}
lab = Lab(NODES)
for n in NODES:
    lab.start(n)
    time.sleep(0.4)
time.sleep(5)
print(f"[fork] folder {lab.dir}; group 1 = A,B; group 2 = C,D (isolated from each other)")
CL = "seed:4"


def jid(out):
    m = re.search(r"job_id = ([0-9a-f]{64})", out)
    return m.group(1) if m else None


# ---- phase 1: two independent chains, with jobs on each side ----
jobs = []
for hub, worker, cheat in [("A", "B", False), ("D", "C", True), ("A", "B", False), ("D", "C", True)]:
    for kind in ("job", "mlp"):
        seed = str(1000 + len(jobs))
        if kind == "job":
            out = run("vearl-cli", "job-submit", lab.addrof(hub), lab.addrof(worker), CL, "5", seed)
        else:
            out = run("vearl-cli", "mlp-submit", lab.addrof(hub), lab.addrof(worker), CL, "5", "2", seed)
        i = jid(out)
        if i:
            jobs.append(dict(kind=kind, id=i, hub=hub, worker=worker, cheat=cheat, seed=seed))
    run("vearl-cli", "transfer", lab.addrof(hub), CL, "seed:5", "3")
# one transfer SPECIFIC TO EACH GROUP (different senders): after the merge, the one of the losing group must be put back into the
# mempool and included in the winning chain (no confirmed transaction must disappear)
run("vearl-cli", "transfer", lab.addrof("A"), "seed:5", "seed:1", "1.11")
run("vearl-cli", "transfer", lab.addrof("D"), "seed:1", "seed:5", "2.22")
t0 = time.time()
while time.time() - t0 < DUR:
    time.sleep(10)
    sts = {n: lab.status(n) for n in NODES}
    print(f"[fork] t={int(time.time()-t0)}s heights {{{', '.join(f'{n}: {s[0]}' for n, s in sts.items() if s)}}}")
# a block can be in transit at the time of reading: we let each group agree (15 s at most), otherwise it is a real defect
for _ in range(30):
    sts = {n: lab.status(n) for n in NODES}
    if sts["A"][1] == sts["B"][1] and sts["C"][1] == sts["D"][1]:
        break
    time.sleep(0.5)
assert sts["A"][1] == sts["B"][1] and sts["C"][1] == sts["D"][1], "each group must be internally consistent"
print(f"[fork] before merge: group 1 height {sts['A'][0]} work {sts['A'][2]}; group 2 height {sts['D'][0]} work {sts['D'][2]}; different tips: {sts['A'][1] != sts['D'][1]}")
loser = "group1" if sts["A"][2] < sts["D"][2] else "group2"

# ---- phase 2: merge (D restarts with A and C as peers) ----
lab.kill("D")
time.sleep(2)
lab.start("D", peers=["A", "C"])
t1 = time.time()
conv = False
while time.time() - t1 < 180:
    time.sleep(3)
    sts = {n: lab.status(n) for n in NODES}
    if all(sts.values()) and len({s[1] for s in sts.values()}) == 1:
        conv = True
        break
print(f"[fork] after merge (t+{int(time.time()-t1)}s): heights {{{', '.join(f'{n}: {s[0]}' for n, s in sts.items() if s)}}} -> {'CONVERGENCE' if conv else 'NO CONVERGENCE'} ; the chain of {loser} was abandoned")
crashed = [n for n, p in lab.procs.items() if p.poll() is not None]
inv = {n: (lab.api_status(n) or {}).get("supply_invariant") for n in ("A", "D")}
print(f"[fork] money supply invariant: {inv}; nodes that stopped by themselves: {crashed or 'none'}")

# ---- phase 3: does the merged network work? an honest job and a cheating job after the reorganisation ----
res = {}
for label, worker, cheat in [("honest", "B", False), ("cheater", "C", True)]:
    out = run("vearl-cli", "job-submit", lab.addrof("A"), lab.addrof(worker), CL, "5", str(7000 + len(res)))
    i = jid(out)
    res[label] = i
    print(f"[fork] job {label} submitted after the merge: {i and i[:8]}")
final = {}
deadline = time.time() + 150
pending = dict(res)
while pending and time.time() < deadline:
    time.sleep(3)
    for label, i in list(pending.items()):
        st = run("vearl-cli", "job-status", lab.addrof("A"), i)
        if "Posted" in st:
            w = "B" if label == "honest" else "C"
            out = run("vearl-cli", "job-finish", lab.addrof("A"), lab.addrof(w), CL, i, str(7000 + list(res).index(label)))
            m = re.search(r"final job status: (\w+)", out)
            final[label] = m.group(1) if m else out[-80:]
            del pending[label]
time.sleep(5)
# the two transfers (one per group) must be in the final chain: sender nonces = 1
nonces = {}
for k in range(30):
    nonces = {sd: int(re.search(r"nonce (\d+)", run("vearl-cli", "balance", lab.addrof("A"), f"seed:{sd}")).group(1)) for sd in (1, 5)}
    if nonces == {1: 1, 5: 1}:
        break
    time.sleep(2)
transfers_ok = nonces == {1: 1, 5: 1}
print(f"[fork] transfers of both groups present in the final chain (nonces {nonces}): {transfers_ok}")
print(f"[fork] final statuses after the merge: {final} (expected: honest Settled, cheater Slashed)")
same = False
for _ in range(30):                      # a block can be in transit at the time of reading (blocks every ~0.6 s here)
    sts = {n: lab.status(n) for n in NODES}
    same = all(sts.values()) and len({s[1] for s in sts.values()}) == 1
    if same:
        break
    time.sleep(0.5)
print(f"[fork] identical tips at the end of the test: {same}")
ok = conv and same and not crashed and all(v for v in inv.values() if v is not None) and final.get("honest") == "Settled" and final.get("cheater") == "Slashed" and transfers_ok
lab.stop_all()
print("[fork] VERDICT:", "OK" if ok else "FAIL", f"(logs in {lab.dir})")
sys.exit(0 if ok else 1)
