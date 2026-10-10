# 17 - Protection and audit of the development fee

The optional Quantus add-on of the miner (docs/15 section 4 bis) mines Quantus for the operator and spends **10 % of its Quantus mining
time** on the Vearl development address. It is OFF by default: the miner mines Vearl only (no third-party program, no fee, no beacon)
unless the operator installs the add-on and answers yes in `vearl-miner setup`. This document says what protects that fee when it is
enabled, what checks it, and, honestly, what nothing can prevent.

## 1. What cannot be prevented

The Vearl source code is open and the Quantus miner is an open-source program that anybody can run directly. Therefore **nobody can
technically force** an operator to pay the fee: a person who compiles a fork without the fee, patches the executable, edits a memory
byte or simply runs the official Quantus miner by hand will not pay it. No software lock changes that (it would only be reversed by the
next patch). What the project can do is make the official miner hard to tamper with *by accident or by a quick edit*, **detect** the
miners that do not pay, and **decide what the project's own services do** about it.

## 2. Hardening of the official build (what makes casual tampering fail)

| Attack | Defence in the release | Where |
|---|---|---|
| change the fee address in `miner.json` or on the command line | there is no such setting: the address and the percentage are constants of the program; `miner.json` only holds the operator's own addresses and the Vearl/Quantus split | `dual.rs`, `miner.rs` |
| hex-edit the address in the executable | the address is stored masked (not searchable) and checked against a BLAKE3 digest at start; if it does not match, Quantus mining is switched OFF (Vearl keeps mining) | `feeguard.rs` |
| replace `quantus-miner` by a script or a patched build that ignores the fee address | the node is compiled with the BLAKE3 hash of the official Quantus miner of each platform (`quantus-miner.pins`) and checks it **before every start** of the Quantus process, user or fee; any other file is refused | `feeguard.rs`, `dual.rs` |
| pass extra arguments to the Quantus miner (other pool, other token) | the Quantus command line is built by the program with fixed arguments; the only choice is the engine (`--qtc-engine cuda|vulkan`) | `node.rs` |
| `--partner-cmd` (run any miner, no fee) | exists only in builds made with `--features devtools`, never in a release | `node.rs`, `Cargo.toml` |
| kill the fee process | it is restarted at its next window; a miner that never pays shows up in the audit (section 3) | `dual.rs` |
| edit `network.json` to remove the seeds (so no beacon is received) | the launcher refuses a release without seeds | `miner.rs` |

These checks run inside the same program a fork can modify: they stop accidents, scripts, hex edits and the replacement of the Quantus
binary, not a programmer who removes the checks from the source.

## 3. Verification: signed beacons and a public-API audit

1. **Beacon.** Every 5 minutes (first one 20 s after the start) a miner in Quantus mode sends each seed of `network.json` a message
   `FeeBeacon` over the normal Noise-encrypted connection (the seed's key is checked against the pinned key): the Vearl account, the
   operator's Quantus address, the worker id (`vrl-<id>`, `vrl-fee-<id>`, where `<id>` is the first 4 bytes of the Vearl account), the
   Quantus/Vearl split, the fee percentage the program uses, the program version and the BLAKE3 hashes of the running `node` and of
   `quantus-miner`. It is signed with the operator's Vearl key (the wizard says so before the first start). No secret and no key leaves
   the machine.
2. **Collection.** Only a seed started with `node --collect-fee-beacons` stores beacons (newest per account, at most 50 000, 10 minutes
   of clock tolerance, signature and worker id checked). They are **never relayed** to other peers and are readable only through
   `GET /api/fee/beacons` on the seed's loopback API (the public nginx answers 404 for that path).
3. **Audit.** `deploy/feeaudit/feeaudit.py` (a timer on the seed) reads the beacons and the **public API of the Quantus pool**
   (`https://quanpool.com/api/miner/mainnet/<address>`, per-worker time-averaged hash rates). The operator's worker `vrl-<id>` and the fee
   worker `vrl-fee-<id>` on the development address are two time slices of the same GPU, so the fee worker must carry about 10 % of the
   Quantus work. Statuses: `ok` (>= 5 %), `low` (2 to 5 %), `no_fee` (< 2 % or no fee worker, judged only 3 hours after the first
   beacon), `warming_up`, `wrong_fee_pct` (the beacon declares another percentage), `unofficial_build` (the hashes are not in the list of
   official release hashes, `deploy/release/known_hashes.py`), `not_mining_quantus`, `offline`, `unverifiable`.
4. **Output.** `fee-report.json` (private: Vearl address, status, detail) and `compliance.json` (Vearl address -> status only, no
   Quantus address), refreshed by the timer.

What a determined cheater can still do: run a patched build that sends a *truthful-looking* beacon, never register at all, or route
the Quantus mining by another program. The audit sees the second and third cases only as the absence of a fee worker for a Vearl
account that is otherwise active; the first one shows up in the pool numbers (the fee worker is missing or too small), which are not
under the miner's control.

## 4. What to do with the result (project policy)

The tool measures; it does not punish. Reasonable options, from the least to the most strict:

* publish the counts (`ok` / total) and a list of the non-compliant Vearl addresses on the site or on the Discord;
* serve the project's own services only to `ok`/`warming_up` accounts: the faucet, the free compute gateway (Vearl Compute), the project's
  pool, the leaderboard of the explorer (they can read `compliance.json`);
* change nothing in consensus: the chain cannot see the Quantus work, and a rule that depends on an off-chain API would be a trust
  assumption in the pool's API and in the project's auditor.

## 5. Operating it

* Seed: `node ... --collect-fee-beacons` (the systemd unit passes it through `/etc/vearl/node.env`: `VEARL_EXTRA_ARGS=--collect-fee-beacons`).
* Auditor: `deploy/feeaudit/install.sh` installs `/opt/vearl/feeaudit/feeaudit.py`, a service and a 30-minute timer; outputs in
  `/var/lib/vearl/feeaudit/`. After each release run `known_hashes.py` and put the file at `/etc/vearl/known-hashes.json`.
* Release: `deploy/release/pin_quantus.py` **before** building `node` and `vearl-miner` (the pins are compiled in), then
  `make_release.py`, then `known_hashes.py`. Changing `quantus-miner` (even a rebuild of the same source) means a new pin and new builds.
* Privacy: the seed operator sees which Vearl accounts mine Quantus and to which Quantus address. Both are already linkable through the
  pool's public worker names; the README and the wizard state it. Do not publish the private report.

## 6. Tests

`feeguard::tests` (address integrity and masking, pin refusal, beacon signature/binding/freshness, registry), `tests/feeguard.rs`
(miner -> seed over Noise with key pinning, refusal when not collecting, 404 through the proxy header, forged/stale/foreign beacons,
the scheduler sends its own beacon), `tests/dual.rs` (an unpinned Quantus miner is never started), `deploy/feeaudit/test_feeaudit.py`
(classification).
