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

## 1 bis. What was tested (red team, 2026-10-16)

The protection of release rc13 was attacked with nothing but the public source, the published release and the official maintenance command
`vearl-miner fee-obf`: the masked address (49 bytes) and its digest (64 characters) were found in `node.exe` by computing the mask of the known
address, and **two same-length replacements** redirected the fee to another address. The patched program passed every local check, printed
"the development fee goes to <another address>" and ran both miners normally. Only the server-side audit (no fee worker on the real development
address) would have noticed. A script that does this is a few lines long and works on every copy of the release.

The changes of the next release (section 2) answer this attack: the same script finds nothing to patch in a build made with the release secret
(`deploy/release/fee_secrets.py`), the maintenance command is gone, and a patch that reaches the constants anyway is caught by several independent
checks. The red-team test is repeated on the new build (results in section 2 bis).

## 2. Hardening of the official build (what makes casual tampering fail)

| Attack | Defence in the release | Where |
|---|---|---|
| change the fee address in `miner.json` or on the command line | there is no such setting: the address and the percentage are hidden constants of the program; `miner.json` only holds the operator's own addresses and the Vearl/Quantus split | `dual.rs`, `miner.rs` |
| hex-edit the address or the percentage in the executable | address, percentage and a digest of both are **generated when the program is built**, from a **release secret** (`VEARL_FEE_SALT`, 32 random bytes, different in every release, never in the repository, `deploy/release/fee_secrets.py`): the bytes are not what anybody can compute, the salt itself is stored in four pieces hidden behind the digest of the masked address, so a patch written for one release finds nothing in the next, and changing any one constant breaks the others. If they do not match the digest, Quantus mining is switched OFF (Vearl keeps mining) | `build.rs`, `feeguard.rs` |
| patch the call that builds the fee miner's command line (the address passed to the pool) | before the fee miner is started its command line is compared with the hidden address; a mismatch switches **all** Quantus mining off for the rest of the run, not only the fee miner | `dual.rs`, `feeguard.rs` |
| patch the percentage or the slice length of the scheduler (the fee miner then runs less) | the scheduler's own time counters are compared with the hidden percentage: after ten minutes of Quantus time, a fee share below 60 % of the announced one switches Quantus mining OFF with a message; the counters are also in the beacon | `dual.rs` |
| use the maintenance command to compute the replacement constants | the `fee-obf` command no longer exists in the program | `miner.rs` |
| replace `quantus-miner` by a script or a patched build that ignores the fee address | the node is compiled with the BLAKE3 hash of the official Quantus miner of each platform (`quantus-miner.pins`) and checks it **before every start** of the Quantus process, user or fee; any other file is refused | `feeguard.rs`, `dual.rs` |
| pass extra arguments to the Quantus miner (other pool, other token) | the Quantus command line is built by the program with fixed arguments; the only choice is the engine (`--qtc-engine cuda|vulkan`) | `node.rs` |
| `--partner-cmd` (run any miner, no fee) | exists only in builds made with `--features devtools`, never in a release | `node.rs`, `Cargo.toml` |
| kill the fee process | it is restarted at its next window; a miner that never pays shows up in the audit (section 3) | `dual.rs` |
| edit `network.json` to remove the seeds (so no beacon is received) | the launcher refuses a release without seeds | `miner.rs` |

These checks run inside the same program a fork can modify: they stop accidents, scripts, generic patches, hex edits and the replacement of the
Quantus binary, not a programmer who removes the checks from the source or follows the program in a debugger. Their purpose is to make every
release cost a determined cheater real work again, and to make sure that a cheater who only changes bytes is caught (the checks are independent:
each patched spot leaves the others standing).

## 2 bis. Release procedure for the secret

1. `python3 deploy/release/fee_secrets.py --dir PRIVATE_DIR --version rcNN` (a folder OUTSIDE the repository; it refuses a folder inside it). Back
   the file up: it is needed to rebuild the release bit for bit. Never publish it, never put it in an archive.
2. Build with `VEARL_FEE_SALT`, `VEARL_FEE_ADDRESS` and `VEARL_FEE_PCT` set (the script prints the commands). Without them a build uses the public
   defaults: it works for development and tests, but it is not an official build (its hashes are not in the list of release hashes).
3. Every release gets a NEW salt. Then `pin_quantus.py` (before the build), `make_release.py`, `known_hashes.py`.
4. The seed must run the new release **before** the miners do: the beacon has three more fields (`quantus_ms`, `fee_ms`, `fee_dest`) and an older
   seed does not understand it.

## 2 ter. Result of the red team on the new build

Repeated on 2026-10-16 with two builds of the same source (the attacker's script computes the masked address from the public source, the public
salt and the known development address, then searches the executable):

| Build | The script finds the constants | Running it |
|---|---|---|
| public defaults (what anybody gets by compiling the repository) | **yes**, 1 hit | patchable like rc13, but it is not an official build: not in the list of release hashes, so the auditor reports `unofficial_build` |
| release secret (`VEARL_FEE_SALT`) | **no**, 0 hit; the dev address does not appear in clear text, no 8-byte piece of the salt is found | unmodified: both miners start, "10 % goes to qzm7tR2a...". With ONE byte of the hidden address changed (by someone who knows the salt, to simulate a clumsy patch): "Quantus mining is OFF: the fee constants of this program were altered", Vearl keeps mining, no Quantus process is started |

What this does not show: that a person who reads the program in a debugger or rebuilds the source cannot remove the fee. Nothing can
(section 1). It shows that the cheap attack, a script that works on every copy of a release, no longer works, and that each remaining
route (a fork, a debugger, a hand-made patch per release) costs real work every time a release changes its secret.

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
   Quantus work (an honest miner shows 8 to 10 %: each fee slice loses a few seconds reconnecting to the pool). Statuses: `ok` (>= 6.5 %),
   `low` (3 to 6.5 %), `no_fee` (< 3 % or no fee worker, judged only 3 hours after the first beacon), `warming_up`, `wrong_fee_pct` (the
   beacon declares another percentage), `wrong_fee_dest` (the beacon says that the fee miner of the run mines to another address, or that
   there is none), `low_declared_split` (the scheduler's own counters, after 20 minutes, show a fee share below 60 % of the announced one),
   `unofficial_build` (the hashes are not in the list of official release hashes, `deploy/release/known_hashes.py`), `not_mining_quantus`,
   `offline`, `unverifiable`. The pool numbers are not under the miner's control, so a beacon that lies (a patched build that reports what
   the official one would) still shows up as `no_fee` or `low`.
4. **Output.** `fee-report.json` (private: Vearl address, status, detail) and `compliance.json` (Vearl address -> status only, no
   Quantus address), refreshed by the timer.

What a determined cheater can still do: run a patched build that sends a *truthful-looking* beacon, never register at all, or route
the Quantus mining by another program (running the official Quantus miner next to the Vearl miner, without the combined scheduler, needs
no patch and cannot be prevented: the fee is the price of the scheduler, not of mining Quantus). The audit sees the second and third cases only as the absence of a fee worker for a Vearl
account that is otherwise active; the first one shows up in the pool numbers (the fee worker is missing or too small), which are not
under the miner's control.

## 4. What the project does with the result

* **Rewards.** `tools/testnet_rewards.py --fee-compliance compliance.json` (the hourly rewards script passes it when the file exists) makes an
  address whose status is `no_fee`, `low`, `wrong_fee_pct`, `wrong_fee_dest`, `low_declared_split` or `unofficial_build` **not eligible** for the
  testnet rewards, with the reason shown in the leaderboard. The rule is in `docs/19` section 4, item 4 (to be reviewed with the other rules
  before the site says it).
* **Public counts.** `fee-stats.json` (compliant / not compliant, no address) is published next to the leaderboard: the number of miners that pay
  is visible to everybody.
* **Not done:** the faucet and the free compute gateway do not read the statuses yet.

Other options, from the least to the most strict, that remain the project's choice:

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

`feeguard::tests` (hidden constants intact, every altered constant noticed, the public-salt patch vs a secret salt, fee command must mine to the
development address, pin refusal, beacon signature/binding/freshness including the new fields, registry), `dual::tests` (a scheduler that pays
less than the hidden percentage is caught, a wrong fee command stops all Quantus mining), `tests/feeguard.rs`
(miner -> seed over Noise with key pinning, refusal when not collecting, 404 through the proxy header, forged/stale/foreign beacons,
the scheduler sends its own beacon), `tests/dual.rs` (an unpinned Quantus miner is never started), `deploy/feeaudit/test_feeaudit.py`
(classification).
