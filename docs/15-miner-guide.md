# 15 — Miner guide (testnet)

**Testnet with no value.** Tokens have no price, the chain can be reset, nothing is audited (`docs/06`, `docs/09`). This guide is for running a GPU miner node and verifying for yourself that it computes correctly.

## 1. Hardware and software

| | requirement | status |
|---|---|---|
| System | Linux x86_64 (tested: Ubuntu 24.04) or Windows 10/11 x86_64 (tested 2026-10-07 on an RTX 4070 SUPER: node, GPU kernels and the 10 GPU tests of `tests/gpu.rs` pass, about 33 M attempts/s at n = 1024; build needs Visual Studio 2022 Build Tools + CUDA 13, see `deploy/windows/build-gpu-windows.bat`, or use the prebuilt release) | — |
| GPU | NVIDIA **Ampere or newer** (compute capability ≥ 8.0): A100 (8.0), RTX 30xx / A10 / A40 (8.6), RTX 40xx / L4 / L40 (8.9), H100 (9.0), B200 (10.0), **RTX 50xx (12.0)** | **measured on A10 and on RTX 5060 Ti (Blackwell, 16 GB)**; compiled for 8.0, 8.6, 8.9, 9.0 and, with CUDA 12.8 or newer, 10.0 and 12.0 (+ PTX of the newest); the other cards are not measured |
| Not supported | Turing and older (RTX 20xx, T4, capability 7.5): the kernel uses `cp.async`, `ldmatrix` and `mma.m16n8k32` int8 | the node refuses to start with `--gpu` and says so |
| Driver + CUDA | recent NVIDIA driver, CUDA Toolkit 12.x (`nvcc`); **12.8 or newer for RTX 50xx** | tested: CUDA 12.0 (A10), 12.9 (RTX 5060 Ti) |
| Rust | recent stable (`rustup`) | — |

Without a GPU, a node works entirely on the CPU (slower): omit `--features gpu` and `--gpu`.

## 2. Build

```bash
cd vearl-core
cargo build --release --features gpu --bins
```
By default the binary contains machine code for sm_80, sm_86, sm_89 and sm_90, plus sm_100 and sm_120 (Blackwell, RTX 50) when `nvcc` is 12.8 or newer. For a faster build targeting a single card: `VEARL_CUDA_ARCH=86 cargo build --release --features gpu --bins` (an RTX 5060 Ti: `VEARL_CUDA_ARCH=120`; values ≥ 80 only, comma separated).

## 3. Check YOUR card before mining (mandatory in practice)

```bash
cargo test --release --features gpu --test gpu
```
These 7 tests compare the GPU **bit for bit** with the CPU reference: tile jackpots, batches of sigmas, noise generated on the GPU, exact product, blocks mined on the GPU accepted by the CPU validator. A failure means your card or driver produces different results: do not mine (your blocks would be refused by the network).

Measure the throughput:
```bash
./target/release/gpubench 1024 63
```
Measured references (rank 63, fold_k 64, millions of attempts/s, batch of 8 sigmas): A10 ≈ 71 (n = 1024, `docs/02` annex); RTX 5060 Ti ≈ 49 (2026-10-06). Development measurements on rented cards, not guarantees.

## 4. Wallet and start-up

```bash
./target/release/vearl-wallet new --file miner.json           # 24 words to write on paper; password requested
./target/release/node --genesis genesis.json --listen 0.0.0.0:9000 \
    --peer HOST:9000@NODE_PUBLIC_KEY \
    --miner-wallet miner.json --gpu --data-dir ./data
```
- `genesis.json` and the bootstrap node addresses are published by the testnet operator; `@KEY` pins the node's Noise key (recommended).
- At start-up the node prints `[node] GPU: <name> (compute capability X.Y)`; otherwise it stops with the reason.
- Password: `VEARL_WALLET_PASSWORD`, `VEARL_WALLET_PASSWORD_FILE` or `--password-file` (never as a command-line argument).
- **Miner log**: a mining node prints a status line every 10 s, for example `[miner] height=1234 peers=5 | 7300.0 sigma/s (29.9 M tiles/s) | blocks found=3 (last 12 min ago) | balance=62.00000000 | mempool=0 | one block alone every ~40 min at this rate` (rate measured over the last 10 s and smoothed; the last figure is the average time one miner alone at this rate needs for a block at the current target: a pool or a luckier draw changes it). It warns when no peer is connected. `--log-file miner.log` also writes every line, prefixed with the UTC date and time, to that file (appended; the node then runs as a child process of the launcher, which is invisible in use).
- **Linux GPU release**: the release archive for Linux contains GPU builds made on Ubuntu 22.04 (glibc 2.35, CUDA 12.9 runtime and libstdc++ linked in; they need only the NVIDIA driver, 575 or newer). Run `./start-miner.sh`. Verified 2026-10-08 on an NVIDIA A10 (driver 595, Ubuntu 26.04): the 10 GPU tests pass, about 37 M attempts/s at n = 4096 by `gpubench`, node and pool miner mine on the GPU.
- **Post-quantum address** (optional, to receive): `vearl-wallet pq-address --file miner.json` (prefix `vrlq1`, `docs/14`).

**Compute worker** (get paid for clients' jobs, in addition to block rewards): add `--worker --ask <price in VRL>`. The account must hold funds for the bond (10 % of the escrow); a wrong result is proven by anyone and the bond is forfeited (`docs/07` §11–13).

**Under systemd**: `deploy/install.sh` installs the binaries and the units (the binary installed by this script is CPU-only; for the GPU, build with `--features gpu` then copy `target/release/node` into `/opt/vearl/bin/`). Mining arguments go through a *drop-in* (`systemctl edit vearl-node`), see `docs/06` §4 (and §8 for preparing a new GPU server: driver, CUDA, Rust).

## 4 bis. The miner program and the optional Quantus (QTC) add-on

The release contains ONE program to open, `vearl-miner` (Windows: double-click `vearl-miner.exe`; Linux: `./vearl-miner`), together with the Vearl node. **By default it mines Vearl only, with the whole GPU, and no third-party program is shipped or started.**

Mining Quantus with the same GPU is an **optional add-on**, a separate download (`vearl-<version>-quantus-addon-<platform>`): the official Quantus miner (Apache-2.0, built unmodified from `github.com/Quantus-Network/quantus-miner`, commit in `licenses/THIRD-PARTY-NOTICES.txt`) and the NVIDIA NVRTC libraries it loads. You extract it into the miner folder, then run `vearl-miner setup` and answer yes to "Also mine Quantus with this miner?" (the default answer is no). Until then the add-on is never started. The rest of this section describes the add-on.

- **First start**: the assistant (1) asks for your Vearl wallet: it creates one (password, 24 recovery words shown ONCE, then it asks you to retype three of them to prove you wrote them down) or uses a wallet file you already have; (2) only if the add-on is installed, asks whether you also want to mine Quantus (default: no) and, if yes, your Quantus address (`qz...`). Settings go to `miner.json`; `vearl-miner setup` runs the assistant again, `vearl-miner check` tests the GPU, the network files and, if installed, the Quantus add-on.
- **After that**: it starts the node with the right options (Vearl only, the whole GPU, unless Quantus is enabled) and, when Quantus is enabled, the Quantus miner, shares the GPU (default **80 % Quantus, 20 % Vearl** in cycles of 60 s) and **gives the whole GPU to Vearl as soon as a paid job is assigned to your worker, even in the middle of a Quantus window**; Quantus resumes when the job is done. Closing the window (Windows: a job object kills the node and the Quantus miner; Linux: parent-death signal plus the node's watchdog) stops everything.
- **Engine check**: at start the launcher benchmarks the Quantus miner for 3 seconds with the CUDA engine, and falls back to the Vulkan engine (3 to 4 times slower, no CUDA library needed) if CUDA does not work on that machine. Measured 2026-10-08: RTX 4070 SUPER 382 MH/s (CUDA) / 139 MH/s (Vulkan); A10 274 MH/s (CUDA).
- **How a paid job takes the GPU** (`dual.rs`, `net.rs`): during the Quantus window the Vearl mining loop is paused but keeps reading the job queue every 400 ms; when a job is claimed by your worker (the claim transaction is relayed to the network and must be in a block, about one block time) it raises the flag `native_work = false`, the scheduler (checks every 250 ms) freezes the Quantus miner and unpauses Vearl: measured under one second in `tests/dual.rs`; `tests/net.rs::a_paused_worker_still_notices_a_paid_job` runs it with real nodes. A worker needs funds for the bond (10 % of the escrow): a new miner earns them from its Vearl blocks first.
- **Status** (every 10 s, also in `miner.log`): `[miner] height=... | rate | blocks found | balance | one block alone every ~...` and `[dual mining] GPU now: Quantus | share of the time since start: Quantus 72 % + dev fee 8 %, Vearl 18 %, paid jobs 2 %`. The Quantus miner's own output is in `external-miner.log` (and `fee-miner.log`), its metrics on `http://127.0.0.1:9900/metrics`.

Manual use of the same machinery (what the launcher does):

```
node --gpu --miner-wallet miner.json --worker \
     --qtc-miner /path/to/quantus-miner \
     --qtc-address YOUR_OWN_QUANTUS_ADDRESS \
     --partner-share 80 --partner-cycle 60
```
(a Quantus miner built from source: `cargo build -p miner-cli --release`, binary `quantus-miner`; the NVRTC library of CUDA 12.x must be next to it or on the library path for the CUDA engine.)

- **You mine to your own address.** `--qtc-address` is the payout address of the operator (starts with `qz`, created with the official Quantus wallet). The pool pays you directly; Vearl never holds your funds. Default pool: quanpool.com (`--qtc-pool`, `--qtc-cert` to change it, `--qtc-args` for the miner's own options, by default `--cuda-gpu --cpu-workers 0`, which uses every CUDA GPU of the machine: see section 4 quater).
- **Protection and audit of the fee** (what is locked, what is checked, what nobody can prevent): `docs/17-fee-protection.md`.
- **Development fee: 10 % of the Quantus mining time** is spent mining to the Vearl development address `qzm7tR2aEJEdMFajwDspjiWefz2F4iDzsys32gE1DK8Dxc59P` (worker `vrl-fee-<id>`). The address and the percentage are constants of the program, there is no option to change them, and both are printed at every start. The fee is paid in slices of about 8 seconds so that the pool sees a steady stream of shares. **This is a convention of the official build, not a protocol rule**: the source is open, anyone can build a version without the fee or run another miner; a fork loses nothing but the support of the project. The fee address is used for the development of Vearl; this is a commitment of the project, not something the program or the Quantus chain enforces.
- `--partner-share 80` (default) gives 80 % of every cycle to Quantus and 20 % to Vearl; `--partner-cycle 60` is the cycle length in seconds. With `--partner-share 0` the external miner never runs, with `100` Vearl never mines.
- **The two never compute at the same moment**: during the Quantus window the Vearl mining loop is paused, during the Vearl window the Quantus miner is frozen. The split is therefore exact, with no contention for the tensor cores. The external miner is frozen with `SIGSTOP` on Linux and `NtSuspendProcess` on Windows (it keeps its CUDA context and VRAM and resumes instantly; its connection to the pool survives).
- **Paid work comes first**: if the node has a job it can compute (`--worker`), Vearl takes the whole GPU and the external miner stays frozen until the job is done (see above for the latency).
- If the node dies while a miner is frozen, a small watchdog (Unix) wakes it up and stops it. Each miner's output goes to `external-miner.log` and `fee-miner.log`. A miner that exits is restarted at its next window.
- `--partner-cmd "COMMAND"` runs any other external miner in the same way, **without the fee** (tests, other coins).
- Vearl does not read, verify or pay anything for the Quantus work: Quantus rewards come from the pool (1 % pool fee on quanpool.com at the time of writing), Vearl rewards from the Vearl blocks found during the Vearl windows. Do not count on QTC income: on 2026-10-06 an RTX 5060 Ti measured 230.7 MH/s with the CUDA path of the official miner, about 0.015 QTC (roughly 1.3 USD at the pool's displayed price) per day over a network of 28 TH/s; these figures change every day.
- Both miners keep their memory on the card; check that the VRAM is enough. A pool may drop a miner that stays silent for too long: keep the cycle at 60 s or less.
- Verified on a Linux server: the official miner connected to quanpool.com, submitted 183 shares in 10 minutes without a single rejection, and the pool credited them to the worker; the duty cycle and the 10 % fee meter are covered by tests in `src/dual.rs`.

## 4 quater. Several GPUs (rigs)

`vearl-miner`, the node and `vearl-poolminer --gpu` use **every NVIDIA GPU of compute capability 8.0 or newer** (Ampere and later) of the machine, with no setting. At start the node prints the list (`[node] GPU: 4 GPUs: #0 ..., #1 ...`).

- **Vearl mining**: one worker thread per GPU, each with its own copy of the job data in its own memory. The search for a winning sigma is cut into batches handed out in turn, so a faster card simply does more of them and cards of different models can be mixed. A block found by any GPU is a normal block (the header does not say which card found it). In a pool, the GPUs share the nonce range of the job in interleaved batches.
- **Quantus add-on**: the Quantus miner is started without `--gpu-devices`, so it starts one worker per CUDA GPU (up to rc16 the launcher forced `--gpu-devices 1`: only the first GPU of a rig mined, for Quantus and for Vearl).
- **Use only some GPUs**: set the standard NVIDIA variable before starting, for example `CUDA_VISIBLE_DEVICES=0,2 ./vearl-miner` (Windows: `set CUDA_VISIBLE_DEVICES=0,2`). It applies to the node and to the Quantus miner alike. `VEARL_GPUS=0,2` selects the CUDA indices for Vearl mining only (the Quantus miner ignores it).
- **Paid jobs** (rendering, inference, linear algebra) still run on one GPU (GPU 0 of the process): one job is one computation, and workers take several jobs with `--max-jobs`. Spreading a single job over several cards is not done.
- **Not measured**: the scaling was tested with several workers on one GPU (same code path) and on a single RTX 4070 SUPER; it has not yet been measured on a real multi-GPU rig. Overclocking and power limits are not managed by the miner: use `nvidia-smi -pl` or the tools of your system.

## 4 ter. Mining in a pool

A pool lets several miners search together and share the rewards in proportion to their work (PPLNS, pool fee 1 % by default). You only need a payout address; no private key goes to the pool.

```
vearl-poolminer --pool POOL_HOST:3333 --address vrl1yourpayoutaddress --worker rig1 --gpu     # NVIDIA, CUDA build
vearl-poolminer --pool POOL_HOST:3333 --address vrl1yourpayoutaddress --threads 8             # CPU
```

- It prints its rate (`sigma/s`), accepted/rejected shares, blocks and receipts every 10 s, and reconnects by itself. The pool sends you signed receipts of your work (`vpool/2`); they are saved in `pool-receipts.jsonl` and are your proof if the pool credits you less than it signed (`vearl-pool-audit`, `docs/16` section 13). The testnet points of a registered pool are shared by verified work (`docs/19` section 3 bis): the pool's page shows the points of your address, and the miner prints them (`testnet points`) in its status line. Rewards are credited after the pool's block is 10 blocks deep and paid on-chain by the pool when your balance reaches the pool's minimum (1 VRL by default); the pool's `/api/miner/<your address>` shows balance, pending and payments.
- A pool mines the **Native Work only**: it does not compute paid jobs. For the compute market run your own node as a worker (`node --worker`); both can run on the same machine at different times.
- Trust: the pool holds the rewards until it pays them, and a dishonest pool can simply not pay. Choose pools you can check (their `/api/stats` exposes `money.solvent`), and check your on-chain balance. Protocol and operator guide: `docs/16-pool-protocol.md`.

## 5. What to know

- **Reward**: block reward (6.66666666 VRL, halved every 3,153,600 blocks = every year, floor 0.16666666 VRL), plus, for a useful-work block, a bonus taken from what the client burned (never more than that amount: `docs/04`). Testnet: no value.
- **Difficulty**: ASERT, aiming at 10 s per block (`docs/13` §5). A miner that switches on and off gains nothing by doing so.
- **Network**: open the P2P port (9000/tcp). Do not expose the API (8545) directly: nginx with TLS and rate limiting (`deploy/nginx`).
- **Machine security**: a node needs no root rights to run; the systemd unit isolates it (dedicated user, read-only file system except the data). No external audit of the code.
- **Known limits**: no protection against selfish mining, no finality (reorganizations are only braked), P2P channel not post-quantum (`docs/13` §6, `docs/14`).
