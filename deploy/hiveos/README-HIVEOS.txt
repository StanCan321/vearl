VEARL for HiveOS (custom miner) - testnet
=========================================

Vearl is a TESTNET: its tokens have no value, the code has not been audited, the network can be reset (https://vearl.org).
This package runs the Vearl miner on a HiveOS rig. It uses EVERY NVIDIA GPU of the rig of the Ampere generation or newer (RTX 30xx,
40xx, 50xx, A-series): one worker per GPU. Two ways of mining, chosen in the flight sheet:
  POOL (the default) .. the rig joins the official Vearl pool (pool.vearl.org:3333). Small regular payouts, the pool keeps a small fee,
                        and NOTHING to set up on the rig: your Vearl address in the flight sheet is enough (no wallet file, no password).
                        (Paid jobs next to the pool need a wallet and its password: the PC miner does that, this package stays pool-only.)
  SOLO ................ the rig runs its own Vearl node and mines alone: all the blocks it finds are yours and it computes paid jobs,
                        but it needs a wallet on the rig (step 2) and a block is rare with a few GPUs. The public testnet starts on 23 October 2026 at 16:00 UTC; before that the miner only waits
(height 0, 0 sigma/s, "peers=1" means it is connected to our seed).

This package has been tested on a plain Linux machine with a simulated Hive environment, NOT yet on a real HiveOS rig: if something does
not work, tell us what "miner log" shows.

1. Install (flight sheet)
-------------------------
In the HiveOS web page: Flight Sheets > Add, then
  Coin ............................ any (for example "CUSTOM")
  Wallet .......................... or the field "Wallet and worker template": YOUR Vearl address (vrl1...), or %WAL%. In pool mode the payouts
                                    go to this address. To name the worker on the pool write  %WAL%.%WORKER_NAME%  (default: the rig name)
  Miner ........................... Custom
  Miner name ...................... vearl   (exactly this: HiveOS checks it against the name of the file, vearl-<version>.tar.gz, and the
                                    archive holds the folder "vearl". If HiveOS answers 'Custom miner name should be "..."', the file
                                    you used has another name: take vearl-<version>.tar.gz from the releases page)
  Installation URL ................ the link of vearl-<version>.tar.gz from https://github.com/vearl-verified-compute/vearl/releases
  Pool URL ........................ pool.vearl.org:3333 (or leave it EMPTY: same thing). Write  solo  to mine alone instead (step 2).
  Hash algorithm .................. anything: not used
  Extra config arguments .......... optional, see step 4
Apply the flight sheet to the rig: in pool mode it starts mining at once (after the testnet start, see above). Nothing else to do.
Where do I get an address? With the wallet window or "vearl-wallet" of the release on your own PC (never create it on the rig for pool
mining: you do not need the private key there). Do not share your 24 words with anybody.

2. Solo mode: wallet (once, solo only)
--------------------------------------
Only if you chose solo (Pool URL = solo, or mode=solo in the extra config). Pool mode needs none of this.
Open the Hive Shell of the rig (or ssh), then:
    miner stop
    /hive/miners/custom/vearl/setup.sh
    miner start
The set-up wizard creates the wallet of the rig: it shows 24 recovery words ONCE (write them on PAPER, nobody from the project will ever ask
for them) and asks three of them back. When it asks "Remember the wallet password on this computer", answer y: the rig then restarts
by itself after a reboot (anybody who can read /hive-config/vearl on the rig could then spend the tokens of this wallet: it is a testnet).
You can also use a wallet you already have: copy wallet.json to /hive-config/vearl/ before running setup.sh and answer "use it".
Everything that must survive an update (wallet, settings, chain data, logs) is in /hive-config/vearl. Back up wallet.json and your 24 words.

3. Check
--------
    miner log                      (the [miner] line every 10 s: speed, blocks found, balance)
    nvidia-smi                     (every GPU should work once the testnet has started)
The first line of the log lists the GPUs the miner uses: "GPU: 4 GPUs: #0 ..., #1 ...". In pool mode the log shows the pool line:
"[poolminer] 123.4 sigma/s; shares accepted A refused R; ..." ("connected" at the end means the rig is joined to the pool).
HiveOS shows the total speed in MH/s (1 MH/s = one million tiles per second). The miner reports ONE total figure, so the web page shows
it split equally between the GPUs: the total is right, the per-card figures of a rig with different cards are only an average.
In solo mode "Accepted shares" on the page is the number of blocks the rig has found; in pool mode it is the number of shares the pool
accepted (and "rejected" those it refused). In pool mode the speed is converted from sigma/s (one sigma = 4096 tiles with the current
network parameters). Your balance and payouts are on https://pool.vearl.org (type your address).

4. Options (Extra config arguments, words key=value separated by spaces or new lines)
-------------------------------------------------------------------------------------
  mode=pool|solo         the way of mining (overrides the Pool URL field)
  pool=HOST:PORT         another pool than the official one (overrides the Pool URL field)
  worker=NAME            the worker name shown by the pool (1-32 letters, digits, . _ -; default: the rig name)
  gpus=0,1,3             use only these GPUs (CUDA indices; applies to Vearl and to the Quantus add-on)
  max_jobs=2             solo only: paid jobs computed at the same time (default 1)
  quantus_address=qz...  solo only: also mine Quantus (QTC) with the same GPUs: only if you copied the Quantus add-on of the release into the
                         package folder (/hive/miners/custom/vearl); 10 % of the Quantus mining time goes to the Vearl development
                         address, see docs/17 of the repository. Off by default.
  partner_share=80       with the add-on: percent of the time for Quantus (default 80)
Overclocking and power limits are managed by HiveOS (OC tab), not by this miner.

5. Update
---------
Change the Installation URL of the flight sheet to the new package (or run  miner stop, install the new archive, miner start). The wallet
and the settings stay in /hive-config/vearl.

6. Problems
-----------
"No Vearl address for the pool" ............... put your address (vrl1...) in the "Wallet and worker template" of the flight sheet.
"Solo mining is not set up on this rig yet" .... run step 2, or switch to the pool (leave the Pool URL empty).
Pool miner says it cannot connect .............. the rig needs outgoing access to pool.vearl.org port 3333; "miner log" shows the retries.
"Wrong password" / "cannot unlock the wallet" .. run setup.sh again and re-enter the password (or delete /hive-config/vearl/wallet.pw).
Hashrate 0 before 23 October 16:00 UTC ......... normal: the chain has not started.
A card that does not mine ....................... it must be an NVIDIA card of the Ampere generation or newer and the driver must be recent
                                                  (580 or newer for the 50xx cards).
Report problems: https://vearl.org (Discord) or security@vearl.org for a vulnerability.
