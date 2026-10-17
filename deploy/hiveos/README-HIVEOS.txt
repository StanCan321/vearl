VEARL for HiveOS (custom miner) - testnet
=========================================

Vearl is a TESTNET: its tokens have no value, the code has not been audited, the network can be reset (https://vearl.org).
This package runs the Vearl miner on a HiveOS rig. It uses EVERY NVIDIA GPU of the rig of the Ampere generation or newer (RTX 30xx,
40xx, 50xx, A-series): one worker per GPU. The public testnet starts on 23 October 2026 at 16:00 UTC; before that the miner only waits
(height 0, 0 sigma/s, "peers=1" means it is connected to our seed).

This package has been tested on a plain Linux machine with a simulated Hive environment, NOT yet on a real HiveOS rig: if something does
not work, tell us what "miner log" shows.

1. Install (flight sheet)
-------------------------
In the HiveOS web page: Flight Sheets > Add, then
  Coin ............................ any (for example "CUSTOM")
  Wallet .......................... or the field "Wallet and worker template": the Vearl address of the rig's wallet (vrl1...), or %WAL%
                                    (see step 2: it is only checked against the wallet file of the rig)
  Miner ........................... Custom
  Miner name ...................... vearl
  Installation URL ................ the link of vearl-hiveos-<version>.tar.gz from https://github.com/vearl-verified-compute/vearl/releases
  Hash algorithm / Pool URL ....... anything (for example seed.vearl.org:9000): not used
  Extra config arguments .......... optional, see step 4
Apply the flight sheet to the rig. The first start stops with a message in "miner log": the rig has no wallet yet (step 2).

2. Wallet (once)
----------------
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
The first line of the log lists the GPUs the miner uses: "GPU: 4 GPUs: #0 ..., #1 ...".
HiveOS shows the total speed in MH/s (1 MH/s = one million tiles per second). The miner reports ONE total figure, so the web page shows
it split equally between the GPUs: the total is right, the per-card figures of a rig with different cards are only an average.
"Accepted shares" on the page is the number of blocks the rig has found.

4. Options (Extra config arguments, words key=value separated by spaces or new lines)
-------------------------------------------------------------------------------------
  gpus=0,1,3             use only these GPUs (CUDA indices; applies to Vearl and to the Quantus add-on)
  max_jobs=2             paid jobs computed at the same time (default 1)
  quantus_address=qz...  also mine Quantus (QTC) with the same GPUs: only if you copied the Quantus add-on of the release into the
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
"Vearl is not set up on this rig yet" ......... run step 2.
"Wrong password" / "cannot unlock the wallet" .. run setup.sh again and re-enter the password (or delete /hive-config/vearl/wallet.pw).
Hashrate 0 before 23 October 16:00 UTC ......... normal: the chain has not started.
A card that does not mine ....................... it must be an NVIDIA card of the Ampere generation or newer and the driver must be recent
                                                  (580 or newer for the 50xx cards).
Report problems: https://vearl.org (Discord) or security@vearl.org for a vulnerability.
