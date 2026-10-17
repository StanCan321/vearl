#!/usr/bin/env bash
# One-time setup of the rig, to run in the Hive Shell (or over ssh) with the miner stopped:
#     miner stop
#     /hive/miners/custom/vearl/setup.sh
#     miner start
# It starts the wizard of vearl-miner: it creates the wallet of the rig (the 24 recovery words are shown ONCE: write them on paper) or
# uses a wallet file you copied to /hive-config/vearl/wallet.json, and can remember the password so that the rig restarts by itself.
cd "$(dirname "$0")" || exit 1
. ./common.sh
vearl_prepare_home
cd "$VEARL_DATA" || exit 1
exec ./vearl-miner setup --home "$VEARL_DATA"
