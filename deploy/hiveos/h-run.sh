#!/usr/bin/env bash
# HiveOS starts the miner with this script (after h-config.sh). Everything the miner prints goes to the Hive log of the miner.
cd "$(dirname "$0")" || exit 1
. ./h-manifest.conf
. ./common.sh
vearl_prepare_home

LOG="${CUSTOM_LOG_BASENAME}.log"
mkdir -p "$(dirname "$LOG")"
[[ -f "$VEARL_DATA/hive.env" ]] && . "$VEARL_DATA/hive.env"

{
  if [[ ! -f "$VEARL_DATA/wallet.json" || ! -f "$VEARL_DATA/miner.json" ]]; then
    vearl_log "Vearl is not set up on this rig yet: there is no wallet ($VEARL_DATA/wallet.json) or no settings."
    vearl_log "Open the Hive Shell of the rig, run  miner stop  then  /hive/miners/custom/vearl/setup.sh  (once), then  miner start."
    vearl_log "Details: README-HIVEOS.txt in the package."
    sleep 60
    exit 1
  fi
  vearl_log "starting vearl-miner ${CUSTOM_VERSION} (data folder $VEARL_DATA${CUDA_VISIBLE_DEVICES:+, GPUs $CUDA_VISIBLE_DEVICES})"
  cd "$VEARL_DATA" || exit 1
  ./vearl-miner --home "$VEARL_DATA" < /dev/null 2>&1
  code=$?
  vearl_log "vearl-miner stopped (exit code $code)"
  exit $code
} 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
