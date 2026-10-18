#!/usr/bin/env bash
# HiveOS runs this before every start of the miner, with the flight sheet in the environment:
#   CUSTOM_TEMPLATE    "Wallet and worker template": your Vearl address (vrl1...), for example %WAL% (or %WAL%.%WORKER_NAME% to name the worker).
#                      POOL mode mines to THIS address; solo mode only CHECKS it against the wallet file (it mines to the wallet file)
#   CUSTOM_URL         "Pool URL": the pool to join, for example pool.vearl.org:3333 (the official pool, also used when this field is
#                      empty and the rig was never set up for solo). Write  solo  to mine alone with the node of the rig instead.
#   CUSTOM_PASS        optional: the wallet password, only if you did not run setup.sh (see README-HIVEOS.txt; the safer way is setup.sh)
#   CUSTOM_USER_CONFIG "Extra config arguments": optional words key=value, separated by spaces or new lines:
#                        max_jobs=N         paid jobs computed at the same time (default 1)
#                        partner_share=N    % of the time for Quantus, only with the Quantus add-on (default 80)
#                        partner_cycle=N    seconds of one Quantus/Vearl cycle (default 60)
#                        quantus_address=qz...   turns the Quantus add-on on (it must be installed in the package folder)
#                        gpus=0,1,3         use only these GPUs (CUDA_VISIBLE_DEVICES: applies to Vearl and Quantus)
#                        mode=pool|solo     the way of mining (overrides the Pool URL field)
#                        pool=HOST:PORT     the pool (overrides the Pool URL field)
#                        worker=NAME        the worker name shown by the pool (default: the Hive rig name)
#   It writes $VEARL_DATA/hive.env (read by h-run.sh) and keeps $VEARL_DATA/miner.json in line with it.
cd "$(dirname "$0")" || exit 1
. ./h-manifest.conf
. ./common.sh
vearl_prepare_home

ENVF="$VEARL_DATA/hive.env"
MINERJSON="$VEARL_DATA/miner.json"
declare -A OPT
for tok in $CUSTOM_USER_CONFIG; do
  [[ "$tok" == *=* ]] || continue
  k="${tok%%=*}"; v="${tok#*=}"
  case "$k" in
    max_jobs|partner_share|partner_cycle) [[ "$v" =~ ^[0-9]{1,4}$ ]] && OPT[$k]="$v" || vearl_log "ignored $tok (a number is expected)" ;;
    quantus_address) [[ "$v" =~ ^qz[A-Za-z0-9]{38,58}$ ]] && OPT[$k]="$v" || vearl_log "ignored quantus_address (it must look like qz...)" ;;
    gpus) [[ "$v" =~ ^[0-9]+(,[0-9]+)*$ ]] && OPT[$k]="$v" || vearl_log "ignored $tok (indices separated by commas, for example 0,1,3)" ;;
    mode) [[ "$v" == pool || "$v" == solo ]] && OPT[$k]="$v" || vearl_log "ignored $tok (pool or solo)" ;;
    pool) [[ "$v" =~ ^[A-Za-z0-9.:/+_-]{3,100}$ ]] && OPT[$k]="$v" || vearl_log "ignored $tok (HOST:PORT expected)" ;;
    worker) [[ "$v" =~ ^[A-Za-z0-9._-]{1,32}$ ]] && OPT[$k]="$v" || vearl_log "ignored $tok (1-32 letters, digits, . _ -)" ;;
    *) vearl_log "ignored unknown option $k" ;;
  esac
done

# the way of mining: the extra config, else the Pool URL field ("solo" = alone; anything else = that pool; empty = the saved choice of the rig,
# and for a rig that was never set up the official pool, which needs no setup at all)
MODE=""; POOL=""
URL="${CUSTOM_URL:-}"; URL="${URL//[[:space:]]/}"
if [[ -n "${OPT[mode]:-}" ]]; then
  MODE="${OPT[mode]}"
elif [[ "${URL,,}" == solo ]]; then
  MODE=solo
elif [[ -n "$URL" ]]; then
  MODE=pool
fi
if [[ -n "${OPT[pool]:-}" ]]; then
  POOL="${OPT[pool]}"
  [[ -z "$MODE" ]] && MODE=pool
fi
[[ "$MODE" == pool && -z "$POOL" && -n "$URL" && "${URL,,}" != solo ]] && POOL="$URL"
[[ -z "$MODE" && ! -f "$MINERJSON" ]] && MODE=pool   # never set up: the pool needs no wallet file

# the payout address and the worker name of the pool: the flight sheet template (vrl1...  or  vrl1....workername), else the wallet file
SHEET_FULL="$(grep -o -E 'vrlq?1[a-z0-9]{20,}(\.[A-Za-z0-9_.-]+)?' <<< "${CUSTOM_TEMPLATE:-}" | head -n 1)"
ADDR="${SHEET_FULL%%.*}"
[[ -z "$ADDR" ]] && ADDR="$(vearl_wallet_address)"
WORKER="${OPT[worker]:-}"
[[ -z "$WORKER" && "$SHEET_FULL" == *.* ]] && WORKER="${SHEET_FULL#*.}"
[[ -z "$WORKER" ]] && WORKER="${WORKER_NAME:-}"
WORKER="$(printf '%s' "$WORKER" | tr -c 'A-Za-z0-9._-' '-' | cut -c1-32)"
[[ -z "$WORKER" ]] && WORKER="$(hostname 2>/dev/null | tr -c 'A-Za-z0-9._-' '-' | cut -c1-32)"
[[ -z "$WORKER" ]] && WORKER=rig

{
  echo "# written by h-config.sh from the flight sheet ($(date -u '+%Y-%m-%d %H:%M:%S') UTC): change the flight sheet, not this file"
  [[ -n "${OPT[gpus]:-}" ]] && echo "export CUDA_VISIBLE_DEVICES=${OPT[gpus]}"
  [[ -n "$MODE" ]] && echo "export VEARL_MODE=$MODE"
  [[ -n "$POOL" ]] && echo "export VEARL_POOL='$POOL'"
  [[ -n "$ADDR" ]] && echo "export VEARL_ADDRESS=$ADDR"
  echo "export VEARL_WORKER='$WORKER'"
} > "$ENVF"
chmod 600 "$ENVF" 2>/dev/null || true

# solo mode: the address of the flight sheet must be the one of the wallet file of the rig
WALLET_ADDR="$(vearl_wallet_address)"
SHEET_ADDR="${SHEET_FULL%%.*}"
if [[ "$MODE" != pool && -n "$SHEET_ADDR" && -n "$WALLET_ADDR" && "$SHEET_ADDR" != "$WALLET_ADDR" ]]; then
  vearl_log "WARNING: the flight sheet says $SHEET_ADDR but the wallet file of this rig is $WALLET_ADDR: solo mining goes to the wallet file"
fi
if [[ "$MODE" == pool ]]; then
  vearl_log "pool mode: ${POOL:-the official pool}, payout address ${ADDR:-(none yet)}, worker $WORKER"
  [[ -z "$ADDR" ]] && vearl_log "ERROR: no Vearl address: put it (vrl1...) in the Wallet and worker template of the flight sheet, for example %WAL%"
fi

# a password given in the flight sheet is stored for the miner (only when there is none yet): setup.sh is the safer way
if [[ -f "$VEARL_DATA/wallet.json" && ! -f "$VEARL_DATA/wallet.pw" && -n "$CUSTOM_PASS" ]]; then
  ( umask 077; printf '%s' "$CUSTOM_PASS" > "$VEARL_DATA/wallet.pw" )
  vearl_log "the wallet password of the flight sheet was saved for the miner (wallet.pw, readable by root only)"
fi

# settings of the miner: created when there is a wallet and a password, otherwise setup.sh does it
if [[ -f "$VEARL_DATA/wallet.json" && -f "$VEARL_DATA/wallet.pw" && ! -f "$MINERJSON" ]]; then
  cat > "$MINERJSON" <<EOF
{
  "version": 1,
  "quantus_address": "${OPT[quantus_address]:-}",
  "vearl_wallet": "wallet.json",
  "vearl_address": "$WALLET_ADDR",
  "account": 0,
  "partner_share": ${OPT[partner_share]:-80},
  "partner_cycle": ${OPT[partner_cycle]:-60},
  "worker": true,
  "max_jobs": ${OPT[max_jobs]:-1},
  "remember_password": true
}
EOF
  chmod 600 "$MINERJSON" 2>/dev/null || true
  vearl_log "settings created ($MINERJSON)"
elif [[ -f "$MINERJSON" ]] && command -v jq >/dev/null 2>&1; then
  tmp="$MINERJSON.tmp$$"
  cp "$MINERJSON" "$tmp"
  for k in max_jobs partner_share partner_cycle; do
    [[ -n "${OPT[$k]:-}" ]] && jq --argjson v "${OPT[$k]}" ".$k = \$v" "$tmp" > "$tmp.2" && mv "$tmp.2" "$tmp"
  done
  [[ -n "${OPT[quantus_address]:-}" ]] && jq --arg v "${OPT[quantus_address]}" '.quantus_address = $v' "$tmp" > "$tmp.2" && mv "$tmp.2" "$tmp"
  mv "$tmp" "$MINERJSON"; chmod 600 "$MINERJSON" 2>/dev/null || true
fi
exit 0
