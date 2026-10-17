#!/usr/bin/env bash
# HiveOS sources this script every minute or so and reads two variables:
#   khs    total hashrate in kH/s          stats   JSON with one entry per GPU
# The miner's own status line (every 10 s) gives the TOTAL speed in millions of tiles per second, not one figure per GPU, so the total
# is shared equally between the GPUs that mine: the total is right, the split of a rig with different cards is not.
_here="$(dirname "${BASH_SOURCE[0]}")"
. "$_here/h-manifest.conf"
. "$_here/common.sh"

khs=0
stats=""
_pid="$(pgrep -f 'vearl-miner --home' | head -n 1)"
if [[ -n "$_pid" ]]; then
  _line="$(tail -n 300 "$VEARL_DATA/miner.log" 2>/dev/null | grep -F '[miner]' | tail -n 1)"
  _mtiles="$(sed -n 's/.*(\([0-9.]*\) M tiles\/s).*/\1/p' <<< "$_line")"
  _blocks="$(sed -n 's/.*blocks found=\([0-9]*\).*/\1/p' <<< "$_line")"
  _mtiles="${_mtiles:-0}"; _blocks="${_blocks:-0}"
  _up="$(ps -o etimes= -p "$_pid" 2>/dev/null | tr -d ' ')"; _up="${_up:-0}"

  # the GPUs: bus number, temperature, fan, and whether they mine (compute capability 8.0 or newer)
  _bus=(); _temp=(); _fan=(); _mine=()
  if command -v nvidia-smi >/dev/null 2>&1; then
    _visible=",${CUDA_VISIBLE_DEVICES:-},"
    while IFS=',' read -r _idx _busid _t _f _cc; do
      _idx="${_idx// /}"; _busid="${_busid// /}"; _t="${_t// /}"; _f="${_f// /}"; _cc="${_cc// /}"
      [[ "$_idx" =~ ^[0-9]+$ ]] || continue
      _b="${_busid#*:}"; _b="${_b%%:*}"
      _bus+=("$((16#${_b:-0}))")
      [[ "$_t" =~ ^[0-9]+$ ]] || _t=0
      [[ "$_f" =~ ^[0-9]+$ ]] || _f=0
      _temp+=("$_t"); _fan+=("$_f")
      _ok=1
      [[ -n "${_cc%%.*}" && "${_cc%%.*}" =~ ^[0-9]+$ && "${_cc%%.*}" -lt 8 ]] && _ok=0
      [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$_visible" != *",$_idx,"* ]] && _ok=0
      _mine+=("$_ok")
    done < <(nvidia-smi --query-gpu=index,pci.bus_id,temperature.gpu,fan.speed,compute_cap --format=csv,noheader,nounits 2>/dev/null)
  fi
  _n=0; for _m in "${_mine[@]}"; do [[ "$_m" == 1 ]] && _n=$((_n + 1)); done
  _hs=()
  for _m in "${_mine[@]}"; do
    if [[ "$_m" == 1 && $_n -gt 0 ]]; then _hs+=("$(awk -v t="$_mtiles" -v n="$_n" 'BEGIN { printf "%.4f", t / n }')"); else _hs+=("0"); fi
  done
  khs="$(awk -v t="$_mtiles" 'BEGIN { printf "%.2f", t * 1000 }')"
  _j() { local IFS=','; echo "[$*]"; }
  stats="$(printf '{"hs":%s,"hs_units":"mhs","temp":%s,"fan":%s,"uptime":%s,"ver":"%s","ar":[%s,0],"algo":"vearl","bus_numbers":%s}' \
    "$(_j "${_hs[@]}")" "$(_j "${_temp[@]}")" "$(_j "${_fan[@]}")" "$_up" "$CUSTOM_VERSION" "$_blocks" "$(_j "${_bus[@]}")")"
fi
