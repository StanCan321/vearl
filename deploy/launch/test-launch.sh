#!/bin/bash
# Logic test of vearl-launch.sh: the real script is sourced, every effect on the world is replaced by a stub, and a made-up timeline is played
# (before the launch, at the launch with an empty chain, the chain starting, failures and their undoing, a restart in the middle).
#   bash deploy/launch/test-launch.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
export PRELAUNCH_CHECKS=0 STATE_DIR="$T/state" LIB="$T/lib" SNIP="$T/snip" SITE="$T/site" NOTIFY_TO="" LAUNCH_TS=1000 GENESIS_HASH=abc MIN_HEIGHT=3 FUND_AMOUNT=100000 RETRY_AFTER=300
mkdir -p "$LIB/faucet" "$SNIP" "$SITE"; echo live-api > "$LIB/nginx-vearl-api-live.conf"; echo live-gw > "$LIB/nginx-vearl-gw-live.conf"; echo page > "$LIB/faucet/index.html"
echo sample > "$SNIP/vearl-api.conf"; echo prelaunch > "$SNIP/vearl-gw.conf"
VEARL_LAUNCH_SOURCE_ONLY=1 . "$HERE/vearl-launch.sh"
sleep() { :; }     # the stages wait with sleep: no waiting in the test
command -v flock >/dev/null 2>&1 || flock() { return 0; }   # no flock in Git Bash: the lock is not what this test is about

# ---- the made-up world
NOW=0; HEIGHT=0; GEN=abc; NODE_UP=1; BAL=0; NGINX_OK=1; GW_OK=1; SEND_OK=1; FAUCET_UP=1; SITE_API_GEN=abc; SITE_GW_INFO='{"kinds":{}}'; SITE_FAUCET='faucet.js'
CALLS="$T/calls"; : > "$CALLS"
now() { echo "$NOW"; }
api_status() { [ "$NODE_UP" = 1 ] || return 1; echo "{\"genesis_hash\":\"$GEN\",\"height\":$HEIGHT}"; }
jget() { python -c 'import sys,json; d=json.load(sys.stdin); print(d.get(sys.argv[1],""))' "$1"; }
site_get() { case "$1" in /api/status) echo "{\"genesis_hash\":\"$SITE_API_GEN\"}";; /gw/info) echo "$SITE_GW_INFO";; /faucet/) echo "$SITE_FAUCET";; esac; }
nginx_set() { echo "nginx_set $1 $(cat "$2")" >> "$CALLS"; cp -a "$SNIP/$1" "$STATE_DIR/$1.bak"; [ "$NGINX_OK" = 1 ] || return 1; cp "$2" "$SNIP/$1"; }
nginx_restore() { echo "nginx_restore $1" >> "$CALLS"; cp -a "$STATE_DIR/$1.bak" "$SNIP/$1"; }
GW_STARTED=0
svc_active() { { [ "$1" = vearl-faucet ] && [ "$FAUCET_UP" = 1 ]; } || { [ "$1" = vearl-gateway ] && [ "$GW_STARTED" = 1 ]; }; }
svc_start() { echo "svc_start $1" >> "$CALLS"; [ "$1" = vearl-gateway ] && GW_STARTED=1; }
gw_info() { [ "$GW_OK" = 1 ] && echo '{"kinds":{}}'; }
wallet_address() { echo vrl1service; }
systemctl() { echo "systemctl $*" >> "$CALLS"; }
wallet_balance() { echo "$BAL"; }
wallet_send() { echo "wallet_send $1" >> "$CALLS"; [ "$SEND_OK" = 1 ] && echo "Transaction accepted by the node: 00" || echo "transaction refused: nonce"; }
install_page() { echo "install_page" >> "$CALLS"; }
notify() { echo "notify $1" >> "$CALLS"; }

PASS=0; FAILN=0
expect() { # $1 description, $2 expected calls (one per line, exact), the rest: nothing
  local got; got=$(cat "$CALLS"); : > "$CALLS"
  if [ "$got" = "$2" ]; then PASS=$((PASS + 1)); echo "  ok   $1"; else FAILN=$((FAILN + 1)); echo "  FAIL $1"; echo "       expected: $(echo "$2" | tr '\n' '|')"; echo "       got:      $(echo "$got" | tr '\n' '|')"; fi
}
run() { main >/dev/null 2>&1; }
snippet() { cat "$SNIP/$1"; }
eq() { if [ "$2" = "$3" ]; then PASS=$((PASS + 1)); echo "  ok   $1"; else FAILN=$((FAILN + 1)); echo "  FAIL $1 (expected '$3', got '$2')"; fi; }

echo "before the launch time"
NOW=500; HEIGHT=0; run; expect "nothing happens" ""
eq "sample API kept" "$(snippet vearl-api.conf)" sample

echo "launch time, chain still empty"
NOW=1000; HEIGHT=0; run
expect "only the API switches" "nginx_set vearl-api.conf live-api
notify api is on line"
eq "API is live" "$(snippet vearl-api.conf)" live-api
eq "gateway still closed" "$(snippet vearl-gw.conf)" prelaunch
NOW=1060; run; expect "next minute: nothing more" ""

echo "the chain starts (2 blocks: not yet)"
NOW=1120; HEIGHT=2; run; expect "waits for the chain" ""

echo "3 blocks: the service wallet is paid, nothing else yet"
NOW=1180; HEIGHT=3; BAL=0; run; expect "transfer sent, nothing else opened" "wallet_send vrl1service"
NOW=1240; run; expect "transfer not sent twice while it is awaited" ""
NOW=1300; BAL=100000; run
expect "wallet funded: gateway, then faucet" "notify fund is on line
svc_start vearl-gateway
nginx_set vearl-gw.conf live-gw
notify gateway is on line
install_page
notify faucet is on line
systemctl disable --now vearl-demo-a
systemctl disable --now vearl-demo-b
systemctl disable --now vearl-demo-c
systemctl disable --now vearl-gateway-demo
notify everything is on line"
eq "gateway is live" "$(snippet vearl-gw.conf)" live-gw
NOW=1360; run; expect "after everything: silence" ""
status >/dev/null && echo "  ok   status runs"

echo "failures are undone"
rm -rf "$STATE_DIR"; GW_STARTED=0; echo sample > "$SNIP/vearl-api.conf"; echo prelaunch > "$SNIP/vearl-gw.conf"
NOW=2000; HEIGHT=0; NGINX_OK=0; run
expect "nginx refuses the snippet: nothing stays" "nginx_set vearl-api.conf live-api"
eq "API snippet unchanged" "$(snippet vearl-api.conf)" sample
NGINX_OK=1; SITE_API_GEN=wrong; NOW=2060; run
expect "the page answers another network: undone" "nginx_set vearl-api.conf live-api
nginx_restore vearl-api.conf"
eq "API back to sample" "$(snippet vearl-api.conf)" sample
NOW=2120; run; expect "third failure raises an alarm" "nginx_set vearl-api.conf live-api
nginx_restore vearl-api.conf
notify stage api keeps failing"
NOW=2150; run; expect "then it waits before trying again" ""
SITE_API_GEN=abc; NOW=2500; run; expect "retry after the pause works" "nginx_set vearl-api.conf live-api
notify api is on line"

echo "a refused transfer, a dead gateway, a missing faucet service"
HEIGHT=5; SEND_OK=0; BAL=0; NOW=2560; run; expect "transfer refused: counted as a failure, nothing opened" "wallet_send vrl1service"
SEND_OK=1; NOW=2620; run; expect "transfer retried and accepted" "wallet_send vrl1service"
BAL=100000; GW_OK=0; NOW=2680; run
expect "gateway does not answer: gateway page stays closed, faucet still opens" "notify fund is on line
svc_start vearl-gateway
install_page
notify faucet is on line"
eq "gateway snippet untouched" "$(snippet vearl-gw.conf)" prelaunch
GW_OK=1; SITE_GW_INFO='{"prelaunch":true,"kinds":{}}'; NOW=3200; run
expect "the page still answers the pre-launch info: undone" "nginx_set vearl-gw.conf live-gw
nginx_restore vearl-gw.conf"
SITE_GW_INFO='{"kinds":{}}'; NOW=3600; run
expect "then the gateway opens and all is on line" "nginx_set vearl-gw.conf live-gw
notify gateway is on line
systemctl disable --now vearl-demo-a
systemctl disable --now vearl-demo-b
systemctl disable --now vearl-demo-c
systemctl disable --now vearl-gateway-demo
notify everything is on line"

echo "a wrong network is never switched, a node that is down is waited for"
rm -rf "$STATE_DIR"; GW_STARTED=0; echo sample > "$SNIP/vearl-api.conf"; GEN=zzz; NOW=4000; run; expect "wrong genesis: nothing done, alarm" "notify wrong genesis"
eq "API untouched" "$(snippet vearl-api.conf)" sample
GEN=abc; NODE_UP=0; run; expect "node down: nothing done" ""
NODE_UP=1

echo "a restart in the middle: the state on disk is enough"
rm -rf "$STATE_DIR"; GW_STARTED=0; echo sample > "$SNIP/vearl-api.conf"; echo prelaunch > "$SNIP/vearl-gw.conf"
NOW=5000; HEIGHT=4; BAL=100000; run >/dev/null
rm -f "$STATE_DIR/gateway.done" "$STATE_DIR/live"; : > "$CALLS"; NOW=5060; run
expect "a stage that was lost is redone, the others are not" "nginx_set vearl-gw.conf live-gw
notify gateway is on line
notify everything is on line"

echo "the check an hour before the launch"
rm -rf "$STATE_DIR"; PRELAUNCH_CHECKS=1; check() { echo "ALL GOOD"; }
NOW=$((LAUNCH_TS - 7200)); run; expect "two hours before: silent" ""
NOW=$((LAUNCH_TS - 3500)); run; expect "an hour before: the check is e-mailed once" "notify check one hour before the launch"
NOW=$((LAUNCH_TS - 3400)); run; expect "and not again" ""
PRELAUNCH_CHECKS=0

echo; echo "$PASS passed, $FAILN failed"
[ "$FAILN" = 0 ]
