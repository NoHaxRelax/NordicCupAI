#!/bin/bash
# elias/replay_portal.sh PLAN.json TAG
# Serve elias/replay_endpoint.py (fixed boxes per frame, no camera moves, no GPU) through a Cloudflare quick
# tunnel and queue ONE validation run on the portal. Reachability is checked by resolving the tunnel name at a
# public resolver, because the laptop's own resolver cannot see a fresh trycloudflare name for minutes.
set -uo pipefail
PLAN=$1; TAG=$2
cd "$(dirname "$0")/.." || exit 1
VP="$HOME/venvs/nordic-drone/Scripts/python.exe"
CF="/c/Users/edlun/AppData/Local/Temp/claude/c--Users-edlun-Desktop-lucky-shots-NordicCupAI/bf48d8ae-2b16-4039-be3b-c570527b37ee/scratchpad/cloudflared.exe"
OUT=elias/out/portal/replay_$TAG; mkdir -p "$OUT"; PORT=${PORT:-9061}
DRONE_REPLAY="$PLAN" DRONE_PORT=$PORT "$VP" elias/replay_endpoint.py > "$OUT/api.log" 2>&1 & API=$!
"$CF" tunnel --url http://localhost:$PORT --no-autoupdate > "$OUT/tunnel.log" 2>&1 & TUN=$!
cleanup() { kill $API $TUN 2>/dev/null; taskkill //F //PID $API > /dev/null 2>&1; taskkill //F //IM cloudflared.exe > /dev/null 2>&1; }
trap cleanup EXIT
URL=""
for _ in $(seq 1 60); do
  URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$OUT/tunnel.log" | head -1)
  [ -n "$URL" ] && curl -sf -m 5 "http://localhost:$PORT/api" | grep -q drone-flyby-replay && grep -q "Registered tunnel connection" "$OUT/tunnel.log" && break
  sleep 3
done
[ -z "$URL" ] && { echo "no tunnel URL"; tail -5 "$OUT/tunnel.log"; exit 1; }
HOSTN=${URL#https://}
for _ in $(seq 1 40); do
  IP=$(nslookup -type=A "$HOSTN" 1.1.1.1 2>/dev/null | grep -A3 "^Name" | grep -oE "([0-9]{1,3}\.){3}[0-9]{1,3}" | head -1)
  if [ -n "$IP" ] && curl -sf -m 8 --resolve "$HOSTN:443:$IP" "$URL/api" | grep -q drone-flyby-replay; then echo "reachable: $URL ($IP)"; break; fi
  sleep 5
done
echo "$URL/predict" > "$OUT/url.txt"
"$VP" elias/portal.py validate "$URL/predict" 2>&1 | tee "$OUT/portal.txt"
curl -s -m 5 "http://localhost:$PORT/api"; echo
