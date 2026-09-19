#!/bin/bash
# elias/proxy_portal.sh TARGET_URL CLASSES TAG: one concealed portal run of a FOREIGN endpoint through the
# class-filtering proxy on this laptop (score x 13 = that class's AP for the target). The laptop must be quiet.
#   bash elias/proxy_portal.sh http://194.68.245.23:22090 helicopter OSCAR_helicopter
set -uo pipefail
TARGET=$1; CLASSES=$2; TAG=$3; PORT=${PORT:-9071}
cd "$(dirname "$0")/.." || exit 1
VP="$HOME/venvs/nordic-drone/Scripts/python.exe"; OUT=elias/out/portal/$TAG; mkdir -p "$OUT"
curl -sf -m 10 "$TARGET/" > /dev/null || { echo "target $TARGET does not answer"; exit 1; }
PROXY_TARGET=$TARGET PROXY_CLASSES=$CLASSES PROXY_PORT=$PORT "$VP" elias/class_proxy.py > "$OUT/proxy.log" 2>&1 &
PROXY_PID=$!
CF="/c/Users/edlun/AppData/Local/Temp/claude/c--Users-edlun-Desktop-lucky-shots-NordicCupAI/bf48d8ae-2b16-4039-be3b-c570527b37ee/scratchpad/cloudflared.exe"
[ -x "$CF" ] || CF=cloudflared
"$CF" tunnel --url http://localhost:$PORT --no-autoupdate > "$OUT/tunnel.log" 2>&1 &
TUN_PID=$!
trap 'kill $PROXY_PID $TUN_PID 2>/dev/null; taskkill //F //IM cloudflared.exe > /dev/null 2>&1' EXIT
for _ in $(seq 1 40); do
  URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$OUT/tunnel.log" | head -1)
  [ -n "$URL" ] && curl -sf -m 8 "http://localhost:$PORT/" > /dev/null && grep -q "Registered tunnel connection" "$OUT/tunnel.log" && break
  sleep 3
done
[ -z "${URL:-}" ] && { echo "no tunnel"; kill $PROXY_PID $TUN_PID 2>/dev/null; exit 1; }
for _ in $(seq 1 20); do curl -sf -m 8 "$URL/api" | grep -q "drone-flyby" && { echo "reachable from outside: $URL"; break; }; sleep 5; done
"$VP" elias/portal.py validate "$URL/predict" 2>&1 | tee "$OUT/portal.txt" | grep -E "RESULT|another|queue ->" | cut -c1-160
kill $PROXY_PID $TUN_PID 2>/dev/null; sleep 1
