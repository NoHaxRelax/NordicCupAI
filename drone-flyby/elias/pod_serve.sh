#!/bin/bash
# elias/pod_serve.sh WEIGHTS [ANSWER_WINDOWS]: serve the endpoint ON a GPU pod behind a Cloudflare quick tunnel.
# Two-model routing: DETECTOR=elias.ensemble:build ELIAS_WEIGHTS="a.pt,b.pt" ELIAS_ROUTE='{"ta-ta": 1}' (class -> model index).
# Prints "URL https://....trycloudflare.com" to /root/logs/serve.url once the tunnel is registered. Stop it with
#   pkill -f "[a]pi.py"; pkill -f "[c]loudflared"
set -uo pipefail
W=$1; WIN=${2:-}
cd /root/work/drone-flyby || exit 1
mkdir -p /root/logs/serve /root/logs/archive; rm -f /root/logs/serve.url
for f in /root/logs/serve/*.jsonl; do [ -f "$f" ] && mv "$f" "/root/logs/archive/$(date -u +%H%M%S)_$(basename "$f")"; done   # keep every run log
[ -x /root/cloudflared ] || { wget -q -O /root/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /root/cloudflared; }
export DRONE_DETECTOR="${DETECTOR:-ultralytics}" DRONE_WEIGHTS="$W" DRONE_AUTO_BAND="${AUTO_BAND:-1}" \
       ELIAS_WEIGHTS="${ELIAS_WEIGHTS:-}" ELIAS_IMGSZ="${ELIAS_IMGSZ:-1280}" ELIAS_ROUTE="${ELIAS_ROUTE:-{\}}" ELIAS_CONTEXT="${ELIAS_CONTEXT:-1.0}" ELIAS_CONF="${ELIAS_CONF:-0.05}" \
       DRONE_CUE_EVERY="${CUE_EVERY:-0}" DRONE_CUE_PX="${CUE_PX:-40}" DRONE_CUE_CONF="${CUE_CONF:-0.4}" DRONE_CUE_COOLDOWN="${CUE_COOLDOWN:-12}" DRONE_CUE_KIND="${CUE_KIND:-all}" DRONE_CUE_CLASSES="${CUE_CLASSES:-}" \
       DRONE_DEVICE=cuda:0 DRONE_IMGSZ=${IMGSZ:-1280} DRONE_PORT=9053 DRONE_LOG_DIR=/root/logs/serve \
       DRONE_OVERVIEW_BETWEEN_SIDES=0 DRONE_MISS_RULE=seen DRONE_CONF=0.05 DRONE_BIRTH_CONFIDENCE=0.25 DRONE_UPDATE_CONFIDENCE=0.15 \
       DRONE_ANSWER_WINDOWS="$WIN" DRONE_ANSWER_CLASSES="${ANSWER_CLASSES:-}" \
       DRONE_CLASS_EXTENT="${CLASS_EXTENT:-{\}}" DRONE_HEDGE_FACTOR="${HEDGE:-0}" DRONE_BOX_SCALE="${BOX_SCALE:-{\}}" \
       DRONE_CLUSTER_BIRTHS="${CLUSTER_BIRTHS:-0}" DRONE_L1_WAYPOINTS="${L1_WAYPOINTS:-4}" \
       DRONE_BOX_HEDGE="${BOX_HEDGE:-{\}}" DRONE_BOX_HEDGE_CONF="${BOX_HEDGE_CONF:-0.3}"
export -p | grep -E '^declare -x (DRONE_|ELIAS_)' > /root/logs/serve.env   # the watchdog restarts api.py with this
nohup python api.py > /root/logs/api.log 2>&1 &
if [ -n "${DIRECT_URL:-}" ]; then
  # the pod's own public TCP port (RunPod maps it straight to 9053): no tunnel, no Cloudflare in the path
  for _ in $(seq 1 60); do curl -sf -m 5 http://localhost:9053/ > /dev/null && break; sleep 3; done
  URL=$DIRECT_URL
else
  nohup /root/cloudflared tunnel --url http://localhost:9053 --no-autoupdate > /root/logs/tunnel.log 2>&1 &
  for _ in $(seq 1 60); do
    URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" /root/logs/tunnel.log | head -1)
    [ -n "$URL" ] && curl -sf -m 5 http://localhost:9053/ > /dev/null && grep -q "Registered tunnel connection" /root/logs/tunnel.log && break
    sleep 3
  done
fi
echo "URL $URL" > /root/logs/serve.url; cat /root/logs/serve.url
