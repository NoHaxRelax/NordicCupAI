#!/bin/bash
# elias/pod_serve.sh WEIGHTS [ANSWER_WINDOWS]: serve the endpoint ON a GPU pod behind a Cloudflare quick tunnel.
# Prints "URL https://....trycloudflare.com" to /root/logs/serve.url once the tunnel is registered. Stop it with
#   pkill -f "[a]pi.py"; pkill -f "[c]loudflared"
set -uo pipefail
W=$1; WIN=${2:-}
cd /root/work/drone-flyby || exit 1
mkdir -p /root/logs/serve; rm -f /root/logs/serve.url /root/logs/serve/*.jsonl
[ -x /root/cloudflared ] || { wget -q -O /root/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /root/cloudflared; }
export DRONE_DETECTOR=ultralytics DRONE_WEIGHTS="$W" DRONE_DEVICE=cuda:0 DRONE_IMGSZ=${IMGSZ:-1280} DRONE_PORT=9053 DRONE_LOG_DIR=/root/logs/serve \
       DRONE_OVERVIEW_BETWEEN_SIDES=0 DRONE_MISS_RULE=seen DRONE_CONF=0.05 DRONE_BIRTH_CONFIDENCE=0.25 DRONE_UPDATE_CONFIDENCE=0.15 \
       DRONE_ANSWER_WINDOWS="$WIN" DRONE_ANSWER_CLASSES="${ANSWER_CLASSES:-}" \n       DRONE_CLASS_EXTENT="${CLASS_EXTENT:-{\}}" DRONE_HEDGE_FACTOR="${HEDGE:-0}" DRONE_BOX_SCALE="${BOX_SCALE:-{\}}"
nohup python api.py > /root/logs/api.log 2>&1 &
nohup /root/cloudflared tunnel --url http://localhost:9053 --no-autoupdate > /root/logs/tunnel.log 2>&1 &
for _ in $(seq 1 60); do
  URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" /root/logs/tunnel.log | head -1)
  [ -n "$URL" ] && curl -sf -m 5 http://localhost:9053/ > /dev/null && grep -q "Registered tunnel connection" /root/logs/tunnel.log && break
  sleep 3
done
echo "URL $URL" > /root/logs/serve.url; cat /root/logs/serve.url
