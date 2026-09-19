#!/bin/bash
# elias/serve_portal.sh WEIGHTS TAG MODE [ANSWER_WINDOWS]
#   MODE rehearse : serve + quick tunnel, then the LOCAL evaluator in --realtime through the tunnel (no portal)
#   MODE validate : serve + quick tunnel, then queue ONE portal validation run and wait for its score
# ANSWER_WINDOWS e.g. "0:125" conceals the rest of the run (example.py, DRONE_ANSWER_WINDOWS).
set -uo pipefail
W=$1; TAG=$2; MODE=$3; WIN=${4:-}
cd "$(dirname "$0")/.." || exit 1
VP="$HOME/venvs/nordic-drone/Scripts/python.exe"
CF="/c/Users/edlun/AppData/Local/Temp/claude/c--Users-edlun-Desktop-lucky-shots-NordicCupAI/bf48d8ae-2b16-4039-be3b-c570527b37ee/scratchpad/cloudflared.exe"
OUT=elias/out/portal/$TAG; mkdir -p "$OUT"; PORT=${PORT:-9053}
export DRONE_DETECTOR=ultralytics DRONE_WEIGHTS="$W" DRONE_DEVICE=cuda:0 DRONE_IMGSZ=${IMGSZ:-960} DRONE_PORT=$PORT DRONE_LOG_DIR="$OUT" \
       DRONE_OVERVIEW_BETWEEN_SIDES=${OVERVIEW:-0} DRONE_MISS_RULE=seen DRONE_CONF=${CONF:-0.05} \
       DRONE_BIRTH_CONFIDENCE=${BIRTH:-0.25} DRONE_UPDATE_CONFIDENCE=${UPDATE:-0.15} DRONE_ANSWER_WINDOWS="$WIN" DRONE_CV_THREADS=6
"$VP" api.py > "$OUT/api.log" 2>&1 & API=$!
"$CF" tunnel --url http://localhost:$PORT --no-autoupdate > "$OUT/tunnel.log" 2>&1 & TUN=$!
cleanup() { kill $API $TUN 2>/dev/null; taskkill //F //PID $API > /dev/null 2>&1; taskkill //F //IM cloudflared.exe > /dev/null 2>&1; }
trap cleanup EXIT
URL=""
for _ in $(seq 1 60); do
  URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$OUT/tunnel.log" | head -1)
  # the local resolver often cannot see a fresh trycloudflare name for minutes; the portal and the pods can
  [ -n "$URL" ] && curl -sf -m 5 "http://localhost:$PORT/" > /dev/null && grep -q "Registered tunnel connection" "$OUT/tunnel.log" && break
  sleep 3
done
[ -z "$URL" ] && { echo "no tunnel URL"; tail -5 "$OUT/tunnel.log"; exit 1; }
# Ready means reachable from OUTSIDE (a fresh quick tunnel answers 530 for a while) and warm (the first
# inference and the SIFT calibration are slow). The laptop's own resolver cannot see a fresh trycloudflare name
# for minutes, so resolve it at a public resolver and connect by address.
HOSTN=${URL#https://}
for _ in $(seq 1 40); do
  IP=$(nslookup -type=A "$HOSTN" 1.1.1.1 2>/dev/null | grep -A3 "^Name" | grep -oE "([0-9]{1,3}\.){3}[0-9]{1,3}" | head -1)
  if [ -n "$IP" ] && curl -sf -m 8 --resolve "$HOSTN:443:$IP" "$URL/api" | grep -q drone-flyby-usecase; then echo "reachable from outside: $URL ($IP)"; break; fi
  sleep 5
done
"$VP" - "$PORT" <<'PYW'
import sys, json, base64, time, requests, numpy as np, cv2
port = sys.argv[1]; ok, png = cv2.imencode('.png', (np.random.rand(540, 960, 3)*255).astype(np.uint8))
for i in range(3):
    body = {'sequence_id': 'warmup', 'frame': i, 'frame_index': i, 'request_id': f'warmup:{i}', 'frame_interval_ms': 333, 'response_timeout_ms': 3333,
            'original_width': 3840, 'original_height': 2160, 'camera_command_feedback': None,
            'view': {'resolution_level': 0, 'center_x': 1920, 'center_y': 1080, 'view_id': f'w{i}', 'image': base64.b64encode(png.tobytes()).decode(),
                     'image_media_type': 'image/png', 'width': 960, 'height': 540, 'source_region_xyxy': [0, 0, 3840, 2160]},
            'camera_constraints': {'maximum_center_delta': 2203.0, 'allowed_resolution_levels': [0, 1], 'full_view_reset_exempt_from_delta': True,
                                   'center_bounds': [{'resolution_level': 0, 'width': 960, 'height': 540, 'minimum_center_x': 1920, 'maximum_center_x': 1920, 'minimum_center_y': 1080, 'maximum_center_y': 1080},
                                                     {'resolution_level': 1, 'width': 960, 'height': 540, 'minimum_center_x': 960, 'maximum_center_x': 2880, 'minimum_center_y': 540, 'maximum_center_y': 1620}]}}
    t = time.time(); r = requests.post(f'http://localhost:{port}/predict', json=body, timeout=30); print(f'warm-up {i}: {r.status_code} {1000*(time.time()-t):.0f} ms')
PYW
echo "serving $W at $URL/predict (windows '${WIN:-all}')"
if [ "$MODE" = rehearse ]; then
  # the organisers' evaluator, real-time clock, run FROM a pod: a true external round trip through the tunnel
  POD="ssh -o ConnectTimeout=25 -i $HOME/.ssh/id_ed25519 -p ${POD_PORT:-42960} root@${POD_HOST:-149.36.0.173}"
  timeout 600 $POD "cd /root/work/drone-flyby && pip install -q --break-system-packages --root-user-action=ignore faster-coco-eval requests > /dev/null 2>&1; python local_evaluator.py --url $URL/predict --scene helsinki --realtime 2>&1 | tail -28" > "$OUT/rehearse.txt"
  grep -E "frames skipped|frames unanswered|responses accepted|timeouts|round trip|COCO mAP" "$OUT/rehearse.txt"
else
  "$VP" elias/portal.py validate "$URL/predict" 2>&1 | tee "$OUT/portal.txt" | tail -6
fi
"$VP" - "$OUT" <<'PY'
import json, glob, sys
rows = [json.loads(l) for f in glob.glob(sys.argv[1]+'/*.jsonl') for l in open(f, encoding='utf-8') if l.strip()]
if rows:
    idx = sorted(r['frame_index'] for r in rows if 'frame_index' in r); ms = sorted(r.get('total_ms', 0) for r in rows)
    print(f"server saw {len(idx)} frames (index {idx[0]}..{idx[-1]}, {idx[-1]-idx[0]+1-len(idx)} gaps), server ms median {ms[len(ms)//2]:.0f} p95 {ms[int(len(ms)*.95)]:.0f}, emitted {sum(1 for r in rows if r.get('emitted', True))}")
PY
