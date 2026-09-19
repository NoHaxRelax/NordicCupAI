#!/bin/bash
# elias/pod_portal.sh WEIGHTS_ON_POD TAG ANSWER_WINDOWS: one concealed portal validation served FROM the GPU pod.
# Run on the laptop: the team key never leaves it (elias/portal.py queues the run from here).
set -uo pipefail
W=$1; TAG=$2; WIN=${3:-}
cd "$(dirname "$0")/.." || exit 1
VP="$HOME/venvs/nordic-drone/Scripts/python.exe"; OUT=elias/out/portal/$TAG; mkdir -p "$OUT"
POD="ssh -o ConnectTimeout=25 -i $HOME/.ssh/id_ed25519 -p ${POD_PORT:-42960} root@${POD_HOST:-149.36.0.173}"
PROBE="$POD"   # the pod curling its own tunnel URL goes out through Cloudflare, which is an outside check
timeout 60 $POD 'pkill -f "[a]pi.py"; pkill -f "[c]loudflared tunnel"; sleep 2; true'
timeout 300 $POD "cd /root/work/drone-flyby && IMGSZ=${IMGSZ:-1280} ANSWER_CLASSES='${ANSWER_CLASSES:-}' bash elias/pod_serve.sh $W '$WIN' > /root/logs/serve_start.log 2>&1; cat /root/logs/serve.url"
URL=$(timeout 30 $POD 'cat /root/logs/serve.url' | awk '{print $2}')
[ -z "$URL" ] && { echo "no URL"; exit 1; }
for _ in $(seq 1 40); do timeout 30 $PROBE "curl -sf -m 8 $URL/ > /dev/null" && { echo "reachable from outside: $URL"; break; }; sleep 5; done
timeout 120 $POD 'cd /root/work/drone-flyby && python - <<PY
import base64, time, requests, numpy as np, cv2
ok, png = cv2.imencode(".png", np.full((540, 960, 3), 90, np.uint8))
for i in range(3):
    body = {"sequence_id": "warmup", "frame": i, "frame_index": i, "request_id": f"warmup:{i}", "frame_interval_ms": 333, "response_timeout_ms": 3333,
            "original_width": 3840, "original_height": 2160, "camera_command_feedback": None,
            "view": {"resolution_level": 0, "center_x": 1920, "center_y": 1080, "view_id": f"w{i}", "image": base64.b64encode(png.tobytes()).decode(),
                     "image_media_type": "image/png", "width": 960, "height": 540, "source_region_xyxy": [0, 0, 3840, 2160]},
            "camera_constraints": {"maximum_center_delta": 2203.0, "allowed_resolution_levels": [0, 1], "full_view_reset_exempt_from_delta": True,
                                   "center_bounds": [{"resolution_level": 0, "width": 960, "height": 540, "minimum_center_x": 1920, "maximum_center_x": 1920, "minimum_center_y": 1080, "maximum_center_y": 1080},
                                                     {"resolution_level": 1, "width": 960, "height": 540, "minimum_center_x": 960, "maximum_center_x": 2880, "minimum_center_y": 540, "maximum_center_y": 1620}]}}
    t = time.time(); r = requests.post("http://localhost:9053/predict", json=body, timeout=30); print("warm-up", i, r.status_code, round(1000*(time.time()-t)), "ms")
PY'
"$VP" elias/portal.py validate "$URL/predict" 2>&1 | tee "$OUT/portal.txt" | grep -E "RESULT|another|queue ->" | cut -c1-160
timeout 60 $POD 'python - <<PY
import json, glob
rows = [json.loads(l) for f in glob.glob("/root/logs/serve/*.jsonl") for l in open(f) if l.strip()]
rows = [r for r in rows if "frame_index" in r and r.get("status") != None]
idx = sorted(r["frame_index"] for r in rows); ms = sorted(r.get("total_ms", 0) for r in rows)
print(f"server saw {len(idx)} frames, {max(idx)-min(idx)+1-len(set(idx))} gaps, server ms median {ms[len(ms)//2]:.0f} p95 {ms[int(len(ms)*.95)]:.0f}, emitted {sum(1 for r in rows if r.get(\"emitted\", True))}")
PY' | tee "$OUT/server.txt"
