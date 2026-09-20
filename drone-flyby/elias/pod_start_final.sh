#!/bin/bash
# elias/pod_start_final.sh HOST SSH_PORT PUBLIC_PORT [routed|deploy|robust|robust_hn]: start the endpoint FOR THE EVALUATION ATTEMPT on a
# prepared pod (pod_serve_setup.sh done, both checkpoints in /root/out/). No answer window, no class filter, the pod's
# own public port, the watchdog on. It queues nothing: only a human queues the attempt, in the browser.
#   routed = F3 for every class, F5 for ta-ta (elias.ensemble), cluster births, launcher box 0.85
#   deploy = F3 alone with the same two switches
#   robust = routed + F3HN (replay fine-tune with unseen-terrain negatives, /root/out/F3HN_m1280.pt) for large_tower,
#            small_launcher MERGED over the three models, and the box hedge (research/07-committee.md). Laptop
#            instruments: harness 0.696 against 0.686, exact-label scene 0.615 against 0.584, false answers on empty
#            terrain 4.6 against 14.4 per frame. NEEDS the morning's portal confirmation before it replaces routed.
#   robust_hn = the same with small_launcher answered by F3HN alone (scene 0.605, false answers 1.6 per frame)
set -uo pipefail
H=$1; P=$2; PUB=$3; MODE=${4:-routed}
cd "$(dirname "$0")/.." || exit 1
POD="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o LogLevel=ERROR -i $HOME/.ssh/id_ed25519 -p $P root@$H"
CE='{"medium_launcher":"detector","ta-ta":"detector"}'; BOX='{"medium_launcher": 0.85}'
ROUTE='{"small_launcher": 0, "medium_launcher": 0, "large_launcher": 0, "small_plane": 0, "medium_plane": 0, "jet_plane": 0, "small_tower": 0, "large_tower": 0, "tank": 0, "mine_roller": 0, "hangar": 0, "helicopter": 0, "ta-ta": 1, "condor": 0, "jammer": 0, "spacecraft": 0}'
DET=elias.ensemble:build; [ "$MODE" = deploy ] && DET=ultralytics
WEIGHTS='/root/out/F3_both_m1280.last.pt,/root/out/F5_fixed_m1280.last.pt'; HEDGE_BOX='{}'
if [ "$MODE" = robust ] || [ "$MODE" = robust_hn ]; then
  WEIGHTS="$WEIGHTS,/root/out/F3HN_m1280.pt"
  HEDGE_BOX='{"medium_launcher": [1.176, 0.82], "large_launcher": [0.88], "ta-ta": [0.8, 1.25], "large_tower": [0.87, 1.15]}'
  # every class listed except small_launcher in robust: an unlisted class is merged over all models
  ROUTE='{"medium_launcher": 0, "large_launcher": 0, "small_plane": 0, "medium_plane": 0, "jet_plane": 0, "small_tower": 0, "large_tower": 2, "tank": 0, "mine_roller": 0, "hangar": 0, "helicopter": 0, "ta-ta": 1, "condor": 0, "jammer": 0, "spacecraft": 0}'
  [ "$MODE" = robust_hn ] && ROUTE="${ROUTE%\}}, \"small_launcher\": 2}"
fi
timeout 60 $POD 'pkill -f "[w]atchdog.py"; pkill -f "[c]lass_proxy.py"; pkill -f "[a]pi.py"; pkill -f "[c]loudflared tunnel"; sleep 2; true'
timeout 300 $POD "cd /root/work/drone-flyby && IMGSZ=1280 ANSWER_CLASSES='' CLASS_EXTENT='$CE' HEDGE=0 BOX_SCALE='$BOX' CLUSTER_BIRTHS=1 L1_WAYPOINTS=4 AUTO_BAND=1 DETECTOR='$DET' ELIAS_WEIGHTS='$WEIGHTS' ELIAS_ROUTE='$ROUTE' BOX_HEDGE='$HEDGE_BOX' ELIAS_CONTEXT=1.0 ELIAS_CONF=0.05 DIRECT_URL='http://$H:$PUB' bash elias/pod_serve.sh /root/out/F3_both_m1280.last.pt '' > /root/logs/serve_start.log 2>&1; cat /root/logs/serve.url"
for _ in $(seq 1 30); do curl -sf -m 8 "http://$H:$PUB/api" | grep -q drone-flyby && break; sleep 4; done
curl -sf -m 8 "http://$H:$PUB/api" | grep -q drone-flyby || { echo "NOT REACHABLE at http://$H:$PUB (RunPod may have remapped the port: check get-pod)"; exit 1; }
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
echo "--- environment the server (and the watchdog's restart) uses:"
timeout 30 $POD 'grep -E "DRONE_ANSWER|DRONE_DETECTOR|DRONE_CLUSTER|DRONE_BOX|DRONE_CLASS_EXTENT|ELIAS_WEIGHTS|ELIAS_ROUTE" /root/logs/serve.env | cut -c1-200'
if timeout 30 $POD 'grep -E "DRONE_ANSWER_(WINDOWS|CLASSES)=\"[^\"]" /root/logs/serve.env'; then echo "STOP: an answer window or class filter is set; the attempt would be answered only in part"; exit 1; fi
timeout 30 $POD 'cd /root/work/drone-flyby && setsid nohup python elias/watchdog.py > /root/logs/watchdog.log 2>&1 < /dev/null & sleep 1; pgrep -af "[w]atchdog.py" | head -n 1'
echo "READY ($MODE). Queue THIS url in the browser:  http://$H:$PUB/predict"
