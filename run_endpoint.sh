#!/usr/bin/env bash
# Start the evaluation endpoint from this checkout with the settings of the served process.
#   ./run_endpoint.sh [PORT]            GPU (as served):  needs CUDA
#   DEVICE=cpu ./run_endpoint.sh        CPU: works, but far slower than the organiser's 333 ms frame interval
set -eu
REPO="$(cd "$(dirname "$0")" && pwd)"; PORT=${1:-19130}; DEVICE=${DEVICE:-cuda:0}
RUN="${NORDIC_RUN_DIR:-$REPO/run}"; mkdir -p "$RUN"
for w in T1-E-SV-s7.pt T2-E-s7.pt fly36-l.pt both_m1280.pt; do [ -f "$REPO/weights/$w" ] || { echo "missing weights/$w"; exit 1; }; done
cat > "$RUN/members.json" <<JSON
[{"kind":"tile","name":"T1-E-SV-s7","weights":"$REPO/weights/T1-E-SV-s7.pt"},
 {"kind":"tile","name":"T2-E-s7","weights":"$REPO/weights/T2-E-s7.pt"},
 {"kind":"views","name":"fly36-l","weights":"$REPO/weights/fly36-l.pt","imgsz":1280}]
JSON
HALF=1; [ "$DEVICE" = cpu ] && HALF=0
cd "$REPO/endpoint"
exec env NORDIC_RUN_DIR="$RUN" PYTHONPATH="$REPO/endpoint/sweep-ens" \
  CP02_DRONE_WEIGHTS="$REPO/weights/both_m1280.pt" \
  CP02_DRONE_DEVICE="$DEVICE" CP02_DRONE_SYNTH_HALF="$HALF" \
  CP02_DRONE_DETECTOR=ensemble_detector:build \
  CP02_DRONE_ENS_MEMBERS="$RUN/members.json" \
  CP02_DRONE_SYNTH_CONF=0.25 CP02_DRONE_BIRTH_CONFIDENCE=0.4 CP02_DRONE_UPDATE_CONFIDENCE=0.3 \
  CP02_DRONE_REVISIT_EVERY=0 CP02_DRONE_EXTENT_POLICY=detector \
  CP02_DRONE_EXTENT_CLASS_POLICY="$REPO/config/classpolicy-ml-sl.json" \
  python3 -m uvicorn api_checkpoint02:app --host 0.0.0.0 --port "$PORT"
