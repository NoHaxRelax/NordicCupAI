#!/bin/bash
# elias/pod_big.sh: does backbone size keep paying on the UNSEEN scene? (s 0.316, m 0.423 at 960 px, same data)
set -uo pipefail
cd /root/work/drone-flyby || exit 1
until grep -q HIRES_DONE /root/logs/pod_hires.log 2>/dev/null; do sleep 30; done
source <(sed -n '/^AUG=/p' elias/pod_ablate.sh)
source <(sed -n '/^res() {/,/^}/p' elias/pod_hires.sh)
COMMON="device=0 workers=12 amp=True patience=0 plots=False exist_ok=True project=/root/runs"
for spec in "V12_model_l yolo26l.pt 960 12" "V13_model_x yolo26x.pt 960 8"; do set -- $spec
  echo "=== $(date -Is) RUN $1"
  yolo detect train model=$2 data=/root/data/yolo_A/data.yaml imgsz=$3 batch=$4 epochs=8 name=$1 $COMMON $AUG > /root/logs/$1.log 2>&1 || echo "FAILED $1"
  res $1
done
echo "=== $(date -Is) BIG_DONE"
