#!/bin/bash
# run_views.sh NAME MODEL SPRITE_SCENES BG_SCENES [EXTRA_GEN_ARGS]: Elias's F3 recipe (commit 0ee9364, pod_final.sh)
# with the scene pools chosen here. Data /root/data/views-NAME, run /root/runs/NAME. Waits for pip + UCM.
set -u
NAME=$1; MODEL=$2; SPR=$3; BG=$4; EXTRA=${5:-}
cd /root/work/drone-flyby || exit 1
until grep -q PIP_DONE /root/logs/pip.log 2>/dev/null; do sleep 10; done
until grep -q UCM_OK /root/logs/ucm.log 2>/dev/null; do grep -q UCM_FAIL /root/logs/ucm.log 2>/dev/null && { echo UCM missing; exit 1; }; sleep 10; done
export PYTHONUNBUFFERED=1 YOLO_CONFIG_DIR=/tmp/Ultralytics SYNTH_ORGANISER_BOXES=1
source <(sed -n '/^AUG=/p' elias/pod_ablate.sh)
mkdir -p /root/runs /root/logs
if [ ! -f /root/data/views-$NAME/data.yaml ]; then
  echo "=== $(date -Is) generate $NAME"
  python elias/data/synth_yolo.py --out /root/data/views-$NAME --n ${N_VIEWS:-12000} --n-val 600 --workers 20 --max-bg-frames 200 \
    --sprite-scenes $SPR --background-scenes $BG --real-val-scenes ${VAL_SCENES:-helsinki} --seed 7 \
    --extra-bg /root/extra/UCMerced_LandUse/Images --extra-bg-prob 0.4 $EXTRA 2>&1 | tail -4
fi
echo "=== $(date -Is) train $NAME $MODEL"
yolo detect train model=$MODEL data=/root/data/views-$NAME/data.yaml imgsz=1280 batch=${BATCH:-8} epochs=${EPOCHS:-5} name=$NAME \
  save_period=1 device=0 workers=12 amp=True patience=0 plots=False exist_ok=True project=/root/runs $AUG > /root/logs/train_$NAME.log 2>&1 || echo "FAILED $NAME"
echo "=== $(date -Is) DONE $NAME"
