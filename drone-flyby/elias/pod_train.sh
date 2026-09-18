#!/bin/bash
# elias/pod_train.sh: on a RunPod training pod, generate synthetic full views and train Ultralytics detectors.
#
#   nohup bash /root/work/drone-flyby/elias/pod_train.sh > /root/logs/pod_train.log 2>&1 &
#
# Stage A is the honest estimate: sprites and backgrounds from helsinki only, validated on REAL views of
# the validation scene, which that model never saw. Stage B is the model to deploy: both scenes, more views,
# longer. Stage C repeats B with a larger backbone if the night allows. Each stage writes STAGE_<x>_DONE.
set -uo pipefail
cd /root/work/drone-flyby || exit 1
mkdir -p /root/data /root/runs /root/logs
export PYTHONUNBUFFERED=1
# Scale is a class cue at fixed altitude and the evaluator never rotates or flips vertically, so the usual
# scale, rotation and vertical-flip augmentations are off; horizontal flips respect the lean rule.
AUG="fliplr=0.5 flipud=0.0 degrees=0.0 shear=0.0 perspective=0.0 scale=0.05 translate=0.05 mosaic=0.5 close_mosaic=5 mixup=0.0 copy_paste=0.0 hsv_h=0.01 hsv_s=0.4 hsv_v=0.3"
COMMON="imgsz=960 device=0 workers=12 amp=True patience=0 plots=False exist_ok=True project=/root/runs"

stage() {  # stage NAME MODEL EPOCHS BATCH DATA
  echo "=== $(date -Is) train $1 ($2, $3 epochs)"
  yolo detect train model="$2" data="$5/data.yaml" epochs="$3" batch="$4" name="$1" save_period=5 $COMMON $AUG \
    || { echo "TRAIN_FAILED $1"; return 1; }
  echo "=== $(date -Is) STAGE_$1_DONE"; touch "/root/runs/STAGE_$1_DONE"
}

MODEL_S=yolo26s.pt; MODEL_M=yolo26m.pt
python - <<'PY' || { MODEL_S=yolo11s.pt; MODEL_M=yolo11m.pt; }
from ultralytics import YOLO
YOLO('yolo26s.pt')
PY
echo "models: $MODEL_S $MODEL_M"

echo "=== $(date -Is) generate A (helsinki -> real validation views)"
python elias/data/synth_yolo.py --out /root/data/yolo_A --n 8000 --n-val 500 --workers 14 --max-bg-frames 200 \
  --sprite-scenes helsinki --background-scenes helsinki --real-val-scenes validation || exit 1
stage A_hel_to_val_s "$MODEL_S" 20 24 /root/data/yolo_A

echo "=== $(date -Is) generate B (both scenes)"
python elias/data/synth_yolo.py --out /root/data/yolo_both --n 16000 --n-val 600 --workers 14 --max-bg-frames 200 \
  --sprite-scenes helsinki validation --background-scenes helsinki validation --real-val-scenes helsinki validation --seed 5 || exit 1
stage B_both_s "$MODEL_S" 40 24 /root/data/yolo_both
stage C_both_m "$MODEL_M" 40 16 /root/data/yolo_both
echo "=== $(date -Is) ALL_DONE"
