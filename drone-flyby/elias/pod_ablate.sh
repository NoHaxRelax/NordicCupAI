#!/bin/bash
# elias/pod_ablate.sh: cross-scene detector ablations. Every run trains on synthetic views made from helsinki
# sprites and backgrounds only and is validated on REAL views of the validation scene, so every number is an
# unseen-scene number. 12 epochs each: the baseline peaks by epoch 3.
#   nohup bash /root/work/drone-flyby/elias/pod_ablate.sh > /root/logs/pod_ablate.log 2>&1 &
set -uo pipefail
cd /root/work/drone-flyby || exit 1
mkdir -p /root/data /root/runs /root/logs; export PYTHONUNBUFFERED=1
AUG="fliplr=0.5 flipud=0.0 degrees=0.0 shear=0.0 perspective=0.0 scale=0.05 translate=0.05 mosaic=0.5 close_mosaic=3 mixup=0.0 copy_paste=0.0 hsv_h=0.01 hsv_s=0.4 hsv_v=0.3"
COMMON="imgsz=960 device=0 workers=12 amp=True patience=0 plots=False exist_ok=True project=/root/runs epochs=12"
gen() {  # gen NAME [generator args...]; environment carries the exposure and alpha settings
  local name=$1; shift
  [ -f /root/data/$name/data.yaml ] || python elias/data/synth_yolo.py --out /root/data/$name --n 8000 --n-val 500 --workers 14 --max-bg-frames 200 \
    --sprite-scenes helsinki --background-scenes helsinki --real-val-scenes validation "$@" | tail -3
}
run() {  # run NAME DATA MODEL BATCH [extra yolo args]
  local name=$1 data=$2 model=$3 batch=$4; shift 4
  echo "=== $(date -Is) RUN $name"
  yolo detect train model=$model data=/root/data/$data/data.yaml batch=$batch name=$name $COMMON $AUG "$@" > /root/logs/$name.log 2>&1 || echo "FAILED $name"
  python - "$name" <<'PY'
import csv, sys
name = sys.argv[1]; rows = list(csv.DictReader(open(f'/root/runs/{name}/results.csv')))
k = [c for c in rows[0] if 'mAP50(B)' in c][0]; r = [c for c in rows[0] if 'recall(B)' in c][0]; p = [c for c in rows[0] if 'precision(B)' in c][0]
m = [float(x[k]) for x in rows]; b = max(range(len(m)), key=m.__getitem__)
print(f"RESULT {name:<22} best mAP50 {m[b]:.3f} at epoch {b+1}/{len(m)}  (P {float(rows[b][p]):.2f} R {float(rows[b][r]):.2f})  last {m[-1]:.3f}  curve " + ' '.join(f'{x:.2f}' for x in m))
PY
}
gen base
run V0_base            base yolo26s.pt 24
run V8_low_colour      base yolo26s.pt 24 hsv_h=0.0 hsv_s=0.1 hsv_v=0.2   # the window classifier lost 0.08 to colour jitter
run V1_freeze10        base yolo26s.pt 24 freeze=10
run V2_model_m         base yolo26m.pt 16
SYNTH_ALPHA_MODE=soft gen soft --sprite-blur-max 0.8
run V4_blend_soft      soft yolo26s.pt 24
SYNTH_ALPHA_MODE=hard gen hard
run V3_blend_hard      hard yolo26s.pt 24
gen freerot --free-rotation
run V5_free_rotation   freerot yolo26s.pt 24
SYNTH_SPRITE_GAIN=0.92,1.08 SYNTH_BG_GAIN=0.96,1.04 gen narrow
run V6_narrow_exposure narrow yolo26s.pt 24
run V7_lowlr_nomosaic  base yolo26s.pt 24 lr0=0.002 mosaic=0.0
echo "=== $(date -Is) ABLATE_DONE"
