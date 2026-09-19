#!/bin/bash
# elias/pod_final.sh: the model to deploy. Recipe fixed by the unseen-scene ablations of 2026-09-19:
# medium backbone, trained at 1280, lean-limited rotation, hard-edged paste, wide independent exposure, SHORT
# schedule (unseen-scene accuracy peaks after about 40-60k samples), sprites and backgrounds from BOTH scenes,
# views that contain a known label error or the unlabelled small planes are never generated.
set -uo pipefail
cd /root/work/drone-flyby || exit 1
mkdir -p /root/data /root/runs /root/logs; export PYTHONUNBUFFERED=1
source <(sed -n '/^AUG=/p' elias/pod_ablate.sh)
echo "=== $(date -Is) generate"
python elias/data/synth_yolo.py --out /root/data/final --n ${N_VIEWS:-12000} --n-val 600 --workers 13 --max-bg-frames 200 \
  --sprite-scenes helsinki validation --background-scenes helsinki validation --real-val-scenes helsinki validation --seed 7 ${GEN_EXTRA:-} | tail -3
echo "=== $(date -Is) train"
yolo detect train model=${MODEL:-yolo26m.pt} data=/root/data/final/data.yaml imgsz=${IMGSZ:-1280} batch=${BATCH:-16} epochs=${EPOCHS:-5} name=${NAME:-F_both_m1280} \
  save_period=1 device=0 workers=12 amp=True patience=0 plots=False exist_ok=True project=/root/runs $AUG > /root/logs/final_train.log 2>&1 || echo "FAILED"
python - <<'PY'
import csv, glob
for d in glob.glob('/root/runs/F_*/results.csv'):
    rows = list(csv.DictReader(open(d))); k = [c for c in rows[0] if 'mAP50(B)' in c][0]
    print('RESULT', d.split('/')[3], 'in-scene real-view mAP50 per epoch:', ' '.join(f'{float(r[k]):.3f}' for r in rows))
PY
echo "=== $(date -Is) FINAL_DONE"
