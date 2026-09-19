#!/bin/bash
# elias/pod_final2.sh: the F3 recipe (pod_final.sh) on the corrected data of 2026-09-19, at batch 16 on a pod.
# Differences from the laptop F4 run that regressed (0.53 against 0.63 on the laptop): batch 16 instead of 2,
# organiser-only boxes exported explicitly, aerial tiles from /root/ucm, shadow-free ta-ta sprites in the bank.
#
#   SYNTH_ORGANISER_BOXES=1 nohup bash /root/work/drone-flyby/elias/pod_final2.sh > /root/logs/final2.log 2>&1 &
set -uo pipefail
cd /root/work/drone-flyby || exit 1
mkdir -p /root/data /root/runs /root/logs /root/out; export PYTHONUNBUFFERED=1 SYNTH_ORGANISER_BOXES=1
source <(sed -n '/^AUG=/p' elias/pod_ablate.sh)
echo "=== $(date -Is) generate"
python elias/data/synth_yolo.py --out /root/data/final2 --n ${N_VIEWS:-12000} --n-val 600 --workers ${WORKERS:-12} --max-bg-frames 200 \
  --sprite-scenes helsinki validation --background-scenes helsinki validation --real-val-scenes helsinki validation --seed 7 \
  --extra-bg /root/ucm --extra-bg-prob 0.4 2>&1 | tail -4
echo "=== $(date -Is) train"
yolo detect train model=${MODEL:-yolo26m.pt} data=/root/data/final2/data.yaml imgsz=${IMGSZ:-1280} batch=${BATCH:-16} epochs=${EPOCHS:-5} name=${NAME:-F5_fixed_m1280} \
  save_period=1 device=0 workers=12 amp=True patience=0 plots=False exist_ok=True project=/root/runs $AUG > /root/logs/final2_train.log 2>&1 || echo "FAILED"
cp /root/runs/${NAME:-F5_fixed_m1280}/weights/last.pt /root/out/${NAME:-F5_fixed_m1280}.last.pt 2>/dev/null
python - <<'PY'
import csv, glob
for d in glob.glob('/root/runs/F5_*/results.csv'):
    rows = list(csv.DictReader(open(d))); k = [c for c in rows[0] if 'mAP50(B)' in c][0]
    print('RESULT', d.split('/')[3], 'in-scene real-view mAP50 per epoch:', ' '.join(f'{float(r[k]):.3f}' for r in rows))
PY
echo "=== $(date -Is) FINAL2_DONE"
