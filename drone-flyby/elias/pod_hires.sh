#!/bin/bash
# elias/pod_hires.sh: does training at a higher input size help on the UNSEEN scene? Same Helsinki-only data as
# stage A / V0_base (mAP50 0.316-0.318 at 960), validated on real validation-scene views.
set -uo pipefail
cd /root/work/drone-flyby || exit 1
source <(sed -n '/^AUG=/p' elias/pod_ablate.sh)
COMMON="device=0 workers=12 amp=True patience=0 plots=False exist_ok=True project=/root/runs"
res() { python - "$1" <<'PY'
import csv, sys
name = sys.argv[1]; rows = list(csv.DictReader(open(f'/root/runs/{name}/results.csv')))
k = [c for c in rows[0] if 'mAP50(B)' in c][0]; m = [float(x[k]) for x in rows]; b = max(range(len(m)), key=m.__getitem__)
print(f"RESULT {name:<22} best mAP50 {m[b]:.3f} at epoch {b+1}/{len(m)}  last {m[-1]:.3f}  curve " + ' '.join(f'{x:.2f}' for x in m))
PY
}
for spec in "H1_1280 1280 16" "H2_1536 1536 10"; do set -- $spec
  echo "=== $(date -Is) RUN $1"
  yolo detect train model=yolo26s.pt data=/root/data/yolo_A/data.yaml imgsz=$2 batch=$3 epochs=8 name=$1 $COMMON $AUG > /root/logs/$1.log 2>&1 || echo "FAILED $1"
  res $1
done
echo "=== $(date -Is) HIRES_DONE"
