#!/bin/bash
# elias/pod_ablate2.sh: second wave of cross-scene ablations, started once the first wave has finished.
set -uo pipefail
cd /root/work/drone-flyby || exit 1
until grep -q ABLATE_DONE /root/logs/pod_ablate.log 2>/dev/null; do sleep 30; done
source <(sed -n '/^AUG=/p;/^COMMON=/p' elias/pod_ablate.sh)
source <(sed -n '/^gen() {/,/^}/p;/^run() {/,/^}/p' elias/pod_ablate.sh)
UCM=/root/extra/UCMerced_LandUse/Images
# The reverse direction: sprites and backgrounds from the VALIDATION scene only (pseudo-labels, noisy), tested on REAL
# helsinki views with organiser ground truth. Tells whether validation-derived data is good enough to add to the final mix.
[ -f /root/data/rev/data.yaml ] || python elias/data/synth_yolo.py --out /root/data/rev --n 8000 --n-val 400 --workers 14 --max-bg-frames 200 \n  --sprite-scenes validation --background-scenes validation --real-val-scenes helsinki | tail -3
run R1_val_to_hel_m     rev yolo26m.pt 16
gen extrabg --extra-bg $UCM --extra-bg-prob 0.4
run V9_extra_bg_40     extrabg yolo26s.pt 24
gen extrabg70 --extra-bg $UCM --extra-bg-prob 0.7
run V10_extra_bg_70    extrabg70 yolo26s.pt 24
echo "=== $(date -Is) ABLATE2_DONE"
