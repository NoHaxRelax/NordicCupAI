#!/bin/bash
# elias/run_harness.sh WEIGHTS TAG [extra run_local_eval args]: a checkpoint through Oscar's REAL pipeline
# (detector on delivered views, perspective tracker, camera policy) on the local validation scene, scored by
# the organisers' scorer against the participant pseudo-labels (11 of the 13 classes present; noisy).
# Policy = what the oracle study found best at realistic recognisable sizes: L1 left-centre-right-centre with
# no L0 overviews, zoom-aware miss rule. Prints mAP and per-class AP; logs under elias/out/harness/<TAG>.
set -uo pipefail
W=$1; TAG=$2; shift 2
cd "$(dirname "$0")/.." || exit 1
VP="$HOME/venvs/nordic-drone/Scripts/python.exe"; OUT=elias/out/harness/$TAG; mkdir -p "$OUT"
DRONE_MISS_RULE=${DRONE_MISS_RULE:-seen} DRONE_CV_THREADS=4 \
  "$VP" run_local_eval.py --weights "$W" --device cuda:0 --scene validation --port ${PORT:-9171} \
  --overview-between-sides ${OVERVIEW:-0} --eval-timeout-s 60 --log-dir "$OUT" "$@" > "$OUT/run.log" 2>&1
sed -n '/AP@0.50 by class/,/COCO mAP/p' "$OUT/run.log" | tr -s ' ' | paste -sd' ' | fold -w 200
grep -E "frames skipped|round trip|responses accepted|invalid" "$OUT/run.log" | tr -s ' ' | paste -sd' '
