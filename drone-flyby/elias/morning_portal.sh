#!/bin/bash
# elias/morning_portal.sh HOST SSH_PORT PUBLIC_PORT: Sunday morning's portal confirmations from a fresh Oslo pod, in order
# of value, every run concealed (one class, or a third). Needs the three checkpoints in /root/out/ (F3_both_m1280.last.pt,
# F5_fixed_m1280.last.pt, F3HN_m1280.pt) and the portal free of teammates' loops. Log: elias/out/logs/morning_portal.log.
# Same host, same hour: every pair below is decided by its own two runs, never against Saturday's numbers.
set -uo pipefail
H=$1; P=$2; PUB=$3
cd "$(dirname "$0")/.." || exit 1
export POD_HOST=$H POD_PORT=$P DIRECT_URL=http://$H:$PUB IMGSZ=1280 CLUSTER_BIRTHS=1
export CLASS_EXTENT='{"medium_launcher":"detector","ta-ta":"detector"}'
A=elias/out/logs/morning_portal.log; F3=/root/out/F3_both_m1280.last.pt
ML='{"medium_launcher": 0.85}'; ML_LL='{"medium_launcher": 0.85, "large_launcher": 0.88}'
W2=/root/out/F3_both_m1280.last.pt,/root/out/F5_fixed_m1280.last.pt; W3=$W2,/root/out/F3HN_m1280.pt
HEDGE='{"medium_launcher": [1.176, 0.82], "large_launcher": [0.88], "ta-ta": [0.8, 1.25], "large_tower": [0.87, 1.15]}'
R_ROUTED='{"small_launcher": 0, "medium_launcher": 0, "large_launcher": 0, "small_plane": 0, "medium_plane": 0, "jet_plane": 0, "small_tower": 0, "large_tower": 0, "tank": 0, "mine_roller": 0, "hangar": 0, "helicopter": 0, "ta-ta": 1, "condor": 0, "jammer": 0, "spacecraft": 0}'
R_ROBUST='{"medium_launcher": 0, "large_launcher": 0, "small_plane": 0, "medium_plane": 0, "jet_plane": 0, "small_tower": 0, "large_tower": 2, "tank": 0, "mine_roller": 0, "hangar": 0, "helicopter": 0, "ta-ta": 1, "condor": 0, "jammer": 0, "spacecraft": 0}'
R_ROBUST_HN='{"small_launcher": 2, "medium_launcher": 0, "large_launcher": 0, "small_plane": 0, "medium_plane": 0, "jet_plane": 0, "small_tower": 0, "large_tower": 2, "tank": 0, "mine_roller": 0, "hangar": 0, "helicopter": 0, "ta-ta": 1, "condor": 0, "jammer": 0, "spacecraft": 0}'
run() { TAG=$1; WIN=$2; shift 2
  echo "=== $(date -Is) $TAG window '$WIN' [$*]" >> "$A"
  env "$@" bash elias/pod_portal.sh $F3 $TAG "$WIN" > elias/out/logs/portal_$TAG.log 2>&1
  grep -E "RESULT|no URL|could not|server saw" elias/out/logs/portal_$TAG.log | cut -c1-220 >> "$A"; sleep 5; }
routed() { run "$1" "$2" ANSWER_CLASSES="$3" BOX_SCALE="$ML" DETECTOR=elias.ensemble:build ELIAS_WEIGHTS=$W2 ELIAS_ROUTE="$R_ROUTED"; }
PAIRS='[["large_launcher","mine_roller"],["large_tower","medium_launcher"]]'
robust() { run "$1" "$2" ANSWER_CLASSES="$3" BOX_SCALE="$ML" BOX_HEDGE="$HEDGE" HEDGE=0.3 HEDGE_GROUPS="$PAIRS" DETECTOR=elias.ensemble:build ELIAS_WEIGHTS=$W3 ELIAS_ROUTE="$R_ROBUST"; }
# 1. the two classes the robust mode answers differently, one class per run (score x 13 = class AP)
routed M_lt_routed "" large_tower;      robust M_lt_robust "" large_tower
routed M_sl_routed "" small_launcher;   robust M_sl_robust "" small_launcher
run M_sl_robusthn "" ANSWER_CLASSES=small_launcher BOX_SCALE="$ML" BOX_HEDGE="$HEDGE" DETECTOR=elias.ensemble:build ELIAS_WEIGHTS=$W3 ELIAS_ROUTE="$R_ROBUST_HN"
# 2. the large_launcher box as the primary answer
routed M_ll_routed "" large_launcher;   robust M_ll_robust "" large_launcher
routed M_mr_routed "" mine_roller;      robust M_mr_robust "" mine_roller
run M_ll_088 "" ANSWER_CLASSES=large_launcher BOX_SCALE="$ML_LL" DETECTOR=elias.ensemble:build ELIAS_WEIGHTS=$W2 ELIAS_ROUTE="$R_ROUTED"
echo "=== $(date -Is) CLASSES_DONE" >> "$A"
# 3. thirds of the whole robust config against routed (0.797 on Sunday 00:50), interleaved
for w in 0:83 83:166 166:100000; do t=${w/:/_}; routed M_routed_$t $w ""; robust M_robust_$t $w ""; done
echo "=== $(date -Is) MORNING_DONE" >> "$A"
