#!/bin/bash
# The validation flight scored against Oscar's MINED labels (0.967 on the validation API): plain F3 with the deploy
# switches, routed (portal 0.797) and robust, then the loss decomposition. Measurement only. Log: mined_campaign.log
cd "$(dirname "$0")/../../../.." || exit 1
L=elias/out/logs/mined_campaign.log; VP="$HOME/venvs/nordic-drone/Scripts/python.exe"
S="C:/Users/edlun/Desktop/lucky shots/drone-data/scenes/validation_mined_v4"
F3=elias/release/both_m1280.pt; F5=elias/release/F5_fixed_m1280.pt; HN=elias/release/F3HN_m1280.pt
ARGS="--imgsz 1280 --conf 0.05 --birth-confidence 0.25 --update-confidence 0.15"
CE='{"ta-ta":"detector","medium_launcher":"detector"}'; ML='{"medium_launcher": 0.85}'
R_ROUTED='{"small_launcher": 0, "medium_launcher": 0, "large_launcher": 0, "small_plane": 0, "medium_plane": 0, "jet_plane": 0, "small_tower": 0, "large_tower": 0, "tank": 0, "mine_roller": 0, "hangar": 0, "helicopter": 0, "ta-ta": 1, "condor": 0, "jammer": 0, "spacecraft": 0}'
R_ROBUST='{"medium_launcher": 0, "large_launcher": 0, "small_plane": 0, "medium_plane": 0, "jet_plane": 0, "small_tower": 0, "large_tower": 2, "tank": 0, "mine_roller": 0, "hangar": 0, "helicopter": 0, "ta-ta": 1, "condor": 0, "jammer": 0, "spacecraft": 0}'
echo "=== $(date -Is) MINED plain F3, no switches" >> "$L"
WEIGHTS=$F3 bash elias/out/committee/sequences/run_scene.sh "$S" MINED_PLAIN $ARGS >> "$L" 2>&1
echo "=== $(date -Is) MINED routed (portal 0.797)" >> "$L"
DRONE_DETECTOR=elias.ensemble:build ELIAS_WEIGHTS="$F3,$F5" ELIAS_ROUTE="$R_ROUTED" ELIAS_CONF=0.05 DRONE_CLUSTER_BIRTHS=1 DRONE_BOX_SCALE="$ML" WEIGHTS=$F3 \
  bash elias/out/committee/sequences/run_scene.sh "$S" MINED_ROUTED $ARGS --class-extent "$CE" >> "$L" 2>&1
echo "=== $(date -Is) MINED robust" >> "$L"
DRONE_DETECTOR=elias.ensemble:build ELIAS_WEIGHTS="$F3,$F5,$HN" ELIAS_ROUTE="$R_ROBUST" ELIAS_CONF=0.05 DRONE_CLUSTER_BIRTHS=1 DRONE_BOX_SCALE="$ML" \
  DRONE_BOX_HEDGE='{"medium_launcher": [1.176, 0.82], "large_launcher": [0.88], "ta-ta": [0.8, 1.25], "large_tower": [0.87, 1.15]}' \
  DRONE_HEDGE_FACTOR=0.3 DRONE_HEDGE_GROUPS='[["large_launcher","mine_roller"],["large_tower","medium_launcher"]]' WEIGHTS=$F3 \
  bash elias/out/committee/sequences/run_scene.sh "$S" MINED_ROBUST $ARGS --class-extent "$CE" >> "$L" 2>&1
echo "=== $(date -Is) MINED_DONE" >> "$L"
