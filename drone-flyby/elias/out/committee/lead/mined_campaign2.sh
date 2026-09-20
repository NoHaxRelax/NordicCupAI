#!/bin/bash
# On the mined-label scene, robust config: tracks of tiny objects die on three silent looks and their frames are lost
# (small_launcher 22 %, ta-ta 21 % of labels). Retired forecasts and the miss count, now measurable against near-true labels.
cd "$(dirname "$0")/../../../.." || exit 1
L=elias/out/logs/mined_campaign2.log
S="C:/Users/edlun/Desktop/lucky shots/drone-data/scenes/validation_mined_v4"; M="C:/Users/edlun/Desktop/lucky shots/drone-data/scenes/scene_malmi25"
F3=elias/release/both_m1280.pt; F5=elias/release/F5_fixed_m1280.pt; HN=elias/release/F3HN_m1280.pt
ARGS="--imgsz 1280 --conf 0.05 --birth-confidence 0.25 --update-confidence 0.15"
CE='{"ta-ta":"detector","medium_launcher":"detector"}'
export DRONE_DETECTOR=elias.ensemble:build ELIAS_WEIGHTS="$F3,$F5,$HN" ELIAS_CONF=0.05 DRONE_CLUSTER_BIRTHS=1 DRONE_BOX_SCALE='{"medium_launcher": 0.85}' WEIGHTS=$F3
export DRONE_BOX_HEDGE='{"medium_launcher": [1.176, 0.82], "large_launcher": [0.88], "ta-ta": [0.8, 1.25], "large_tower": [0.87, 1.15]}'
export DRONE_HEDGE_FACTOR=0.3 DRONE_HEDGE_GROUPS='[["large_launcher","mine_roller"],["large_tower","medium_launcher"]]'
export ELIAS_ROUTE='{"medium_launcher": 0, "large_launcher": 0, "small_plane": 0, "medium_plane": 0, "jet_plane": 0, "small_tower": 0, "large_tower": 2, "tank": 0, "mine_roller": 0, "hangar": 0, "helicopter": 0, "ta-ta": 1, "condor": 0, "jammer": 0, "spacecraft": 0}'
one() { TAG=$1; shift
  echo "=== $(date -Is) $TAG [$*]" >> "$L"
  env "$@" bash elias/out/committee/sequences/run_scene.sh "$S" MINED_$TAG $ARGS --class-extent "$CE" >> "$L" 2>&1
  env "$@" bash elias/out/committee/sequences/run_scene.sh "$M" MALMI_$TAG $ARGS --class-extent "$CE" >> "$L" 2>&1; }
one RET40_03 DRONE_RETIRED_TICKS=40 DRONE_RETIRED_SCALE=0.3
one RET40_06 DRONE_RETIRED_TICKS=40 DRONE_RETIRED_SCALE=0.6
one MISS5    DRONE_MISS_RETIRE=5
one MISS5RET DRONE_MISS_RETIRE=5 DRONE_RETIRED_TICKS=40 DRONE_RETIRED_SCALE=0.5
echo "=== $(date -Is) MINED2_DONE" >> "$L"
