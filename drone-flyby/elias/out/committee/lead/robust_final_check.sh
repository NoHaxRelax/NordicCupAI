#!/bin/bash
# The exact environment of `pod_start_final.sh ... robust` on the three laptop instruments (harness, exact-label scene,
# eight empty flights). Log: elias/out/logs/robust_final_check.log
cd "$(dirname "$0")/../../../.." || exit 1
L=elias/out/logs/robust_final_check.log
S="C:/Users/edlun/Desktop/lucky shots/drone-data/scenes/scene_malmi25"; E="C:/Users/edlun/Desktop/lucky shots/drone-data/oscar-pod/x/runs/sets/empty"
F3=elias/release/both_m1280.pt; F5=elias/release/F5_fixed_m1280.pt; HN=elias/release/F3HN_m1280.pt
ARGS="--imgsz 1280 --conf 0.05 --birth-confidence 0.25 --update-confidence 0.15"
CE='{"ta-ta":"detector","medium_launcher":"detector"}'
export DRONE_DETECTOR=elias.ensemble:build ELIAS_WEIGHTS="$F3,$F5,$HN" ELIAS_CONF=0.05 DRONE_CLUSTER_BIRTHS=1 DRONE_BOX_SCALE='{"medium_launcher": 0.85}'
export DRONE_BOX_HEDGE='{"medium_launcher": [1.176, 0.82], "large_launcher": [0.88], "ta-ta": [0.8, 1.25], "large_tower": [0.87, 1.15]}'
export DRONE_HEDGE_FACTOR=0.3 DRONE_HEDGE_GROUPS='[["large_launcher","mine_roller"],["large_tower","medium_launcher"]]'
export ELIAS_ROUTE='{"medium_launcher": 0, "large_launcher": 0, "small_plane": 0, "medium_plane": 0, "jet_plane": 0, "small_tower": 0, "large_tower": 2, "tank": 0, "mine_roller": 0, "hangar": 0, "helicopter": 0, "ta-ta": 1, "condor": 0, "jammer": 0, "spacecraft": 0}'
echo "=== $(date -Is) ROBUST harness" >> "$L"; bash elias/run_harness.sh $F3 RC_ROBUST $ARGS --class-extent "$CE" >> "$L" 2>&1
echo "=== $(date -Is) ROBUST scene" >> "$L"; WEIGHTS=$F3 bash elias/out/committee/sequences/run_scene.sh "$S" RC_ROBUST $ARGS --class-extent "$CE" >> "$L" 2>&1
for f in nl_flevoland_lane0 nl_flevoland_lane1 nl_texel_lane0 fi_pasila_lane0 nl_rotterdam_lane0 fi_vuosaari_lane0 nl_veluwe_lane0 nl_maasvlakte_lane0; do
  WEIGHTS=$F3 bash elias/out/committee/sequences/run_scene.sh "$E/$f" RC_ROBUST_EMPTY_$f $ARGS --class-extent "$CE" > /dev/null 2>&1; done
echo "=== $(date -Is) ROBUST_FINAL_DONE" >> "$L"
