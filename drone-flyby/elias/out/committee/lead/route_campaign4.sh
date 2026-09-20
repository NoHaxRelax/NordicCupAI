#!/bin/bash
# The robust candidate: F3 for everything, F5 for ta-ta, F3HN for large_tower, small_launcher merged over the three
# models, cluster births, launcher box 0.85, box hedge H4. Two variants: small_launcher merged (M3LT) or from F3HN (HNLT).
cd "$(dirname "$0")/../../.." || exit 1
L=elias/out/logs/route_campaign4.log; VP="$HOME/venvs/nordic-drone/Scripts/python.exe"
S="C:/Users/edlun/Desktop/lucky shots/drone-data/scenes/scene_malmi25"; E="C:/Users/edlun/Desktop/lucky shots/drone-data/oscar-pod/x/runs/sets/empty"
F3=elias/release/both_m1280.pt; F5=elias/release/F5_fixed_m1280.pt; HN=elias/out/committee/finetune/runs/F3HN/weights/last.pt
ARGS="--imgsz 1280 --conf 0.05 --birth-confidence 0.25 --update-confidence 0.15"
CE='{"ta-ta":"detector","medium_launcher":"detector"}'; ML='{"medium_launcher": 0.85}'
H4='{"medium_launcher": [1.176, 0.82], "large_launcher": [0.88], "ta-ta": [0.8, 1.25], "large_tower": [0.87, 1.15]}'
R_M3LT='{"medium_launcher": 0, "large_launcher": 0, "small_plane": 0, "medium_plane": 0, "jet_plane": 0, "small_tower": 0, "large_tower": 2, "tank": 0, "mine_roller": 0, "hangar": 0, "helicopter": 0, "ta-ta": 1, "condor": 0, "jammer": 0, "spacecraft": 0}'
R_HNLT='{"small_launcher": 2, "medium_launcher": 0, "large_launcher": 0, "small_plane": 0, "medium_plane": 0, "jet_plane": 0, "small_tower": 0, "large_tower": 2, "tank": 0, "mine_roller": 0, "hangar": 0, "helicopter": 0, "ta-ta": 1, "condor": 0, "jammer": 0, "spacecraft": 0}'
FLIGHTS="nl_flevoland_lane0 nl_flevoland_lane1 nl_texel_lane0 fi_pasila_lane0 nl_rotterdam_lane0 fi_vuosaari_lane0 nl_veluwe_lane0 nl_maasvlakte_lane0"
one() { TAG=$1; ROUTE=$2
  export DRONE_DETECTOR=elias.ensemble:build ELIAS_WEIGHTS="$F3,$F5,$HN" ELIAS_ROUTE="$ROUTE" ELIAS_CONF=0.05 DRONE_CLUSTER_BIRTHS=1 DRONE_BOX_SCALE="$ML" DRONE_BOX_HEDGE="$H4"
  echo "=== $(date -Is) $TAG" >> "$L"
  bash elias/run_harness.sh $F3 RC_${TAG} $ARGS --class-extent "$CE" >> "$L" 2>&1
  WEIGHTS=$F3 bash elias/out/committee/sequences/run_scene.sh "$S" RC_${TAG} $ARGS --class-extent "$CE" >> "$L" 2>&1
  for f in $FLIGHTS; do WEIGHTS=$F3 bash elias/out/committee/sequences/run_scene.sh "$E/$f" RC_${TAG}_EMPTY_$f $ARGS --class-extent "$CE" > /dev/null 2>&1; done; }
one M3LT "$R_M3LT"
one HNLT "$R_HNLT"
echo "=== $(date -Is) ROUTE_CAMPAIGN4_DONE" >> "$L"
