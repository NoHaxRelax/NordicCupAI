#!/bin/bash
# After the route campaign and the data build: (1) the candidate hedge config H4 on both scenes, (2) the vertically
# flipped validation flight (bottom band, first end-to-end run of that branch), (3) the held-out hard-negative replay
# fine-tune F3HN and its three checks: harness, exact-label scene, false births on the five held-out empty sites.
cd "$(dirname "$0")/../../.." || exit 1
until grep -q "ROUTE_CAMPAIGN_DONE" elias/out/logs/route_campaign.log 2>/dev/null && grep -q "HARDNEG_BUILD_DONE" elias/out/logs/hardneg_build.log 2>/dev/null; do sleep 15; done
L=elias/out/logs/post_campaign.log; VP="$HOME/venvs/nordic-drone/Scripts/python.exe"; FT=elias/out/committee/finetune
S="C:/Users/edlun/Desktop/lucky shots/drone-data/scenes/scene_malmi25"; V="C:/Users/edlun/Desktop/lucky shots/drone-data/scenes/validation_vflip"
E="C:/Users/edlun/Desktop/lucky shots/drone-data/oscar-pod/x/runs/sets/empty"; F3=elias/release/both_m1280.pt
ARGS="--imgsz 1280 --conf 0.05 --birth-confidence 0.25 --update-confidence 0.15"
CE='{"ta-ta":"detector","medium_launcher":"detector"}'; ML='{"medium_launcher": 0.85}'
H4='{"medium_launcher": [1.176, 0.82], "large_launcher": [0.88], "ta-ta": [0.8, 1.25], "large_tower": [0.87, 1.15]}'
export DRONE_CLUSTER_BIRTHS=1 DRONE_BOX_SCALE="$ML"
echo "=== $(date -Is) H4 harness + scene" >> "$L"
DRONE_BOX_HEDGE="$H4" bash elias/run_harness.sh $F3 HC_H4 $ARGS --class-extent "$CE" >> "$L" 2>&1
DRONE_BOX_HEDGE="$H4" WEIGHTS=$F3 bash elias/out/committee/sequences/run_scene.sh "$S" HC_H4 $ARGS --class-extent "$CE" >> "$L" 2>&1
echo "=== $(date -Is) VFLIP (bottom band) and the same 120 frames unflipped for reference" >> "$L"
WEIGHTS=$F3 bash elias/out/committee/sequences/run_scene.sh "$V" VFLIP $ARGS --class-extent "$CE" >> "$L" 2>&1
grep -a -o '"band": {[^}]*}' elias/out/committee/sequences/runs/VFLIP/local.jsonl | sort | uniq -c | head -n 3 >> "$L"
echo "=== $(date -Is) train F3HN" >> "$L"
( cd $FT && "$VP" train_finetune.py --tiles data/tv_full --replay data/replay_both_empty_ho --replay-only --name F3HN --epochs 1 --n-train 2600 --batch 4 --val replay > ../../logs/train_F3HN.log 2>&1 )
W="$FT/runs/F3HN/weights/last.pt"; ls -la "$W" >> "$L" 2>&1
echo "=== $(date -Is) F3HN harness + scene" >> "$L"
bash elias/run_harness.sh "$W" HN_F3HN $ARGS --class-extent "$CE" >> "$L" 2>&1
WEIGHTS="$W" bash elias/out/committee/sequences/run_scene.sh "$S" HN_F3HN $ARGS --class-extent "$CE" >> "$L" 2>&1
echo "=== $(date -Is) false births on the held-out sites: F3, F5, F3HN" >> "$L"
FR=("$E"/nl_flevoland_*/images/*.png "$E"/nl_texel_*/images/*.png "$E"/fi_pasila_*/images/*.png "$E"/nl_rotterdam_*/images/*.png "$E"/fi_malmi_*/images/*.png)   # an array: the paths contain a space
"$VP" elias/out/committee/backgrounds/bg_fp.py --weights "$F3=F3,elias/release/F5_fixed_m1280.pt=F5,$W=F3HN" --frames "${FR[@]}" --tag heldout --out elias/out/committee/lead/bgfp --levels 1 >> "$L" 2>&1
echo "=== $(date -Is) POST_CAMPAIGN_DONE" >> "$L"
