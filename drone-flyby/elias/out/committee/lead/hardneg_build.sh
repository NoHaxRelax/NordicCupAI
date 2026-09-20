#!/bin/bash
# Item 5 data (CPU): a replay set in the F3 recipe (Helsinki + validation sprites, organiser boxes) with Oscar's empty
# renders as most of the extra backgrounds, plus pure negative views, FIVE SITES HELD OUT for the false-alarm test.
cd "$(dirname "$0")/../../.." || exit 1
VP="$HOME/venvs/nordic-drone/Scripts/python.exe"; FT=elias/out/committee/finetune; L=elias/out/logs/hardneg_build.log
X="C:/Users/edlun/Desktop/lucky shots/drone-data/oscar-pod/x"
UCM="C:/Users/edlun/AppData/Local/Temp/claude/c--Users-edlun-Desktop-lucky-shots-NordicCupAI/bf48d8ae-2b16-4039-be3b-c570527b37ee/scratchpad/ucm/images"
HOLD="nl_flevoland nl_texel fi_pasila nl_rotterdam fi_malmi"
KEEP="fi_helsinki_central,fi_viikki,fi_vuosaari,nl_amersfoort,nl_heuvelrug,nl_lelystad,nl_maasvlakte,nl_veluwe,nl_zeeland"
echo "=== $(date -Is) replay set" >> "$L"
EMPTY_DIR="$X/runs/sets/empty" EMPTY_SHARE=0.8 EMPTY_EXCLUDE="$HOLD" SYNTH_ORGANISER_BOXES=1 "$VP" $FT/synth_yolo_emptybg.py --out "$PWD/$FT/data/replay_both_empty_ho" \
  --n 2600 --n-val 200 --workers 5 --max-bg-frames 200 --sprite-scenes helsinki validation --background-scenes helsinki validation \
  --real-val-scenes helsinki validation --seed 21 --extra-bg "$UCM" --extra-bg-prob 0.6 2>&1 | grep -v libpng | tail -n 4 >> "$L"
echo "=== $(date -Is) negatives" >> "$L"
"$VP" elias/out/committee/backgrounds/make_negatives.py --frames-root "$X/runs/sets/empty" --sites "$KEEP" --into "$PWD/$FT/data/replay_both_empty_ho" --out "$PWD/$FT/data/negatives_ho" --n 300 --seed 5 2>&1 | tail -n 3 >> "$L"
ls $FT/data/replay_both_empty_ho/images/train | wc -l >> "$L"
echo "=== $(date -Is) HARDNEG_BUILD_DONE" >> "$L"
