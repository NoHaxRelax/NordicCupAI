#!/bin/bash
# launch_final2.sh NAME SEED VARIANT (adds C2 = adjacency recipe without validation material, E2 = C2 + real validation tiles + val-neg): make sure all final data is local, build the VARIANT dataset with the strict hold-out filter and
# start the final on the original schedule (explore_final.sh). VARIANT: D1 (T1-A mix + real validation tiles), D2 (T2-B mix + real
# validation tiles), C1 (train-only + generic negatives, no validation material), E1 (C1 + real validation tiles).
NAME=$1; SEED=$2; V=$3; C=/workspace/drone/explore-20260920; mkdir -p /root/logs; cd /root
[ -f /root/logs/final_boot.done ] || bash $C/code/bootstrap_final.sh > /root/logs/boot_final.log 2>&1
cp $C/code/build_ds_final4.py $C/code/explore_final.sh $C/code/HELDOUT-VIEWS.json /root/; chmod +x /root/explore_final.sh
S=/root/data/flyover-synth-20260919
[ -d $S/val-pos-v1 ] || tar xf $C/data-new/val-pos-v1.tar -C $S
[ -d $S/flypaste-adj-train-v1fix/labels ] || { tar xf $C/data-new/adjfix-labels.tar -C $S; ln -sfn $S/flypaste-adj-train-v1/images $S/flypaste-adj-train-v1fix/images; }
[ -d /root/data/land-neg-v2 ] || tar xf $C/data-new/final-negs.tar -C /root/data
G="/root/data/helsinki-neg-v1:0.04 /root/data/water-neg-train-v1:0.04 /root/data/land-neg-v2:0.04"
case $V in
  D1) SETS="flyover-train-v1 flyover-val-v1 val-pos-v1"; NEG="$G /root/data/val-neg-v1:0.03";;
  D2) SETS="flyover-train-v1 flyover-val-v1 flypaste-adj-train-v1fix val-pos-v1"; NEG="$G /root/data/val-neg-v1:0.03";;
  C1) SETS="flyover-train-v1"; NEG="/root/data/helsinki-neg-v1:0.05 /root/data/water-neg-train-v1:0.05 /root/data/land-neg-v2:0.05";;
  E1) SETS="flyover-train-v1 val-pos-v1"; NEG="/root/data/helsinki-neg-v1:0.05 /root/data/water-neg-train-v1:0.05 /root/data/land-neg-v2:0.05";;
  C2) SETS="flyover-train-v1 flypaste-adj-train-v1fix"; NEG="/root/data/helsinki-neg-v1:0.05 /root/data/water-neg-train-v1:0.05 /root/data/land-neg-v2:0.05";;
  E2) SETS="flyover-train-v1 flypaste-adj-train-v1fix val-pos-v1"; NEG="$G /root/data/val-neg-v1:0.03";;
esac
python3 build_ds_final4.py final-$V --sets $SETS --syn --neg $NEG > logs/build_final_$V.log 2>&1
mkdir -p $C/finals/$NAME; { echo "$NAME: yolo26m 256 px tiles, seed $SEED, schedule epochs=60 save_period=10 -> epoch40.pt shipped; variant $V; sets: $SETS + synthetic-train; negatives: $NEG; hold-out filter strict (HELDOUT-VIEWS.json)"; cat logs/build_final_$V.log | grep -v "^labels per class" ; grep "^labels per class" logs/build_final_$V.log; } > $C/finals/$NAME/README.txt
SEED=$SEED WORKERS=16 DS=/root/datasets/final-$V ./explore_final.sh $NAME > logs/runfinal_$NAME.log 2>&1
