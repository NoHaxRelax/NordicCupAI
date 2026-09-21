#!/bin/bash
# explore_final.sh NAME: FINAL tile member on the ORIGINAL schedule of m-fly-epoch40-tile - epochs=60, save_period=10, and the
# epoch-40 checkpoint is the model (decided up front, no picking). A watcher ships epoch40.pt to the volume and stops the run.
# Recipe otherwise identical to the base (AdamW 1e-3, mosaic 1.0 / close 10, mixup 0.15, scale ${SCALE:-0.3}, degrees 15).
set -u
NAME=$1; DS=$DS; OUT=/workspace/drone/explore-20260920/finals/$NAME
export YOLO_CONFIG_DIR=/tmp/Ultralytics PYTHONUNBUFFERED=1
mkdir -p /root/runs-final /root/logs $OUT; cd /root
echo "=== $(date -Is) FINAL $NAME seed=${SEED:-7} scale=${SCALE:-0.3} data=$DS schedule=60 take=epoch40"
( until [ -f /root/runs-final/$NAME/weights/epoch40.pt ]; do sleep 20; done; sleep 15
  cp /root/runs-final/$NAME/weights/epoch40.pt $OUT/epoch40.pt; cp /root/runs-final/$NAME/results.csv /root/runs-final/$NAME/args.yaml $DS/data.yaml $OUT/ 2>/dev/null
  echo "=== $(date -Is) SHIPPED $NAME epoch40" >> /root/logs/runfinal_$NAME.log; pkill -f "name=$NAME project=/root/runs-final" ) &
yolo detect train model=yolo26m.pt data=$DS/data.yaml imgsz=256 batch=64 epochs=60 name=$NAME project=/root/runs-final \
  optimizer=AdamW lr0=0.001 seed=${SEED:-7} patience=0 plots=False exist_ok=True device=0 workers=${WORKERS:-12} cache=False save_period=10 pretrained=True \
  mosaic=1.0 close_mosaic=10 mixup=0.15 translate=0.1 scale=${SCALE:-0.3} degrees=15 flipud=0.5 fliplr=0.5 > /root/logs/trainfinal_$NAME.log 2>&1
echo "=== $(date -Is) DONE $NAME"
