#!/bin/bash
# elias/pod_clone_launch.sh SRC_HOST SRC_PORT DST_HOST DST_PORT "ENV ASSIGNMENTS"
# Copies /root/work and /root/ucm from a pod that already has the data to a fresh pod, pod to pod at datacenter
# speed, installs the Python deps there and starts pod_final2.sh with the given variant environment, e.g.
#   bash elias/pod_clone_launch.sh 47.47.180.99 14002 1.2.3.4 5555 "MODEL=yolo26l.pt BATCH=8 NAME=F6_fixed_l1280"
# The source pod gets its own key pair for the hop; the laptop only ferries the public key.
set -uo pipefail
SH=$1; SP=$2; DH=$3; DP=$4; VARIANT=$5
OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=25 -o LogLevel=ERROR -i $HOME/.ssh/id_ed25519"
SRC="ssh $OPTS -p $SP root@$SH"; DST="ssh $OPTS -p $DP root@$DH"
PUB=$($SRC "[ -f /root/.ssh/id_pod ] || ssh-keygen -q -t ed25519 -f /root/.ssh/id_pod -N ''; cat /root/.ssh/id_pod.pub") || { echo "source ssh failed"; exit 1; }
$DST "mkdir -p /root/.ssh && echo '$PUB' >> /root/.ssh/authorized_keys && mkdir -p /root/work /root/ucm /root/logs && nvidia-smi --query-gpu=name --format=csv,noheader" || { echo "destination ssh failed"; exit 1; }
echo "=== $(date +%H:%M:%S) copy pod to pod"
$SRC "tar cf - -C /root work ucm | ssh -o StrictHostKeyChecking=no -i /root/.ssh/id_pod -p $DP root@$DH 'tar xf - -C /root'" || { echo "copy failed"; exit 1; }
$DST "ls /root/work/drone-flyby/src/validation/images | wc -l; find /root/ucm -type f | wc -l; du -sh /root/work /root/ucm"
echo "=== $(date +%H:%M:%S) install"
$DST "cd /root/work/drone-flyby && pip install -q --break-system-packages ultralytics opencv-python-headless pydantic fastapi uvicorn requests 2>&1 | tail -2; python -c 'import ultralytics, torch; print(ultralytics.__version__, torch.__version__, torch.cuda.is_available())'"
echo "=== $(date +%H:%M:%S) start: $VARIANT"
$DST "cd /root/work/drone-flyby && (SYNTH_ORGANISER_BOXES=1 $VARIANT setsid nohup bash elias/pod_final2.sh > /root/logs/final2.log 2>&1 &) ; sleep 2; echo started"
