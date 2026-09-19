#!/bin/bash
# elias/pod_fetch_launch.sh DST_HOST DST_PORT TAR_URL "ENV ASSIGNMENTS": a fresh pod fetches the data tarball that
# pod 1 serves over a quick tunnel (pod-to-pod ssh is blocked), installs deps and starts pod_final2.sh with a variant.
set -uo pipefail
DH=$1; DP=$2; URL=$3; VARIANT=$4
OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=25 -o LogLevel=ERROR -i $HOME/.ssh/id_ed25519"
DST="ssh $OPTS -p $DP root@$DH"
echo "=== $(date +%H:%M:%S) fetch"
$DST "mkdir -p /root/logs && cd /root && curl -sS -L --retry 3 -o /root/data.tar '$URL' && tar xf /root/data.tar -C /root && rm -f /root/data.tar && ls /root/work/drone-flyby/src/validation/images | wc -l && find /root/ucm -type f | wc -l" || { echo "fetch failed"; exit 1; }
echo "=== $(date +%H:%M:%S) install"
$DST "cd /root/work/drone-flyby && sed -i 's/\r$//' elias/*.sh elias/data/*.py *.py && pip install -q --break-system-packages ultralytics opencv-python-headless pydantic fastapi uvicorn requests 2>&1 | tail -2; python -c 'import ultralytics, torch; print(ultralytics.__version__, torch.__version__, torch.cuda.is_available())'"
echo "=== $(date +%H:%M:%S) start: $VARIANT"
$DST "cd /root/work/drone-flyby && (SYNTH_ORGANISER_BOXES=1 $VARIANT setsid nohup bash elias/pod_final2.sh > /root/logs/final2.log 2>&1 &) ; sleep 2; echo started"
