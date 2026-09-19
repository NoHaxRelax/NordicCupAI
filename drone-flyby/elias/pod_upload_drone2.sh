#!/bin/bash
# elias/pod_upload_drone2.sh POD_HOST POD_PORT [UCM_DIR]
# Same job as pod_upload_drone.sh, but the big folders go by scp in parallel streams instead of one tar pipe, which
# stalled on Windows after 1.5 GB. Code first (small tar), then Helsinki frames, validation frames and aerial tiles as
# three concurrent scp transfers, then install and start pod_final2.sh detached.
set -uo pipefail
HOST=$1; PORT=$2; UCM=${3:-"/c/Users/edlun/AppData/Local/Temp/claude/c--Users-edlun-Desktop-lucky-shots-NordicCupAI/bf48d8ae-2b16-4039-be3b-c570527b37ee/scratchpad/ucm/images"}
cd "$(dirname "$0")/../.." || exit 1
OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=25 -o LogLevel=ERROR -o Compression=no -i $HOME/.ssh/id_ed25519"
SSH="ssh $OPTS -p $PORT root@$HOST"
$SSH "mkdir -p /root/work/drone-flyby/src/validation /root/work/drone-flyby/src/helsinki /root/ucm /root/logs && rm -rf /root/work/drone-flyby/src/validation/images /root/work/drone-flyby/src/helsinki/images" || { echo "ssh failed"; exit 1; }
echo "=== $(date +%H:%M:%S) code"
tar cf - --exclude='drone-flyby/elias/out' --exclude='drone-flyby/runs' --exclude='drone-flyby/logs' --exclude='drone-flyby/.git' \
    --exclude='__pycache__' --exclude='drone-flyby/elias/_review' --exclude='drone-flyby/elias/sprites/_candidates' \
    --exclude='drone-flyby/validation' --exclude='drone-flyby/release' --exclude='drone-flyby/solution' --exclude='drone-flyby/oscar-sprite-synthetic' \
    --exclude='drone-flyby/training' --exclude='drone-flyby/research' --exclude='drone-flyby/src/validation/images' --exclude='drone-flyby/src/helsinki/images' \
    drone-flyby | $SSH "tar xf - -C /root/work 2>/dev/null"
echo "=== $(date +%H:%M:%S) frames and tiles, three streams"
scp -q $OPTS -P $PORT -r drone-flyby/src/helsinki/images root@$HOST:/root/work/drone-flyby/src/helsinki/ &
scp -q $OPTS -P $PORT -r drone-flyby/src/validation/images root@$HOST:/root/work/drone-flyby/src/validation/ &
scp -q $OPTS -P $PORT -r "$UCM"/. root@$HOST:/root/ucm/ &
wait
echo "=== $(date +%H:%M:%S) uploaded"
$SSH "ls /root/work/drone-flyby/src/validation/images | wc -l; ls /root/work/drone-flyby/src/helsinki/images | wc -l; find /root/ucm -type f | wc -l; du -sh /root/work/drone-flyby /root/ucm"
echo "=== $(date +%H:%M:%S) install"
$SSH "cd /root/work/drone-flyby && sed -i 's/\r$//' elias/*.sh elias/data/*.py *.py && pip install -q --break-system-packages ultralytics opencv-python-headless pydantic fastapi uvicorn requests 2>&1 | tail -2; python -c 'import ultralytics, cv2, torch; print(ultralytics.__version__, cv2.__version__, torch.__version__, torch.cuda.is_available())'"
echo "=== $(date +%H:%M:%S) start"
$SSH "cd /root/work/drone-flyby && (SYNTH_ORGANISER_BOXES=1 setsid nohup bash elias/pod_final2.sh > /root/logs/final2.log 2>&1 &) ; sleep 2; echo started"
