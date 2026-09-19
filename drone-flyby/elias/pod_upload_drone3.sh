#!/bin/bash
# elias/pod_upload_drone3.sh POD_HOST POD_PORT FRAMES_DIR [UCM_DIR]: frames re-encoded as JPEG 95 (4x smaller), two scp
# streams for the two scenes, aerial tiles as one tar stream, then install and start pod_final2.sh. Code was sent already.
set -uo pipefail
HOST=$1; PORT=$2; FR=$3; UCM=${4:-"/c/Users/edlun/AppData/Local/Temp/claude/c--Users-edlun-Desktop-lucky-shots-NordicCupAI/bf48d8ae-2b16-4039-be3b-c570527b37ee/scratchpad/ucm/images"}
cd "$(dirname "$0")/../.." || exit 1
OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=25 -o LogLevel=ERROR -o Compression=no -i $HOME/.ssh/id_ed25519"
SSH="ssh $OPTS -p $PORT root@$HOST"
$SSH "rm -rf /root/work/drone-flyby/src/validation/images /root/work/drone-flyby/src/helsinki/images /root/ucm/*; mkdir -p /root/work/drone-flyby/src/validation /root/work/drone-flyby/src/helsinki /root/ucm" || { echo "ssh failed"; exit 1; }
echo "=== $(date +%H:%M:%S) frames (jpeg) and tiles"
scp -q $OPTS -P $PORT -r "$FR/validation/images" root@$HOST:/root/work/drone-flyby/src/validation/ &
scp -q $OPTS -P $PORT -r "$FR/helsinki/images" root@$HOST:/root/work/drone-flyby/src/helsinki/ &
tar cf - -C "$UCM" . | $SSH "tar xf - -C /root/ucm 2>/dev/null" &
wait
echo "=== $(date +%H:%M:%S) uploaded"
$SSH "ls /root/work/drone-flyby/src/validation/images | wc -l; ls /root/work/drone-flyby/src/helsinki/images | wc -l; find /root/ucm -type f | wc -l; du -sh /root/work/drone-flyby /root/ucm"
echo "=== $(date +%H:%M:%S) install"
$SSH "cd /root/work/drone-flyby && sed -i 's/\r$//' elias/*.sh elias/data/*.py *.py && pip install -q --break-system-packages ultralytics opencv-python-headless pydantic fastapi uvicorn requests 2>&1 | tail -2; python -c 'import ultralytics, cv2, torch; print(ultralytics.__version__, cv2.__version__, torch.__version__, torch.cuda.is_available())'"
echo "=== $(date +%H:%M:%S) start"
$SSH "cd /root/work/drone-flyby && (SYNTH_ORGANISER_BOXES=1 setsid nohup bash elias/pod_final2.sh > /root/logs/final2.log 2>&1 &) ; sleep 2; echo started"
