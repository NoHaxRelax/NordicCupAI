#!/bin/bash
# elias/pod_upload_drone.sh POD_HOST POD_PORT [UCM_DIR]
# Ships the drone-flyby folder (code, sprites, bank, keep-outs, the Helsinki and validation frames) plus the aerial
# tiles to a fresh RunPod pod, installs what the image lacks, and starts pod_final2.sh detached. Nothing under .claude/
# is sent, no keys or tokens; the team key never leaves the laptop. Excludes the local outputs and runs.
set -uo pipefail
HOST=$1; PORT=$2; UCM=${3:-"/c/Users/edlun/AppData/Local/Temp/claude/c--Users-edlun-Desktop-lucky-shots-NordicCupAI/bf48d8ae-2b16-4039-be3b-c570527b37ee/scratchpad/ucm/images"}
cd "$(dirname "$0")/../.." || exit 1          # repo root of the worktree
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -i $HOME/.ssh/id_ed25519 -p $PORT root@$HOST"
ssh-keygen -R "[$HOST]:$PORT" > /dev/null 2>&1
$SSH "mkdir -p /root/work /root/ucm /root/logs && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" || { echo "ssh failed"; exit 1; }
echo "=== $(date +%H:%M:%S) upload frames and code"
tar cf - --exclude='drone-flyby/elias/out' --exclude='drone-flyby/runs' --exclude='drone-flyby/logs' --exclude='drone-flyby/.git' \
    --exclude='__pycache__' --exclude='drone-flyby/elias/_review' --exclude='drone-flyby/elias/sprites/_candidates' \
    --exclude='drone-flyby/validation' --exclude='drone-flyby/release' --exclude='drone-flyby/solution' --exclude='drone-flyby/oscar-sprite-synthetic' \
    --exclude='drone-flyby/training' --exclude='drone-flyby/research' \
    drone-flyby | $SSH "tar xf - -C /root/work"
echo "=== $(date +%H:%M:%S) upload aerial tiles"
tar cf - -C "$(dirname "$UCM")" "$(basename "$UCM")" | $SSH "tar xf - -C /root/ucm --strip-components=1"
echo "=== $(date +%H:%M:%S) install"
$SSH "cd /root/work/drone-flyby && sed -i 's/\r$//' elias/*.sh elias/data/*.py *.py && pip install -q --break-system-packages ultralytics opencv-python-headless pydantic fastapi uvicorn requests 2>&1 | tail -2; python -c 'import ultralytics, cv2, torch; print(ultralytics.__version__, cv2.__version__, torch.__version__, torch.cuda.is_available())'; ls /root/ucm | head -3; ls /root/work/drone-flyby/src/validation/images | wc -l"
echo "=== $(date +%H:%M:%S) start"
$SSH "cd /root/work/drone-flyby && (SYNTH_ORGANISER_BOXES=1 setsid nohup bash elias/pod_final2.sh > /root/logs/final2.log 2>&1 &) ; sleep 2; echo started"
