#!/bin/bash
# elias/pod_serve_setup.sh HOST PORT: prepare a serving pod (code only, no frames), install deps, report RTT to the
# competition server and the pod's public endpoint port mapping is printed by the caller.
set -uo pipefail
HOST=$1; PORT=$2
cd "$(dirname "$0")/../.." || exit 1
OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=25 -o LogLevel=ERROR -o Compression=no -i $HOME/.ssh/id_ed25519"
SSH="ssh $OPTS -p $PORT root@$HOST"
$SSH "mkdir -p /root/work /root/logs /root/out" || { echo "ssh failed"; exit 1; }
tar cf - --exclude='drone-flyby/elias/out' --exclude='drone-flyby/runs' --exclude='drone-flyby/logs' --exclude='drone-flyby/.git' \
    --exclude='__pycache__' --exclude='drone-flyby/elias/_review' --exclude='drone-flyby/elias/sprites/_candidates' \
    --exclude='drone-flyby/validation' --exclude='drone-flyby/release' --exclude='drone-flyby/solution' --exclude='drone-flyby/oscar-sprite-synthetic' \
    --exclude='drone-flyby/training' --exclude='drone-flyby/research' --exclude='drone-flyby/src/validation/images' --exclude='drone-flyby/src/helsinki/images' \
    drone-flyby | $SSH "tar xf - -C /root/work 2>/dev/null"
$SSH "cd /root/work/drone-flyby && sed -i 's/\r$//' elias/*.sh elias/data/*.py *.py && pip install -q --break-system-packages ultralytics opencv-python-headless pydantic fastapi uvicorn requests 2>&1 | tail -2; python -c 'import ultralytics, torch; print(ultralytics.__version__, torch.__version__, torch.cuda.is_available())'; [ -x /root/cloudflared ] || (wget -q -O /root/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /root/cloudflared); for i in 1 2 3; do curl -s -o /dev/null -w 'RTT to cases.nordicaicup.com: connect %{time_connect}s total %{time_total}s\n' https://cases.nordicaicup.com/; done; curl -s ipinfo.io 2>/dev/null | grep -E '\"(city|country)\"' | tr -d '\n '; echo"
echo "=== $(date +%H:%M:%S) serving pod ready"
