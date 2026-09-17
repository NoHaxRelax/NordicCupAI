#!/bin/zsh
# Run a trapper batch on the Windows PC ("mypc", 24 cores) and copy the results back.
#   scripts/trapper/remote_batch.sh LABEL "--seeds 1 8 --repeats 2 --seconds 600 --modes both"
# Syncs the current survival-simulator code first (results and venv excluded).
set -e
LABEL=$1; ARGS=$2
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT/.."
tar czf /tmp/survsim-sync.tgz --exclude='survival-simulator/results' --exclude='survival-simulator/.venv' --exclude='__pycache__' --exclude='survival-simulator/logs' survival-simulator
scp -q -o ConnectTimeout=10 /tmp/survsim-sync.tgz mypc:C:/Users/oscar/survsim-sync.tgz
ssh -o ConnectTimeout=10 -o BatchMode=yes mypc "powershell -NoProfile -ExecutionPolicy Bypass -Command \"tar -xzf C:\Users\oscar\survsim-sync.tgz -C C:\Users\oscar\NordicCupAI; Set-Location C:\Users\oscar\NordicCupAI\survival-simulator; \$env:PYGAME_HIDE_SUPPORT_PROMPT='1'; \$env:SDL_VIDEODRIVER='dummy'; .venv\Scripts\python.exe scripts\trapper\batch.py $ARGS --workers 20 --label $LABEL 2>&1 | Select-String -NotMatch 'Hello from'\""
mkdir -p "$ROOT/results/trapper/batches" "$ROOT/results/trapper/remote"
scp -q -o ConnectTimeout=10 "mypc:C:/Users/oscar/NordicCupAI/survival-simulator/results/trapper/batches/${LABEL}-*.json" "$ROOT/results/trapper/batches/"
scp -q -o ConnectTimeout=10 "mypc:C:/Users/oscar/NordicCupAI/survival-simulator/results/trapper/*-${LABEL}-r*-*.json" "$ROOT/results/trapper/remote/" 2>/dev/null || true
echo "results copied to $ROOT/results/trapper/{batches,remote}"
