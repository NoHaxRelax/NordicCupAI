#!/bin/bash
# Push the repo to DTU scratch (code only) or pull run artifacts back.
#   hpc/sync.sh up      code  -> $BLACKHOLE/nordic
#   hpc/sync.sh down    runs  <- $BLACKHOLE/nordic/runs  (into ./runs/hpc/)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="dtu:/dtu/blackhole/1e/205502/nordic"  # literal: $BLACKHOLE is only set in login shells
case "${1:-up}" in
  up)
    rsync -az --delete \
      --exclude .git --exclude runs --exclude __pycache__ --exclude '*.pyc' \
      --exclude .impeccable --exclude .venv \
      "$HERE/" "$REMOTE/"
    echo "synced code -> $REMOTE" ;;
  down)
    mkdir -p "$HERE/runs/hpc"
    rsync -az --include '*/' --include 'best.pt' --include 'last.pt' \
      --include 'metrics.jsonl' --include 'config.json' --exclude '*' \
      "$REMOTE/runs/" "$HERE/runs/hpc/"
    echo "pulled runs -> $HERE/runs/hpc" ;;
  *) echo "usage: $0 up|down"; exit 2 ;;
esac
