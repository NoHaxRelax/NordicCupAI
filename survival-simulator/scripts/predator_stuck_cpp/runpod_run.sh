#!/usr/bin/env bash
# Run inside the unpacked source directory on a Runpod CPU Pod.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
test -x .venv/bin/python || uv venv --python python3 .venv
uv pip install --python .venv/bin/python -r scripts/predator_stuck_cpp/requirements.txt
.venv/bin/python scripts/predator_stuck_cpp/build.py
mkdir -p runs
.venv/bin/python scripts/predator_stuck_cpp/verify.py --seconds 65 --seeds 0 1 --output runs/parity-linux.json
.venv/bin/python scripts/predator_stuck_scan.py --backend cpp --diagnostics --games "${SCAN_GAMES:-1250}" --start-seed "${SCAN_START_SEED:-0}" --predators 100 \
  --seconds 600 --stuck-seconds 60 --radius 15 --workers "${SCAN_WORKERS:-32}" \
  --no-images --output runs/predator-stuck-diagnostics
touch runs/BATCH_COMPLETE
