#!/usr/bin/env bash
# Recommended Pod: CPU5 compute optimized (cpu5c), 32 vCPU, 64 GB RAM, no GPU.
# Catalog quote 2026-09-18: $0.035/vCPU/h = $1.12/h. Check at deployment.
# This uses an already provisioned Pod. It never creates or stops cloud resources.
set -euo pipefail
SIM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_WORKSPACE="${TASK_WORKSPACE:-/workspace}"
TASK_ENV="${TASK_ENV:-$TASK_WORKSPACE/predator-search-venv}"
TASK_OUT="${TASK_OUT:-$TASK_WORKSPACE/predator-search}"
if [[ ! -x "$TASK_ENV/bin/python" ]]; then
    printf '%s\n' 'Run bash scripts/runpod_setup.sh first.' >&2
    exit 1
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
exec "$TASK_ENV/bin/python" -u "$SIM_ROOT/scripts/runpod_night.py" \
    --out "$TASK_OUT" --budget "${SEARCH_BUDGET_USD:-25}" \
    --reserve "${SEARCH_RESERVE_USD:-5}" --hourly-rate "${POD_HOURLY_USD:-1.12}" \
    --max-hours "${SEARCH_MAX_HOURS:-9}" --workers "${SEARCH_WORKERS:-0}" "$@"
