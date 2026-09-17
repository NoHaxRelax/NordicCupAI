#!/bin/bash
# Central DTU HPC / LSF 10. Submit from survival-simulator:
#   bsub < hpc/submit_bo.sh
# 24 CPU slots on one node; 4 GB per slot = 96 GB reserved RAM. No GPU.
# Any eligible hpc CPU model is acceptable; avoiding a model constraint gives
# the scheduler more options among DTU's 24-, 32- and 48-core Intel nodes.
#BSUB -J survival_bo
#BSUB -q hpc
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -W 12:00
#BSUB -o bo_%J.out
#BSUB -e bo_%J.err

set -euo pipefail
cd "${LS_SUBCWD:?Submit this script using bsub from survival-simulator}"
if [[ ! -f tune_policy.py || ! -f .venv-hpc/dtu-python-module.txt ]]; then
    printf '%s\n' 'Run bash hpc/setup_dtu.sh in survival-simulator before submitting.' >&2
    exit 1
fi
module load "$(cat .venv-hpc/dtu-python-module.txt)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1

# Persistent shared storage is intentional: episodes and Optuna's journal
# survive allocation termination. A single coordinator writes the journal.
# Re-submit this same script to resume. Never run two jobs in the same output.
# The first 8 hours search; the remaining 3.5 hours validate. A 30-minute
# margin before LSF's limit lets cooperative timeouts finish and save results.
exec .venv-hpc/bin/python -u tune_policy.py \
    --output runs/bo-dtu \
    --workers "${LSB_DJOB_NUMPROC:?LSF CPU allocation missing}" \
    --trials 300 \
    --startup-trials 24 \
    --search-hours 8 \
    --total-hours 11.5 \
    --train-seconds 600 \
    --train-seeds 1 7 42 \
    --validation-seconds 3000 \
    --validation-seeds 1001 1007 1042 \
    --finalists 7
