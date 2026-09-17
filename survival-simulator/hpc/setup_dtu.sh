#!/bin/bash
# Run once from survival-simulator on a DTU login node; no simulations run here.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# DTU documents this Python generation in its current Python job example.
# Override with an available Python >=3.11 module if needed:
#   DTU_PYTHON_MODULE=python3/<version> bash hpc/setup_dtu.sh
DTU_PYTHON_MODULE="${DTU_PYTHON_MODULE:-python3/3.11.7}"
module load "$DTU_PYTHON_MODULE"
python3 -c 'import sys; assert sys.version_info >= (3, 11), "Python >=3.11 is required"'
python3 -m venv .venv-hpc
.venv-hpc/bin/python -m pip install -r requirements-tuning.txt
printf '%s\n' "$DTU_PYTHON_MODULE" > .venv-hpc/dtu-python-module.txt
.venv-hpc/bin/python -m pip freeze > .venv-hpc/installed-packages.txt
printf '%s\n' 'Environment ready. Submit with: bsub < hpc/submit_bo.sh'
