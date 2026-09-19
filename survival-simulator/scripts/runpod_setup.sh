#!/usr/bin/env bash
# Setup only. No simulations, optimizer runs, or cloud resource creation.
set -euo pipefail
SIM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_WORKSPACE="${TASK_WORKSPACE:-/workspace}"
TASK_ENV="${TASK_ENV:-$TASK_WORKSPACE/predator-search-venv}"
export UV_CACHE_DIR="$TASK_WORKSPACE/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$TASK_WORKSPACE/.local/share/uv/python"
mkdir -p "$TASK_WORKSPACE"

if command -v uv >/dev/null 2>&1; then
    TASK_UV="$(command -v uv)"
else
    # Fresh CPU images need python3 + python3-venv. Bootstrap stays on the volume.
    python3 -m venv "$TASK_WORKSPACE/.uv-bootstrap"
    "$TASK_WORKSPACE/.uv-bootstrap/bin/python" -m pip install --disable-pip-version-check 'uv==0.8.22'
    TASK_UV="$TASK_WORKSPACE/.uv-bootstrap/bin/uv"
fi

if [[ ! -x "$TASK_ENV/bin/python" ]]; then
    "$TASK_UV" venv --python 3.12 "$TASK_ENV"
fi
"$TASK_UV" pip install --python "$TASK_ENV/bin/python" --only-binary :all: -r "$SIM_ROOT/requirements.txt"
if ! command -v "${CXX:-c++}" >/dev/null 2>&1; then
    printf '%s\n' 'A C++17 compiler is required; install build-essential in this image.' >&2
    exit 1
fi
"$TASK_ENV/bin/python" "$SIM_ROOT/fastsim/build.py"
printf '%s\n' "Environment ready: $TASK_ENV" 'No search has been started.'
