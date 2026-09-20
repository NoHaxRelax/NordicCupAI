#!/usr/bin/env bash
# Run from an isolated copy of survival-simulator on a Linux Runpod pod.
set -euo pipefail
: "${SEED_SEARCH_THREADS:?Set this to the allocated vCPU count reported by Runpod}"
: "${SEED_SEARCH_POD_ID:?Set this to the pod id}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
export CARGO_HOME="$PWD/.cargo" RUSTUP_HOME="$PWD/.rustup"
export PATH="$CARGO_HOME/bin:$PATH"
if ! test -x "$CARGO_HOME/bin/rustc"; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/seed-rustup.sh
    sh /tmp/seed-rustup.sh -y --no-modify-path --profile minimal --default-toolchain 1.98.1
fi
if ! command -v uv >/dev/null; then
    python3 -m pip install --break-system-packages uv==0.8.22
fi
uv venv --python 3.12.12 .venv
uv pip install --python .venv/bin/python numpy==2.3.5 scipy==1.16.3 shapely==2.1.2 pygame==2.6.1 pydantic==2.12.4
cargo test --locked --manifest-path scripts/seed_search_rust/Cargo.toml
cargo build --release --locked --manifest-path scripts/seed_search_rust/Cargo.toml
.venv/bin/python scripts/seed_search_rust/tests/verify_surveys.py
.venv/bin/python scripts/seed_search_rust/tests/verify_evidence.py
mkdir -p runs/seed-search-runpod
.venv/bin/python scripts/seed_search_rust/tests/runpod_benchmark.py \
    --threads "$SEED_SEARCH_THREADS" --pod-id "$SEED_SEARCH_POD_ID" \
    --output runs/seed-search-runpod/benchmark.json
touch runs/seed-search-runpod/COMPLETE
