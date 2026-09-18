# Rust simulation engine

The game runtime is implemented in Rust under `rust/src/`, exposed as the
`survival-native` Python extension. Existing `SimulationCore`, `Environment`,
policies, servers, and pygame viewers retain their interfaces and live Python
entity objects.

## Install and run

Install a stable Rust toolchain (including Cargo) and the existing Python 3.13+
dependencies. From the repository root:

```sh
uv sync --extra rust
uv run --extra rust python survival-simulator/simulation_server.py
```

For an existing virtual environment, installing the extension directly also works:

```sh
uv pip install ./survival-simulator/rust
```

Builds use the optimized release profile. Reinstall after changing Rust sources.
Do not use `uv sync` without `--extra rust` if you intend to keep the optional
engine installed; syncing without that extra removes the extension.

`SimulationCore(..., backend="auto")` (the default) selects Rust when installed
and Python otherwise. Select explicitly per instance:

```python
sim = SimulationCore(seed=42, backend="rust")
reference = SimulationCore(seed=42, backend="python")
```

Alternatively set `SURVIVAL_BACKEND=rust` or `SURVIVAL_BACKEND=python` for existing
scripts. Explicit Rust selection fails with installation instructions if the
extension is unavailable. `sim.backend` reports the selected engine; reset
retains that selection. Engines can coexist in one process.

## What runs in Rust

- Actions, movement limits and energy costs, biome movement, collision sliding,
  bounds, and turning.
- Visibility ray intersections, polygon construction, observation assembly,
  and predator behavior.
- Tick sequencing, eating, aging, deaths, reproduction and trait mutations,
  fruit growth/rot, tree growth/death, and entity spawning.
- Dynamic spatial-grid rebuilding and neighboring-object collection.

Python retains world/map generation, static obstacle construction, pygame
rendering, entity constructors and storage, and the public adapter/API. NumPy's
transcendental functions and Shapely's strict polygon containment are used by
both engines. This is a Rust runtime with Python interoperability, not a
standalone Rust executable. It retains the GIL; it does not add multithreading.

The compatibility boundary is deliberate: replacing the random generator,
floating-point kernels, or Python object sets changes the simulation. Rendering
the initial biome map also consumes the game RNG, so skipping it changes every
subsequent spawn. World construction is retained and excluded from tick-speed
measurements; its time is reported separately.

## Exactness and regression testing

```sh
.venv/bin/python -m unittest discover -s survival-simulator/scripts -p 'test_*.py' -v
.venv/bin/python survival-simulator/scripts/benchmark_rust_engine.py --ticks 300
```

`test_rust_engine.py` compares every entity's state, vision polygon, ordered
observations (including duplicate edge hits), score, time, IDs, grid membership,
and the full RNG state. Floating-point values are compared bit for bit, including
signed zero, with no epsilon. Coverage includes random trajectories, late-game
spawning, contact ties, all biome multipliers, collision sliding, mutation caps,
reproduction, sleeping predators, starvation, old age, fruit rot, and tree death.
Visibility is also compared independently across 150 randomized geometries.

The reference remains in the existing Python source. Each test restores the same
world and replays the same constructors using the same object identities before
running Rust. This is necessary because the original uses sets with address-based
entity hashes. Two independent Python worlds with the same seed are **not** a
reliable bit-identical oracle: observation order, ties, and accumulation order can
already differ. The port preserves the original order instead of introducing a
new sorting rule.

Existing quirks are retained, including live-list removal skipping the next
entity, observations preceding fruit consumption and predator movement, and
failed tree-placement attempts still reaching the biome spawn test. The original
empty-edge sensing error is retained too. Fixing these would be a separate rule
change. Parity is verified for the same Python/NumPy/Shapely runtime; this does not
promise cross-platform or cross-library-version floating-point reproducibility.

## Measured performance

Release build, Python 3.13.14, NumPy 2.5.3, Shapely 2.1.2, Linux x86-64.
Three 300-tick runs on 1600×1200 maps, 31 initial agents and 30 predator spawn
attempts. Initialization can reject predator positions, so actual initial counts
are 22, 23, and 21. Each timed tick is replayed on identical state and actions;
snapshotting and exact comparisons happen outside the timed regions. Timings
include observations, the Python API boundary, and constructor-replay hooks.

| Seed | Python ticks/s | Rust ticks/s | Speedup |
| --- | ---: | ---: | ---: |
| 0 | 114.2 | 264.9 | 2.32× |
| 1 | 100.5 | 226.2 | 2.25× |
| 17 | 104.7 | 230.6 | 2.20× |

All 900 benchmark ticks matched exactly. These scripted actions produce varying
populations (mean 14.6–16.9 agents), so this is a workload-specific measurement,
not a universal speedup. The benchmark prints runtime versions, population
counts, setup time, total step time, throughput, and parity results as JSON.

For Rust development checks:

```sh
cargo fmt --manifest-path survival-simulator/rust/Cargo.toml --check
cargo clippy --manifest-path survival-simulator/rust/Cargo.toml -- -D warnings
```
