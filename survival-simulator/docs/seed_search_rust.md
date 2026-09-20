# Rust seed search

Implemented 19 September 2026; evidence collection extended 20 September. The native scanner searches candidate integers using observed biome samples. It reproduces CPython's seeded biome generator and rejects incompatible seeds before generating expensive worlds. A Python runner verifies survivors against founder headings, rock dimensions and every recorded public frame. New surveys also check reconstructed spawn positions/distances, paired rock dimensions and coordinates, observed biome labels including rivers, and initial tree sightings. See [the evidence implementation and validation](seed_evidence_2026-09-20.md).

The implementation is in [scripts/seed_search_rust](../scripts/seed_search_rust/); the runner is [search_survey_rust.py](../scripts/search_survey_rust.py). It consumes the existing [30-second survey format](seed_recovery_survey_2026-09-19.md) or standalone public samples. Survey collection and native world replay remain in Python. The Rust portion is the parallel seed search.

## Measured result

A subsequent [Runpod run](seed_search_runpod_2026-09-19.md) used all nine allocated CPU threads: **8.63 seconds median for ten million seeds**, and **12.52 seconds for search plus exact replay of a fresh Linux survey**. The saved Windows biome signature matched on Linux, but its exact replay hashes did not; see the report's platform-boundary findings.

On this machine, an Intel Core i7-13800H with 20 logical CPUs, using Rust 1.98.1, a release build and the held-out seed 78431's 64 observed biome samples:

| Implementation | Candidates checked | Threads | Prefix search time | Survivors |
| --- | ---: | ---: | ---: | --- |
| Earlier Python measurement | 1,000,000 | 1 | 22.49 s | 78431 |
| Rust | 1,000,000 | 1 | 7.28 s | 78431 |
| Rust | **10,000,000** | **20** | **6.96 s** | **78431** |

The ten-million run actually checked every seed from 0 through 9,999,999. Search plus Python startup/imports, native world initialization and verification took **15.14 seconds**. The survivor matched all five recovered founder headings, the observed rock dimensions and **all 300 public frames** under the stored actions. This timing begins with an already collected survey; add the collection period separately. Thirty simulated seconds need not equal thirty wall-clock seconds in a local accelerated simulation.

Single-thread throughput was about three times the earlier Python measurement. The larger run also benefits from CPU parallelism. These are individual local measurements on one survey, not latency guarantees for other machines, maps or weaker evidence. The full 2^32 range was not scanned.

Measurements, input/binary/source hashes and native verification results: [seed_search_rust_benchmark_2026-09-19.json](seed_search_rust_benchmark_2026-09-19.json). The earlier reference is [seed_survey_million_benchmark_2026-09-19.json](seed_survey_million_benchmark_2026-09-19.json).

## Run it

Commands below start in `survival-simulator`. The release binary is already built locally. Search a saved survey and verify its survivors:

```powershell
python scripts/search_survey_rust.py --input docs/seed_survey_heldout_2026-09-19.json --count 10000000 --verify --output docs/my_seed_search.json
```

Choose a new output filename on each run. Omit `--output` for JSON on stdout. The runner selects the release binary automatically; `--binary PATH` overrides it. A survey containing multiple test cases requires `--target N` to select the record. This record identifier is never used to narrow the candidate range. `--checkpoint 10` selects the ten-second biome snapshot; verification still checks the full recorded action log.

Useful options shared with the Rust executable:

- `--start N --count N`: search the half-open interval `[N, N + count)`, within unsigned 32-bit seeds. Defaults: start 0, count 10,000,000.
- `--threads N`: 1 through 256, default available CPU threads. Fewer threads can leave CPU capacity for a live policy.
- `--max-results N`: retain at most N candidates, default 100,000. The full interval is still scanned, and the exact survivor count is reported even when the returned list is truncated.

The runner's `--verify` checks all survivors only if the returned list is complete and at most `--max-native-worlds` candidates remain (default 20). Exceeding either cap preserves the search result, marks verification incomplete and returns exit code 2. It never labels a truncated or unchecked list as a uniquely verified seed. A successful complete search with no survivors is a valid result for that range.

For a direct Rust invocation on Windows:

```powershell
scripts/seed_search_rust/target/release/seed-search.exe --input docs/seed_survey_heldout_2026-09-19.json --count 10000000
```

On Linux/macOS the binary has the same path without `.exe` (build it on that platform). Direct Rust output contains prefix candidates; use the runner for native verification.

## Supply a shared-map snapshot

The Rust executable also accepts a small JSON document with no known seed or simulator audit fields:

```json
{
  "samples": [
    {"x": 100.25, "y": 200.5, "biome": "forest", "radius": 7.0},
    {"x": 850.0, "y": 420.0, "biome": "grassland", "radius": 8.5}
  ]
}
```

Use world coordinates from anchored shared-map groups. The existing `seed_survey_probe.shared_samples(estimator)` function extracts and selects eligible observations, adding the uncertainty radius. Its first return value can be serialized as `{"samples": samples}`. Export this snapshot after the planned survey and run the scanner; standalone sample files do not contain enough information for `--verify`.

Accepted land labels are `forest`, `swamp`, `desert`, and `grassland`. River observations are ignored because the river overwrites the base land map. Empty/all-river input fails with an explanation: wait for anchored land observations before starting the scan. Nonfinite values, negative radii, unknown labels, invalid ranges and excessive thread counts are rejected.

The filter uses the same necessary condition as the Python reference:

```text
distance to nearest site of observed biome
    <= distance to nearest site of any biome + 2 * sample radius
```

It preserves the existing floating-point tolerance. It uses actual observed labels, not predicted biomes. Radii remain heuristic localization bounds: a bad map alignment can eliminate the real seed. Candidates require final verification; one surviving biome prefix alone is insufficient proof.

## Build and validate

With a standard Rust toolchain on PATH:

```powershell
cargo build --release --locked --manifest-path scripts/seed_search_rust/Cargo.toml
cargo test --locked --manifest-path scripts/seed_search_rust/Cargo.toml
cargo clippy --locked --all-targets --manifest-path scripts/seed_search_rust/Cargo.toml -- -D warnings
python scripts/seed_search_rust/tests/verify_surveys.py
python scripts/seed_search_rust/tests/verify_evidence.py
```

This workspace uses a project-local GNU Windows toolchain. To use that installation in a new PowerShell session, first set these process-local variables:

```powershell
$env:CARGO_HOME = Join-Path (Get-Location) 'scripts/seed_search_rust/.toolchain/cargo'
$env:RUSTUP_HOME = Join-Path (Get-Location) 'scripts/seed_search_rust/.toolchain/rustup'
$env:PATH = "$env:CARGO_HOME/bin;$env:PATH"
```

The toolchain and build directory are ignored by Git. `Cargo.lock` pins dependencies. The scanner itself needs no Python runtime; native verification needs the simulator's Python environment and dependencies.

Validation completed:

- Seven Rust tests covering CPython output across multiple MT twist cycles, biome prefixes for 32 seeds including both unsigned boundaries, Python filter parity, uncertainty bounds, checkpoint selection, invalid ranges and deterministic results across thread counts/output caps.
- Five Python integration tests comparing the four saved final surveys, the held-out ten-second checkpoint, raw public-sample input without target metadata, CLI failure paths and verification caps.
- Clippy with warnings denied and a successful release build.
- A completed ten-million scan followed by exact native replay of the sole survivor.

The Python-generated test vectors are checked in. Regenerate them with `python scripts/seed_search_rust/tests/generate_python_vectors.py`. Reproduce the benchmark, including native verification, with:

```powershell
python scripts/seed_search_rust/tests/benchmark.py --output docs/my_rust_benchmark.json
```

## Compatibility and scope

This version targets the native 1600 x 1200 map with ten biome sites and nonnegative integer seeds below 2^32. Larger/negative Python integers, alternate world dimensions, different type order or a changed map generator require corresponding changes and new compatibility vectors.

The RNG uses CPython's one-word integer `init_by_array` seeding and its `getrandbits` rejection rules. A generic MT19937 seed initializer or modulo-based range selection would generate different maps. It precomputes the constant initialization pass, twists words on demand, checks required biome labels early and distributes contiguous seed ranges across workers. The implementation follows the algorithms in [CPython's random C module](https://github.com/python/cpython/blob/v3.12.12/Modules/_randommodule.c) and [Python random.py](https://github.com/python/cpython/blob/v3.12.12/Lib/random.py); independent CPython vectors check compatibility.

This is efficient bounded enumeration, not algebraic seed inversion or a precomputed seed catalog. Rocks, spawn coordinates and headings are expensive to reproduce because the native generator advances the same RNG through terrain rendering before generating them. They remain in native candidate verification, before dynamic observations are checked by replay. Better public geometry now supplies tighter biome samples and transition brackets to the Rust scan; it does not eliminate seed enumeration. Live policy integration, background cancellation and automatic use of the recovered full map are separate work from this offline search backend.
