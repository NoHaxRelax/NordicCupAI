# Seed cracking V4 handover for Lucas

Prepared 20 September 2026 (UTC; exact packaging time in SOURCE_PROVENANCE.json).
This is the latest **live-tested seed-recovery checkpoint**, V4. V5 is excluded:
it compiled but never completed its integration test. This is a frozen, isolated
package; do not combine its engine, headers or acquisition code with another branch.
The build/preflight wrappers in this package are new handover tooling.

## What is proven

V4 recovered unknown live seed 614466944 in 19.18 seconds after dispatch, at
84.9 simulated seconds (132.03 seconds after the first callback). All six completed
shards scanned exactly 4,294,967,296 seeds and accepted one wall-verified seed.
The last complete mid-run audit matched all 5,598 public observations. The final
whole-game logs/replay were not retrieved, so **full final-game parity is not
proven**. The bounded V4 integration fixture passed 1,289 observations. Earlier
V1/V2 live games had complete 8,566/4,317 observation audits.

See evidence/coverage-audit.json, shard receipts, last-verified-snapshot.json and
integration-result.json. These are historical receipts, not newly run benchmarks.
No latency or universal reliability guarantee applies to Lucas's CPU.

## Rebuild on Lucas's machine

Use Linux or Ubuntu WSL, CPython 3.12 with development headers, NumPy 2.3.5,
g++, pkg-config, nlohmann-json3-dev and libcpp-httplib-dev. The historical live
runtime was Linux, Python 3.12.3, NumPy 2.3.5 on AMD EPYC 9654. Build on the actual
CPU: `-march=native` binaries are not portable. Preserve `-ffp-contract=off`;
never add `-ffast-math`. CPython embeds NumPy's native math loops even though the
simulation/search loops are C++.

From this package directory, on a Linux installation providing Python 3.12:

```sh
sudo apt-get install g++ pkg-config python3.12-dev python3.12-venv nlohmann-json3-dev libcpp-httplib-dev
python3.12 -m venv .venv
.venv/bin/python -m pip install numpy==2.3.5
.venv/bin/python preflight.py
.venv/bin/python build.py
export PYTHONPATH="$PWD/.venv/lib/python3.12/site-packages"
.venv/bin/python preflight.py --native
```

The explicit PYTHONPATH lets the embedded interpreter find the same NumPy used
at compilation. Positive and negative native checks exercise terrain filtering
and wall verification on one seed each. The positive must accept 1854492595;
the negative must accept nothing. These are known-answer checks, not a blind
search. The preflight also checks all packaged hashes and independently verifies
the historical public terrain samples using CPython's RNG.

A fresh build/native preflight was **not run on the packaging laptop**: it lacks
the target Linux toolchain, and heavy local computation is out of scope. Package
integrity, source closure, Python syntax, historical receipts and independent RNG
fixture checks were verified locally. Run the native checks on the target host
before changing a live endpoint.

## Full-range diagnostic on one machine

This bypasses distributed transport and searches the complete uint32 domain.
It can take much longer on a small CPU. The frozen coordinator uses 20 search
threads plus four verifier threads; its thread allocation is not configurable.
Use a new output directory for every invocation.

```sh
N="$PWD/survival/research/seed_inference"
"$N/streaming_verification/coordinator" stream \
  "$N/terrain_filter" "$PWD/fixtures/terrain.txt" \
  "$PWD/fixtures/initial-public.json" \
  "$PWD/survival/results/orchard/best-config.json" \
  0 4294967296 "$PWD/runs/full-domain-001"
```

The fixture comes from live game 1789879801738700113 and must retain/verify seed
1854492595. Inspect verified.txt, receipt.json and filter.log. Require
`complete=true`, `filter_exit_code=0`, and a final `tested=4294967296` filter
receipt. Do not confuse a candidate, a wall match, and verified replay parity.
The bundled fixture is a regression input, not a new unknown-seed benchmark.

## Live topology and CLI

**The server alone scans only shard 0, about one-sixth of the domain.** It does
not automatically fall back to a whole-domain search when workers are absent.
Use the full-range diagnostic above to isolate host reliability first.

V4 needs one server and five workers with distinct shard IDs 1, 2, 3, 4 and 5.
Each range is `[floor(2^32*i/6), floor(2^32*(i+1)/6))`. Each process's coordinator
uses 24 CPU threads; launching all six on a small machine will oversubscribe it.
The worker CLI's THREADS argument is legacy and does not alter the coordinator's
fixed 20+4 allocation. Do not assume it limits CPU use.

```text
paced_native_v4 serve OUT CONFIG ABSOLUTE_TERRAIN_FILTER TOKEN_FILE PORT
paced_native_v4 worker URL_FILE ABSOLUTE_TERRAIN_FILTER OUT 24 SHARD_ID
paced_native_v4 replay PACKETS_JSONL RECOVERED_SEED OUT CONFIG
```

Use absolute paths. CONFIG is the included survival/results/orchard/best-config.json.
Workers resolve that same config relative to the terrain-filter executable;
retain the package directory layout. URL_FILE contains the server's private
base URL including its token path and optionally `/predict`. Keep token and URL
files outside Git. Worker processes handle one assigned game and then exit;
start fresh workers for each game. Preserve separate outputs and all six receipts.
The server binds all interfaces; use authenticated private connectivity and do
not expose it until the intended endpoint setup is ready. This handover neither
deploys an endpoint nor authorizes competition API submissions.

Acquisition starts at 128 anchored samples across at least three biomes, or 400
samples across two biomes after 180 simulated seconds. Lack of informative
terrain can delay dispatch. The included freshness guard rejects stale pose
information; do not remove it or replace it with the older localization code.
V4 recognizes both horizontal and vertical boundaries. Capture full-precision
public JSON and exact ordered actions; rounded debugger frames are not inputs.

## Diagnose Lucas's failure by stage

1. **Build/startup fails:** verify Python headers/libpython, pinned NumPy,
   PYTHONPATH, cpp-httplib and the included public_freshness.hpp. Rebuild all
   binaries on this CPU. Never copy the old Runpod executables.
2. **No search begins:** inspect status samples/labels and the acquisition
   threshold above. This is distinct from a completed scan with no match.
3. **Search runs but misses seeds:** confirm all six distinct workers completed,
   or run the standalone full-domain command. Check child exit codes and tested
   counts, not just HTTP health. Preserve the real production terrain_filter;
   stream_integration_filter is intentionally bounded test code and must never
   be used in production. Check the frozen source hashes and full-precision
   coordinates, boundary localization and freshness handling.
4. **Wall match but model_failed:** preserve packets.jsonl, responses.jsonl,
   model checks and first mismatch. Check engine/config consistency, exact
   ordered action history, NumPy version and compiler floating-point flags.
   A correct seed does not guarantee numerical replay agreement.
5. **Works but too late:** separate observation acquisition, scan, verification,
   model catch-up and transport time. Six-second pacing and a 240-second extra
   wait budget are fixed here. A smaller CPU may not meet live timing limits.

The cause on Lucas's machine is not yet established. Useful evidence to return:
preflight output, compiler/Python/NumPy versions, CPU model, exact invocation,
search-input.json, all shard receipts/filter logs, verified.txt, and first model
mismatch if present. Exclude credentials and token-bearing URLs.

## Provenance and rollback

Source files are copied without algorithm changes. Engine/search/verifier files
match the preserved streaming benchmark staging tree; the acquisition config
comes from the preserved live deployment staging tree, not today's policy config.
The missing include is supplied from the actual staged freshness header. See
SOURCE_PROVENANCE.json for file origins and SHA256SUMS.json for exact content.
Keep this directory intact as the V4 rollback reference while diagnosing newer
implementations. This is separate from the older codex/seed-shadow implementation.
