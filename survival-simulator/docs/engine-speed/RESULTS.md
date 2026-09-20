# Exact native-engine speed pass

Branch: `codex/engine-speed`  
Baseline: `f31f586`

The requested 2.5x engine-only gain was not reached. The retained changes give a
smaller exact speedup while preserving the Python reference trajectory.

## Retained changes

- Batch object distance and angle kernels once per observer instead of once per
  object type, while restoring the original fruit/agent/predator/tree order.
- Deduplicate identical obstacle corners before generating visibility rays.
- Reject ray/edge pairs whose finite segment bounding boxes cannot intersect.
- Avoid the second intersection division except near the `u == 1` rounding
  boundary; uncertain cases still execute the original division and comparisons.
- Reuse cached neighborhood sets without copying and deleting the observing agent;
  the observation iterator already skips that agent in the same table position.
- Replace linear agent-ID lookup and observation hash lookup with ID-indexed vectors.
- Reuse direction/cone trigonometry across the four object categories.

## Timing

Workload: expanded-food baseline, seed 19001, predators enabled, native policy and
engine in the same C++ loop. Initialization is excluded. The short comparison used
the first 300 simulated seconds (3,000 ticks), five warm repetitions on the same
laptop. Median engine attribution was approximately:

| Build | Engine time per tick |
| --- | ---: |
| `f31f586` | 242 us |
| optimized | 196 us |
| speedup | 1.23x |

A simultaneous full-game comparison to extinction (11,524 ticks) measured 257.5
us/tick for the baseline and 230.7 us/tick for the optimized engine, or 1.12x.
The simultaneous run reduces frequency/load bias but is still a single paired run.
Both builds ended at score `1178.9597461384703` and time
`1152.4000000000474`.

Temporary phase instrumentation showed that observation generation consumed about
85% of engine time and static-wall visibility about two thirds of engine time.
Only 534 of 333,863 observed poses repeated exactly, so visibility-polygon caching
does not help this workload. PGO, `-O3`, `-march=native`, LTO, and a per-ray sorted
wall index were measured and rejected because they were neutral or slower.

## Exactness checks

- Random actions: seeds 1-10, 300-second requested horizon, two starting predators,
  full state/RNG comparison every tick: all passed. Several runs naturally ended
  before the requested horizon.
- Orchard policy: seeds 1-3, 300 seconds, two starting predators, full state/RNG
  comparison every tick: all 9,003 ticks passed.
- A final post-cleanup Orchard seed-1 repeat also passed all 3,001 ticks.

The checks compare returned observations, ordered world state, RNG state, events,
score, deaths and spawns bit for bit using `fastsim/verify.py`.

Reaching 2.5x on one core now requires a more fundamental visibility algorithm or
SIMD implementation. Parallelizing observers could reduce one game's latency, but
would consume cores currently used to run independent games and is therefore
unlikely to improve BO throughput.
