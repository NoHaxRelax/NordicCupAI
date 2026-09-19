# fastsim: native survival simulator engine

Integrated here from `survival-simulator/oscar-fastsim` at `696bbd86`.
Local adaptations add Windows/MinGW linking, build provenance, and optional
evaluator-only energy/birth/fruit/death accounting. See
[integration and verification](../docs/lucas_fastsim_integration.md).
The research supervisor uses this backend for focused/BO screening; promotion
comparisons and final holdout still use the unmodified Python reference engine.

C++ port of the survival simulator in `survival-simulator/src` (upstream commit acfc31a) exposed as a CPython
extension with the same `SimulationCore` interface. Given the same seed and the
same actions it aims to reproduce Python observations, scores, deaths and the
random-number stream under the matched object-ordering condition below. It is
not a promise of the identical trajectory of an arbitrary stock Python run.

## Build

    cd survival-simulator
    python fastsim/build.py        # needs a C++17 compiler, Python headers and numpy

No other dependencies. Build it on each machine (laptop, pods); the `.so` is
platform-specific (`.pyd` on Windows). The build creates `build-info.json` with
source/binary hashes and compiler/Python/NumPy provenance. Rebuild after source,
Python or NumPy changes. Research snapshots copy and verify the local binary;
portable source bundles omit generated binaries and build metadata.

## Use

    from fastsim import SimulationCore
    sim = SimulationCore(seed=1)                    # same defaults as src.core
    sim = SimulationCore(seed=1, predators=False)   # = harness_np's no-op spawn_predator
    state = sim.step(actions)                       # same dict as step_environment
    sim.env.agents / fruits / trees / predators     # snapshots, read-only
    sim.pop_events()                                # deaths and eaten fruit since last call

`fastsim` only needs numpy at runtime, plus `src` (pydantic, pygame, shapely,
scipy) for `verify.py`/`bench.py`. In Oscar's research workspace the harness
switch is `SURVIVAL_ENGINE=fast`, which makes `research/society/harness.py`
(and `harness_np`/`sweep`/`opt`) use fastsim with unchanged result JSON.
`policy/orchard_ref.py` + `policy/best-config.json` are a frozen copy of the
tuned orchard policy (branch survival-simulator/oscar-orchard-population),
used by `verify.py --policy orchard` and `bench.py`.

## What "identical" means

The Python engine iterates sets of Agent/Fruit/Tree/Predator objects in
several places (neighbourhood lookups, observation order, which fruit is eaten
first). Those objects hash by memory address, so the Python engine's own runs
are not reproducible even with a fixed seed. The native engine replays
CPython's set algorithm exactly, with a creation counter standing in for the
address. The result is one of the orderings the Python engine can produce, and
it is the same on every run.

`fastsim/verify.py` gives the Python classes that same counter hash (nothing
else changes), runs both engines in lockstep on identical actions, and compares
every returned state, the full world state and the RNG state:

    python fastsim/verify.py --seeds 1 2 3 --horizon 300 --starting-predators 2   # random actions
    python fastsim/verify.py --seeds 1 --horizon 3000 --policy orchard --no-predators --check-every 10

Other details reproduced:
- CPython's Mersenne Twister, including the ~1.9M draws the biome renderer spends
  at start-up.
- numpy's own float64 loops for sin/cos/arctan2/hypot, called through the ufunc
  objects. numpy uses SVML `arctan2` on AVX-512 CPUs, which differs from libm.
- Python float `%` and `//` semantics.
- GEOS ray-crossing point-in-polygon (shapely `contains`) with GEOS's
  double-double orientation test.
- Skipped iterations when agents, fruits or trees are removed mid-loop.
- int vs float types of agent traits in the state dict.

The build uses `-ffp-contract=off`, so no multiply-add is fused.

Results depend on the CPU the same way the Python engine's do. A run recorded
on the Mac is not reproduced on an AVX-512 pod by either engine, because numpy's
`arctan2` differs between them.

## Speed

Engine step only, orchard policy actions (about 20 agents), M4 laptop: Python
about 5.6 ms/step, native about 0.10 ms/step. End-to-end orchard runs are then
dominated by the policy itself (about 8 ms/step).

## Native policy (`_policy` module)

The engine port left the *policy* in Python, and a measured full game then spends
92% of its wall time there. `fastsim/_policy.cp312-win_amd64.pyd` is a second
extension that carries the engine **and** a C++ port of
`models/orchard_evasion_policy.py` (vendored orchard foraging/population core from
survival-simulator/oscar-fastsim `51ca0680`, plus our own-sighting evasion layer),
so a decision is a direct C++ call and a whole game is one Python call.

    from fastsim.fastpolicy import PolicySimulationCore
    sim = PolicySimulationCore(seed=1, predators=True)
    sim.policy_init(1, orchard_kwargs(defaults()))   # OrchardEvasionPolicy keywords
    steps, peak, ns_iface, ns_policy, ns_engine = sim._engine.run_policy(3000., 3000.)

`_engine.cpp`, `_engine*.pyd` and `build-info.json` are **not** touched: `_policy.cpp`
`#include`s `_engine.cpp` verbatim and builds to its own binary with its own
`build-info-policy.json`, so the engine's recorded provenance stays valid.

    python fastsim/build_policy.py      # builds _policy<EXT_SUFFIX>
    python fastsim/check_boundary.py    # the observation-only guarantee
    python fastsim/verify_evasion.py --seeds 1 2 3 --horizon 3000
    python fastsim/bench_policy.py --seed 1 --horizon 3000

### The observation-only boundary

The Python policy worker is kept observation-only by process isolation,
`sanitize_states` and `_assert_no_simulator()`. Linking the policy into the engine
binary removes all three, so it is replaced by a compile-time boundary: the policy is
its own translation unit (`_orchard_policy.cpp`) compiled against `policy_abi.hpp`
alone - no `Python.h`, no numpy, no `_engine.cpp`, and the build does not even pass
it the Python include path. Engine internals, true coordinates, the world seed,
fruit/tree ages, predator state and the engine RNG are undeclared there, so touching
one is a compile error. `check_boundary.py` checks the include list, the identifiers,
the published field list, and the compiled object's undefined symbols.

### Interface

`run_policy` keeps the whole tick loop in C++ with the GIL released. Observations
cross as a pointer to the engine's own `Obs` vector (layout `static_assert`ed against
`polabi::Obs`), agent state as a POD array reused across ticks, actions as a packed
struct: no JSON, no dicts, no Python objects, no per-tick allocation. Measured at
0.02% of native wall time.

### Numbers (this laptop, seed 1, full game to extinction at t=2143.9, 21438 ticks)

| run | wall | note |
|---|---|---|
| Python policy, fastsim engine, per tick | 100.1 s | policy = 92.1% |
| native policy, driven per tick from Python | 25.6 s | same decisions, Python boundary |
| native policy, `run_policy` | 20.9 s | **4.8x end to end** |

Native split: policy 76.0%, engine 23.9%, interface 0.02%. Within the policy,
`observe` is ~62% and its mark-matching loop alone ~46%; see "Known slowness" below.

### Known slowness

The native policy is only about 5x faster than the Python one, not the 20-50x a port
usually gives. Profiling (`policy_profile(True)` then `policy_phases()`) puts 46% of
it in the visual-odometry mark-matching loop in `observe()`, which is
|marks| x |prev_marks| exact CPython `math.dist` calls. Pruning it with the existing
`dist_cmp` already took the full game from 32 s to 21 s. Breaking the remaining
quadratic (a sorted band over `prev_marks`) is the obvious next step; it must keep
the original iteration order so ties still resolve to the same mark.
