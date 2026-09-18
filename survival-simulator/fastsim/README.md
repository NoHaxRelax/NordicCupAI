# fastsim: native survival simulator engine

C++ port of the survival simulator in `survival-simulator/src` (upstream commit acfc31a) exposed as a CPython
extension with the same `SimulationCore` interface. Given the same seed and the
same actions it produces the same observations, scores, deaths and random-number
stream as the Python engine, bit for bit.

## Build

    cd survival-simulator
    python fastsim/build.py        # needs a C++17 compiler, Python headers and numpy

No other dependencies. Build it on each machine (laptop, pods); the `.so` is
platform-specific.

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
