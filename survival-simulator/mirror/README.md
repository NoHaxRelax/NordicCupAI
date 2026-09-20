# fastsim._mirror — synchronized shadow model primitives

20 September 2026. Adds `clone()`, `snapshot()` and `restore()` to the fastsim engine so a
policy that knows the world seed can run an exact private model and branch off it.

`fastsim/_engine.cpp` is **not modified**. This follows the arrangement `_policy.cpp`
already documents: `_engine.cpp` is a provenance-pinned artefact (`fastsim/build-info.json`
records its `source_sha256`), so it is `#include`d verbatim into its own translation unit,
which compiles to its own `_mirror<EXT_SUFFIX>`. `import fastsim` keeps using `_engine`
exactly as before. The three methods are attached to the ready type as descriptors in the
module init rather than by editing the included `Engine_methods[]` table.

## Why this was the blocker

`class Engine` (`../fastsim/_engine.cpp:539`) has only value members — `std::vector`,
`std::unordered_map`, and `PySetEmu{std::vector<Entry>, …}` — with no raw pointers, no
references, and no user-declared copy constructor, copy assignment or destructor. **The
implicitly generated copy constructor was already a correct deep copy**, including the
624-word Mersenne Twister state. There was simply never a binding for it, so earlier
seed-aware work had to hand-write partial forecasts instead: `seed-aware-policy/bench/
predictive_safety.hpp:1-3` says outright that "births, meals and other predators' kills are
omitted from this short rollout."

## The three primitives

| method | what it does | cost |
|---|---|---|
| `clone()` | full deep copy; returns an independent `Engine` | **1.48 ms** |
| `snapshot()` | capture only the dynamic state | **0.002 ms** |
| `restore(snap)` | rewind this engine to a snapshot | **0.001 ms** |

**Use `snapshot`/`restore` for search; use `clone` only when you genuinely need two live
engines.** A clone copies ~2 MB of map data that is fixed at construction — the 1.92 MB
biome raster, obstacles, edges, the chunk grids over them, corner/edge cells. Search does
not need a second engine, it needs to rewind one, which is why snapshot+restore comes out
**476x cheaper**.

What a snapshot captures: `rng`, `score`, `time`, the three id counters, `agents`,
`predators`, `fruits`, `trees`, the event-log length and the diagnostics flag.

What it deliberately omits, and why that is safe:

- **Immutable after construction** — `biome`, `obstacles`, `edges`, `edge_set_order`,
  `grid_obstacles`, `grid_edges`, `local_*_cache`, `corner_cells`, `edge_cells`.
- **Rebuilt from the dirty flags** — `grid_agents/fruits/trees/predators`, `uc_*`,
  `key_to_index_*`. `restore` sets all four dirty flags, and `refresh_grids()`
  (`_engine.cpp:877`) rebuilds them, invalidates the union caches and reindexes keys.
- **Static-map-keyed caches that stay valid across a rewind** — `cell_edge_cand`,
  `radius_ids`. Keeping them warm is a benefit, not a leak, because they are a function of
  the map and a vision radius, neither of which a rewind changes.
- **Cleared on restore** — `diagnostics` only.

`agent_observations` **is** carried, and this is a subtlety worth knowing. It is written in
the agents loop (`_engine.cpp:1423`) and read by `build_state` (`:1662`), so it is what a
policy sees on the *first* tick after a restore. An earlier version of this file cleared it.
That did **not** break engine exactness — the engine's own evolution never reads it, so the
retrace test below still passed — but it left each search branch blind for one tick, i.e.
silently solving a different problem from the one being evaluated. If you add state to the
snapshot, ask which side of that line it falls on: *"does the engine read it"* is the wrong
question; *"does anything read it"* is the right one.

## Verification

`python mirror/test_clone.py`:

```
  seed          1: 2000 ticks bit-identical  (score 228.275291)
  seed          2: 2000 ticks bit-identical  (score 227.521284)
  seed          3: 2000 ticks bit-identical  (score 225.470569)
  seed  614466944: 2000 ticks bit-identical  (score 226.655809)
  independence: parent unchanged after clone ran 500 different ticks
  branch-and-replay: 400-tick log reproduced exactly from a snapshot
  seed          1: restore + 600-tick retrace bit-identical
  seed  614466944: restore + 600-tick retrace bit-identical
  restore recovers exactly after an unrelated 400-tick branch
ALL CLONE CHECKS PASSED
```

The fingerprint is a SHA-256 over the full RNG state, `info()`, and every agent, predator,
fruit and tree — not just the score, which would hide a divergence that has not yet
surfaced.

The tests that matter most are the last two. **Restore is checked to be reversible after
the engine has been driven down an unrelated branch**, which is exactly what a search does
and the only case where a missed cache would show up.

## Build and use

```sh
python mirror/build_mirror.py          # writes fastsim/_mirror<EXT_SUFFIX>
PYTHONPATH=$PWD python mirror/test_clone.py
```

Same flags as `fastsim/build.py`, including `-ffp-contract=off` and the disabled sin/cos
builtins — a shadow model that rounds differently from the engine it shadows is worse than
no shadow at all.

```python
import numpy as np
from fastsim import _mirror
_mirror.set_numpy_loops(np.sin, np.cos, np.arctan2, np.hypot)   # required, as for _engine

e = _mirror.Engine([seed], env_width=1600, env_height=1200)
snap = e.snapshot()
for candidate in plans:                 # evaluate each branch from the same state
    e.restore(snap)
    for acts in candidate:
        e.step(acts)                    # acts: [(agent_id, {...}), ...]
    score(e)
e.restore(snap)                         # back to the live trunk
```

`_mirror` has its own private copy of the engine, so it does **not** share state or the
numpy ufunc loop pointers with `_engine` or `_policy`. Call `set_numpy_loops` on each
module you import, the way `fastsim/__init__.py` does.

## Scope and limits

- This is the **model-side** primitive only. It does not make the policy seed-aware; it
  makes a seed-aware policy possible to write. Per the project's standing rule, the policy
  that consumes it has to be C++ as well — driving this from Python per tick would throw
  away the point.
- Exactness is verified **against this engine**. The live evaluation server is a different
  process on a different machine, so a shadow model still needs a resync tripwire:
  compare forecast tree/fruit/predator spawns against the incoming public observations and
  fall back when they disagree. Note `fastsim/README.md:42-56` on CPython set iteration
  order being address-based.
- Costs above were measured on an i7-13800H, Windows, MinGW g++ 13.2, at a mid-game state
  with 6 predators, 84 trees and 299 fruits. `snapshot`/`restore` scale with the live
  entity count; `clone` is roughly constant because the map dominates.
