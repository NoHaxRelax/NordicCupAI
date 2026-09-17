# Experiments

These produced every number in [`../README.md`](../README.md). None of them are
needed at runtime — `trap_sites.py` is the only file the controller imports.

All of them run the unmodified engine. Predator AI, energy, rest cycles, sensing
and collision handling are never patched. What the scenes *do* control is the
scaffolding around the measurement: bait energy and ageing are held constant so a
starving agent cannot be confused with a caught one, and unrelated tree, fruit
and predator spawning is switched off.

Two shortcuts are taken for speed, neither of which touches physics: the
per-pixel terrain *render* is skipped (it is pure drawing, and costs ~384k
`set_at` calls per scene), and the sweeps fill the biome map with grassland so
movement and drain modifiers do not confound the geometry. Grassland is the
worst case — every other biome slows the predator down, which only helps the
trap. `verify_sites.py` uses the real generated biome map.

| Script | Question it answers |
| --- | --- |
| `wall_geometry.py` | How thin, how long, and how close does a rock have to be? |
| `slot_geometry.py` | How wide a gap, and how deep must the agent sit? |
| `verify_sites.py` | Does every site the scanner reports actually hold a predator? |
| `coverage_scan.py` | How often does a generated map contain no usable site? |

## wall_geometry.py

Drops a predator already pinned against the far face of a parameterised rock —
the state a lure delivers it into — and runs from there. Separates the two
failure modes: *never engaged* (an acquisition problem) from *escaped* (the
geometry failed).

```console
python experiments/wall_geometry.py --mode width    # the thickness cliff
python experiments/wall_geometry.py --mode margin2  # rock extent either side
python experiments/wall_geometry.py --mode gap      # bait standoff vs thickness
```

## slot_geometry.py

Builds a corridor between two rocks and puts an agent in it. Reports the
clearance (distance to the nearest predator-legal point) alongside the outcome,
which is what showed clearance to be the only variable that matters.

```console
python experiments/slot_geometry.py --mode gap      # the 10..20 window
python experiments/slot_geometry.py --mode depth    # trap vs shelter
python experiments/slot_geometry.py --mode length   # corridor length
python experiments/slot_geometry.py --mode stress   # 648 randomised approaches
```

## verify_sites.py

The one that matters. Generates real maps, asks `trap_sites.find_trap_sites`
for sites, then replays each one with a real predator and checks the agent
survives and the predator stays.

```console
python experiments/verify_sites.py --maps 50 --steps 3000  --safety measured
python experiments/verify_sites.py --maps 30 --steps 30000 --safety safe
python experiments/verify_sites.py --maps 45 --kinds slot
```

## coverage_scan.py

Samples obstacle sets from the same distribution as `create_environment`
(80 rocks, `w,h ~ U(30,100)`, plus the four boundary walls) and counts sites per
map for all three presets. Slow — it runs the full scan three times per map.

```console
python experiments/coverage_scan.py --maps 2000 --workers 5
```

## A note on `--workers`

Default 5. Each worker loads numpy, scipy and pygame; 20 of them exhausted the
Windows page file on a 20-core machine, so raise it carefully.
