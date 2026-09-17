# Predator traps

Finding places on a survival-simulator map where an agent can stand and a
predator can never eat it — and, in most cases, where that predator gets stuck
permanently trying.

Drop this `traps/` folder anywhere inside your `survival-simulator` checkout.
The scripts locate `src/` themselves; no `PYTHONPATH` setup needed.

```console
cd survival-simulator/traps
python trap_sites.py --seed 3 --render map.png     # see the sites on a map
python -m unittest test_trap_sites                 # 25 tests
```

```python
from trap_sites import find_trap_sites

sites = find_trap_sites(env.obstacles, env.width, env.height)
best = sites[0]                       # sorted safest-first
bait_x, bait_y = best.bait            # park an agent exactly here
```

---

## 1. The idea

A predator in this simulator has three properties that combine into an
exploitable hole.

**It smells through walls.** `Creature.observe` adds every entity inside
`hearing_radius` to the observation list with no line-of-sight test at all —
occlusion is only applied to the vision cone
([`../src/elements/creature.py:131`](../src/elements/creature.py#L131)).
A predator's `hearing_radius` is **60**.

**It charges whatever it senses, without pathfinding.** `Predator.step` picks the
nearest observed agent and drives straight at it. The "only chase when behind"
condition is bypassed entirely whenever the agent is inside
`hearing_radius * 1.5` = 90 units, so a nearby agent is always charged, even one
staring right at it ([`../src/elements/predator.py:32-37`](../src/elements/predator.py#L32-L37)).

**Blocked moves rotate, they do not reroute.** `update_entity_position` keeps the
requested distance and rotates the heading in 10° steps until the destination is
clear ([`../src/elements/environment.py:543`](../src/elements/environment.py#L543)).
The predator slides along a rock face; it never plans a path around it.

Put an agent just out of reach but inside 60 units, and the predator spends the
rest of the game charging a wall. It burns energy, drops to zero, rests,
recharges at 30/s, wakes at 50%, and charges again — forever. A reference run
logged **236 rest cycles over 3000 simulated seconds**.

### Two kinds of site, with very different guarantees

|  | `wall` | `slot` |
| --- | --- | --- |
| Mechanism | Behavioural | Geometric |
| Setup | Bait across a thin rock | Agent in a gap too narrow for a predator |
| Why it holds | The predator *could* walk around, its AI won't | No legal predator position is within reach |
| Breaks if | The AI ever changes | Never |
| `site.guaranteed` | `False` | `True` |

Slots are strictly stronger and are ranked first. Wall traps still matter:
they work on lone rocks, which are far more common than narrow gaps.

---

## 2. Wall traps

Bait sits 5 units off the face of a thin rock; the predator pins itself against
the opposite face and oscillates there.

```
                 rock
   predator  |          |  bait
      O      |  <-- t ->|   o
             |          |
      <----- separation ----->
```

### Requirements

| Quantity | Limit | Why |
| --- | --- | --- |
| **separation** | ≤ **52.5** | The predator swings ~7.5 units while pinned. 52.5 + 7.5 = 60, its smell radius. Past that it loses the scent and wanders off. |
| **thickness** | ≤ **37.5** | The separation limit with the bait at its minimum 5-unit standoff: `t + 5 + 10 ≤ 52.5`. |
| **margin** | ≥ **32.5** either side | Less and the predator slides along the face and rounds the end. |

`separation = thickness + 5 (bait standoff) + 10 (predator radius)`.
The reference scene in `predator_trap.py` — a 30-wide rock — gives exactly 45.

### What was measured

Thickness sweep, **750 runs** (5 approach offsets × 2 rest phases × 3 seeds
per geometry, 3000 ticks each):

| thickness | 30 – 37.5 | 38.75 | 40+ |
| --- | --- | --- | --- |
| held | **30/30 at every step** | 0/30 | 0/30 |
| furthest the predator got | 52.8 → **59.88** | — | — |

At thickness 37.5 the predator swings to 59.88 against a 60-unit smell radius.
One more unit of rock and the trap is gone. The failure mode is *escape*, not
death: a too-thick rock loses you the predator, it does not feed it.

Margin sweep, **1080 runs**:

| thickness | margin needed |
| --- | --- |
| 30 | ≥ 30 |
| 32.5 – 37.5 | ≥ 32.5 |

Short rocks are the dangerous case. At margin 10, **12 of 30 runs ended with the
bait eaten**. 40 covers every thickness with room to spare.

Gap sweep, **600 runs**, confirmed the combined rule `thickness + standoff ≤ 42.5`
independently: a 30-thick rock tolerates a 10-unit standoff, a 35-thick rock does
not.

Predator randomness — its wander RNG, its rest-cycle phase, and where along the
face it arrives — changed nothing. Every failure was geometric.

---

## 3. Slot traps

Two rocks with a gap between them. The agent fits; the predator does not.

```
   rock A      gap g      rock B
 +--------+ .         . +--------+
 |        | .    o    . |        |
 |        | .   /|\   . |        |
 +--------+ .         . +--------+
            ^ mouth
     predator stuck out here
```

### Requirements

| Quantity | Limit | Why |
| --- | --- | --- |
| **gap** | ≥ **10** | Agent radius is 5, so a narrower gap has no legal standing point. |
| **gap** | < **20** | Predator radius is 10. From 20 up it walks straight in. |
| **clearance** | ≥ **15** | Capture needs distance < 15 (`predator.size + agent.size`), and every predator position clears rocks by 10. |

**Clearance** is the distance from the agent to the nearest position a predator
may legally occupy. It is the whole story.

### Clearance is the only number that matters

Gap sweep, **60 runs**:

| gap | outcome |
| --- | --- |
| 8 – 9 | agent does not fit |
| 10 – 19.5 | agent untouchable |
| ≥ 20 | **EATEN** — closest distance exactly 15.00, the touch threshold |

Stress sweep, **648 runs** (6 gap widths × 6 depths × 3 seeds × 6 approach
directions):

| clearance | outcome |
| --- | --- |
| 14 | **EATEN**, 18/18 at every gap width from 10 to 19 |
| ≥ 15 | **trapped**, 18/18 at every gap width from 10 to 19 |

Gap width made **no difference at all** once clearance was fixed. It only decides
whether clearance can exceed zero. Corridor length is not independent either — a
corridor just 10 long works if the agent sits 5 units in. Depth sweep, 48 runs:

| clearance | outcome |
| --- | --- |
| ≤ 14 | eaten |
| 15 – 45 | trapped: predator stuck at the mouth |
| ≥ 55 | safe, but too deep to be smelled — a **shelter**, not a trap |

Sites with clearance above `SLOT_HOLD_MAX` (45) are reported as `kind="shelter"`
and are **not** returned by default. They still keep an agent alive; they just
don't hold anything. Ask for them with `kinds=("wall", "slot", "shelter")`.

### Not just pairs of rocks

Slots are found with a distance transform over the whole map rather than by
pairing rectangles, so the same code catches notches where rocks overlap,
dead-end corners, and gaps between a rock and the map's boundary wall.

---

## 4. Using it in a controller

```python
from trap_sites import find_trap_sites

# Once per map. Partial knowledge is fine - pass whatever you have mapped,
# and re-run as you discover more.
sites = find_trap_sites(env.obstacles, env.width, env.height, safety="safe")

for site in sites:                    # sorted safest-first, slots before walls
    if site.guaranteed:               # slot: capture is geometrically impossible
        send_agent_to(site.bait)
        break
```

`obstacles` accepts `Obstacle` objects (`env.obstacles`) or plain
`(x, y, width, height)` tuples.

### `TrapSite` fields

| Field | Meaning |
| --- | --- |
| `kind` | `"wall"`, `"slot"` or `"shelter"` |
| `bait` | **where to stand.** `(x, y)` in world coordinates |
| `predator_side` | where the predator ends up |
| `separation` | bait-to-predator distance |
| `clearance` | slot only: distance to the nearest predator-legal point |
| `thickness` | wall: rock thickness. slot: local gap width |
| `margin` | wall: rock extent either side. slot: clearance above the touch threshold |
| `axis` | wall: `"x"` if the rock is thin along x |
| `score` | ranking; higher is safer |
| `worst_case_distance` | closest the predator ever gets |
| `guaranteed` | `True` only when geometry alone prevents capture |

### Presets

| preset | thickness | margin | separation | clearance |
| --- | --- | --- | --- | --- |
| `measured` | ≤ 37.5 | ≥ 32.5 | ≤ 52.5 | ≥ 15 |
| `safe` *(default)* | ≤ 35 | ≥ 40 | ≤ 50 | ≥ 18 |
| `paranoid` | ≤ 33 | ≥ 45 | ≤ 48 | ≥ 22 |

`measured` sits exactly on the empirical knee. It is a cliff edge, not a slope —
sites right at the boundary are one bad bounce from failing. `safe` buys a
buffer on every limit and costs sites, not reliability.

### Cost

About **0.16 s/map** for walls only, **1.4 s/map** with slots (the slot pass runs
a distance transform plus sub-unit verification). That is a once-per-map cost —
cache the result, don't call it per tick.

---

## 5. Command line

```console
python trap_sites.py --seed 3                     # scan a map
python trap_sites.py --seed 3 --real              # that seed's ACTUAL game map
python trap_sites.py --seed 3 --kinds slot        # slots only
python trap_sites.py --seed 3 --safety paranoid
python trap_sites.py --seed 3 --render map.png    # annotated image
python trap_sites.py --seed 3 --json              # machine-readable
```

Without `--real`, obstacles are drawn from the same distribution as the real
generator but are **not** that seed's actual map — a real `Environment` consumes
the RNG for biome generation first. `--real` builds the genuine map (slower).

In the rendered image: **blue dot** = where the agent stands, **red dot** = where
the predator ends up, **green ring** = a slot (capture geometrically impossible),
**amber line** = a wall trap.

---

## 6. What is in this folder

| File | What it is |
| --- | --- |
| `trap_sites.py` | The search. Library + CLI. This is the only file you need at runtime. |
| `test_trap_sites.py` | 25 tests pinning the geometry against the engine's own constants. |
| `_bootstrap.py` | Finds the simulator root so `src.*` imports work from anywhere. |
| `experiments/wall_geometry.py` | Thickness / margin / standoff sweeps for wall traps. |
| `experiments/slot_geometry.py` | Gap / depth / length / stress sweeps for slot traps. |
| `experiments/verify_sites.py` | End-to-end: every site the scanner reports must hold a real predator. |
| `experiments/coverage_scan.py` | How often a generated map contains no usable site. |

Everything under `experiments/` produced the numbers in this document. None of it
is needed at runtime.

---

## 7. Reproducing the measurements

All of these run the unmodified engine — predator AI, energy, rest cycles,
sensing and collision are never touched.

```console
cd survival-simulator/traps

# Wall traps: where the thickness cliff is (750 runs, ~3 min)
python experiments/wall_geometry.py --mode width   --steps 3000 --seeds 3

# Wall traps: how much rock is needed either side (1080 runs, ~5 min)
python experiments/wall_geometry.py --mode margin2 --steps 3000 --seeds 3

# Wall traps: bait standoff vs thickness (600 runs)
python experiments/wall_geometry.py --mode gap     --steps 3000 --seeds 3

# Slot traps: the gap window (60 runs, ~1 min)
python experiments/slot_geometry.py --mode gap     --steps 2000 --seeds 3

# Slot traps: trap vs shelter (48 runs)
python experiments/slot_geometry.py --mode depth   --steps 2000 --seeds 3

# Slot traps: the 648-run stress sweep, randomised approaches
python experiments/slot_geometry.py --mode stress  --steps 2000 --seeds 3

# End-to-end on real generated maps
python experiments/verify_sites.py --maps 50 --steps 3000 --safety measured
python experiments/verify_sites.py --maps 30 --steps 30000 --safety safe   # full game

# How often a map has no site at all (slow: minutes to an hour)
python experiments/coverage_scan.py --maps 2000 --workers 5
```

Add `--workers N` to any of them. The default of 5 is deliberate: each worker
loads numpy, scipy and pygame, and 20 of them exhausted the page file on a
20-core machine.

---

## 8. Results

Every site the scanner reports was replayed in the real engine with a real
predator:

| Check | Horizon | Result |
| --- | --- | --- |
| Wall sites, `measured` | 3 000 ticks | **199/199 held** |
| Wall sites, `safe` | 3 000 ticks | **77/77 held** |
| Wall sites, `safe` | 30 000 ticks (full game) | **47/47 held** |
| Slot sites, `measured` | 3 000 ticks | **202/202 held** |
| Walls + slots, `measured` | 3 000 ticks | **327/327 held** (198 slot, 129 wall) |

Roughly 4 300 full simulation runs in total across the sweeps and the
verification passes. Not one site the scanner reported ever lost its agent.

### Is there always somewhere to stand?

Site availability on generated 1600×1200 maps (80 rocks). Walls-only measured
over 20 000 maps, walls+slots over 1 500:

| preset | walls only | | walls + slots | | |
| --- | --- | --- | --- | --- | --- |
| | sites/map | **maps with none** | sites/map | of which slots | **maps with none** |
| `measured` | 4.17 | 1.44% (1 in 69) | **8.23** | 4.09 | **0 of 1 500** |
| `safe` | 1.42 | 23.9% (1 in 4) | **5.17** | 3.79 | **0.07%** (1 in 1 500) |
| `paranoid` | 0.39 | 67.9% | **3.75** | 3.36 | **0.40%** (1 in 250) |

Slots are what make this practical. Wall traps alone leave nearly a quarter of
maps with nothing at the default `safe` preset; adding slots drops that to one
map in 1 500.

**But there is no guarantee, and controllers must not assume one.** Obstacle
generation is 80 independent uniform draws, so nothing forces a rock thinner than
37.5 or two rocks landing 10–18 apart. At `safe` we have a concrete counterexample
(1 map in 1 500 had nothing); at `measured` none appeared in 1 500 maps, which
bounds the rate below roughly 0.2% without proving it is zero.

`find_trap_sites` returns `[]` in that case, so guard it:

```python
sites = find_trap_sites(env.obstacles, env.width, env.height)
if not sites:
    ...          # fall back to ordinary evasion
```

A site existing is also not the same as a site being worth using — it may sit far
from any fruit, or be awkward to reach. That call belongs to your controller.

---

## 9. Limitations — read before relying on this

**The bait has to survive.** Everything here is about *predators*. A stationary
agent still burns roughly 1 energy per simulated second plus `0.01 * age` per
tick once past its hidden 60–120 s ageing threshold. With 500 max energy that is
a few hundred seconds before it starves. The reference demo in
`predator_trap.py` sidesteps this with `bait.max_age = inf` and a per-tick energy
refill — **that is scaffolding, not something you get in a real game.** A real
trap needs a fruit supply near the bait, or bait rotation before old age.

**Getting there is your problem.** This finds coordinates. It does not path an
agent to them. Slots in particular are 10–18 units wide, so walking in needs
reasonable steering.

**Slot capacity.** The geometric guarantee is about one agent's standing point.
Several agents crowded into one slot have not been tested.

**A wall trap is only as good as the predator AI.** If the AI ever pathfinds,
every wall site stops working at once. Slot sites do not care.

**Coverage numbers are for the default map.** 1600×1200 with `env_width // 20`
= 80 obstacles. A different map size changes the obstacle count and the rates.

**The scan under-reports.** Wall detection only accepts walls with an unchanging
cross-section along the whole stretch, and slot verification inflates rocks by
slightly less than the predator radius. Both bias towards rejecting usable sites
rather than accepting unusable ones, so the real site count is somewhat higher
than reported.

---

## 10. One pitfall worth knowing about

The first slot implementation computed clearance with a distance transform on a
1-unit integer grid. That is **wrong in the dangerous direction**.

Where a gap is close to the predator's 20-unit diameter, the set of legal
predator positions is a sliver that can fall entirely between grid points. The
grid then reports "nothing legal anywhere near" — clearance 28 — for a spot whose
true clearance is **0.05**, which a predator walks straight into. Six sites
marked safe got the agent eaten.

The fix is in `_legal_distance`: sample at a quarter unit, and inflate rocks by
slightly *less* than the predator radius. Both changes push the error the safe
way, so reported clearance is now consistently a slight under-estimate of the
truth (17.32 reported vs 17.65 actual). Checked against brute-force exact
geometry at 0.05 resolution: **0 unsafe out of 25**.

The general lesson: when a grid approximation decides whether something is safe,
make sure the rounding direction is the conservative one, and verify against
continuous geometry rather than trusting the raster.
