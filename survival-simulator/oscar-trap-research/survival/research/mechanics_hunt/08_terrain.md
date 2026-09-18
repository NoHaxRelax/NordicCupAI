# Mechanics hunt 08: terrain transitions and energy

Tested 17 September 2026 against the unmodified vendored simulator at commit
`acfc31a4003a5f91bf11032a02cd98c178ddbd7e`. The probes are local and make no
network, evaluation or submission calls.

Run from the repository root:

```sh
survival/.venv/bin/python survival/research/mechanics_hunt/08_terrain.py
```

The assertions write the complete measurements to
`survival/results/mechanics_hunt/08_terrain.json`.

## Ranked findings

### 1. Thin slow-terrain strips can be crossed without slowdown

**Proven mechanic; contract-compliant; medium situational value.** The engine
looks up `move_penalty` at `int(start_x), int(start_y)` once, after charging
movement energy, and multiplies the whole action by that single value. It does
not integrate terrain along the path. A founder at x=99.9 on forest crossed a
river boundary to x=109.9 with a walk-10 request. Starting at x=100.0 on river
moved only 3.0. Both actions cost 0.5 energy.

This creates a real, bounded terrain-tunnelling effect: if a slow strip is less
than the legal action length and both endpoints are on full-speed terrain, the
agent skips the strip at full speed. On default-size generated biome maps:

| Seed | Walk 1–9 | Founder sprint 1–19 | Trait-cap sprint 1–39 | Reproduced sample |
| --- | ---: | ---: | ---: | --- |
| 1 | 46 | 46 | 61 | A 5-pixel river run was crossed with a legal walk-6 at full speed; river-origin control moved 1.8. |
| 7 | 17 | 21 | 30 | A 1-pixel swamp run was crossed with walk-2 at full speed; swamp-origin control moved 1.0. |
| 42 | 0 | 0 | 1 | No founder-length qualifying run found. |

These are horizontal and vertical **scanline runs**, not unique terrain
components; adjacent rows can count the same feature. The maps and terrain are
genuine generated outputs, and the agents use normal founder traits and 150
energy. The experiment injected their starting positions after privileged map
inspection and used an otherwise empty environment with only normal boundary
walls. It therefore proves the physics and natural occurrence, not that a live
policy will locate every sliver.

The ordinary state exposes the agent's current biome, energy and traits, but no
absolute position or map. A controller could reuse a discovered crossing with
action-history dead reckoning and local landmarks, but discovering and reliably
aligning with one is harder than the privileged scan. No hidden predator or
energy state is needed. Every candidate action uses one action per agent.

A synthetic trait-cap control gave an agent the engine's legal inherited maxima
of speed 20 and sprint speed 40. From forest it crossed a 30-pixel river strip
in one 40-unit action for 11 energy; the same request launched inside river moved
only 12. The traits are mechanically reachable but rare, and the arranged strip
and exact starting point are privileged. This is a fixture-only upper bound, not
evidence for investing in sprint breeding or a decoy policy.

### 2. Slow terrain charges for requested distance, not distance achieved

**Proven mechanic; contract-compliant; high route-planning value.** A normal
walk-10 costs 0.5 movement energy everywhere but realizes 10 units in forest or
grassland, 8 in desert, 5 in swamp and 3 in river. The resulting movement-energy
cost per realized unit is 0.05, 0.0625, 0.10 and 0.1667 respectively. A sprint-20
costs 5.5 everywhere and realizes 20, 16, 10 or 6 units, making sprint especially
expensive in water.

The 90-unit, one-action-per-tick trials start with normal founder energy 150 and
include normal passive drain:

| Terrain | Walk time / total energy | Sprint time / total energy |
| --- | ---: | ---: |
| Forest or grassland | 0.9 s / 5.4 | 0.5 s / 28.0 |
| Desert | 1.2 s / 7.2 | 0.6 s / 33.6 |
| Swamp | 1.8 s / 10.8 | 0.9 s / 50.4 |
| River | 3.0 s / 18.0 | 2.1 s / 57.6 |

The river sprint becomes self-defeating: after energy falls below 100, the
engine caps the 20-unit request to walking speed before applying the 0.3 river
modifier. Twelve of its 21 ticks were low-energy-capped. This is not an exploit;
it is a strong reason to walk through slow terrain except under immediate danger.

For a long homogeneous walking segment, one slow-terrain unit consumes the same
ticks and movement energy as `1 / move_penalty` fast units. Ignoring obstacles,
danger, boundary launch and discovery cost, a fast-terrain detour can add up to
25% of the desert segment, 100% of a swamp segment or 233% of a river segment
before tying the straight route. This is an idealized break-even rule, not a
tested global path planner.

Current biome and energy are ordinary observations, so a policy can apply the
energy rule without privileged state. Selecting the globally best detour needs
a learned local map or privileged geometry; the public response alone does not
provide global coordinates.

### 3. Boundary effects are asymmetric for exactly one action

**Proven mechanic; contract-compliant; low-to-medium tactical value.** A move
launched on fast terrain keeps full speed even after entering slow terrain. A
move launched one pixel inside the slow terrain keeps the slow modifier even if
it exits to fast terrain. Ending exactly on the integer boundary immediately
reports the destination biome in the next state.

This can provide a one-action launch into a river or swamp and can clear a thin
feature entirely. It does not provide repeated free travel across a wide river:
the following action starts in slow terrain. Oscillation therefore does not
create net energy or a sustainable speed multiplier.

### 4. Terrain does not change passive drain, and the river has no current

**Proven negatives; contract-compliant.** Every biome inherits
`energy_drain_rate = 1.0`; no subclass overrides it. An idle 0.1-second tick cost
0.1 energy in forest, grassland, desert, swamp and river. River declares
`stream_flow_speed = 5.0`, but the engine never applies it: an idle river agent
remained at exactly the same coordinates for 1.0 second.

This rejects two tempting strategy assumptions. Water increases energy per
distance only by slowing movement while charging the unscaled request; it does
not add passive drain or drift.

### 5. Low energy applies the walking cap before terrain slowdown

**Proven mechanic; contract-compliant.** Below 20% of maximum energy, a sprint-20
request is first capped to walk-10, costs 0.5, and then becomes 10/8/5/3 realized
units depending on terrain. For a founder with max energy 500, this threshold is
100 energy. Increasing max energy through inheritance raises the absolute sprint
cutoff, so a high-capacity descendant can lose sprint access at an energy level
where an ordinary founder could still sprint.

Energy, max energy, speed, sprint speed and current biome are all public state;
no privileged information is required to enforce a safe movement budget.

## Fixture boundaries and controls

- The stripe and uniform-biome cases are explicit diagnostic fixtures. They
  isolate the original action, energy, biome and time-step methods without
  changing vendor code.
- Generated-map seeds 1, 7 and 42 use default dimensions and the original map
  generator. Starting positions were selected with privileged map inspection;
  no claim is made that the policy initially knows those coordinates.
- The 90-unit trials begin with realistic founder energy and traits, use exactly
  one action per tick and leave ordinary non-agent stepping enabled. There were
  no predators; automatically generated trees did not intersect the route.
- Forest and grassland are the displacement controls. Slow-origin actions are
  the transition and sliver negative controls. Idle tests isolate passive drain
  and reject river drift.
- Results concern local upstream physics only. They do not establish hosted
  behavior, competition permission beyond the documented one-action contract,
  or a full-game score improvement.

## Rejected hypotheses

- River `stream_flow_speed` pushes stationary agents.
- Slow terrain discounts movement energy in proportion to realized distance.
- Biomes have different passive energy drains.
- A boundary-crossing action integrates the fractions spent in each biome.
- A sprint request below 20% max energy retains sprint displacement.
- Repeated boundary oscillation supplies free distance or energy across a wide
  slow region.

## Practical recommendation

Use terrain-aware walking and favor surprisingly long dry detours around swamp
or river when danger and food do not dominate. Record encountered biome changes
and local landmarks so a controller can reuse known narrow crossings. Treat
thin-strip skipping as a small opportunistic optimization, not a core survival
plan: its generated-map occurrence varies substantially, and the public state
does not directly reveal map geometry.
