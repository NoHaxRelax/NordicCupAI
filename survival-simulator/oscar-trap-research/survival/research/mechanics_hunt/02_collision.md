# Mechanics hunt 02 – collision geometry and endpoint movement

Tested locally on 17 September 2026 against the unmodified vendored simulator at
commit `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`. No network, hosted evaluator,
submission, renderer, or vendor edit was used. Every executed tick used at most one
action per agent, so every finding here respects the documented action-count
contract.

## Reproduce

```bash
survival/.venv/bin/python survival/research/mechanics_hunt/02_collision.py
```

This regenerates
[`02_collision.json`](../../results/mechanics_hunt/02_collision.json). Assertions
include positive cases and nearby negative controls. The run takes about 35 seconds
on the development machine because it creates three full maps and checks 300,000
sampled moves.

## Ranked findings

### 1. Real obstacle corners can be cut at founder speed – proven, narrow advantage

The collision test examines only the proposed endpoint. A founder at `(195, 209)`
moving `19.799` units at −45° ends at `(209, 195)`. Both endpoints are outside the
agent-radius-expanded box around a synthetic obstacle beginning at `(200, 200)`, but
the segment passes through the physical rectangle. The engine accepts the endpoint.
A 19-unit control ends inside the expanded box and is redirected instead.

This extends the already documented straight wall-tunneling result in an important
way: corner cutting does **not** require an evolved sprint trait or a thin obstacle.
At distance 20, the deepest symmetric path reaches about 2.071 units inside each
physical edge. At the trait cap of 40 it reaches about 9.142 units. It can shave a
small detour around a corner or cross an otherwise separating corner, but it cannot
cross the full width of ordinary generated walls.

Generated-map validation used seeds 1, 7, and 42, with their ordinary 80 internal
obstacles per map. Among clear-endpoint, uniformly sampled moves:

| Requested trait move | Clear moves | Segment crossed expanded collision box | Segment crossed physical rectangle |
| --- | ---: | ---: | ---: |
| founder sprint 20 | 108,772 | 389 (0.358%) | 13 (0.0120%) |
| cap sprint 40 | 102,212 | 1,290 (1.262%) | 336 (0.329%) |

The 40-unit requests were reduced normally by the start biome; the saved examples
include forest and desert cases. Twelve physical crossings were replayed through the
original `update_entity_position` method and all reached the analytically predicted
endpoint. These rates measure geometric opportunity from privileged, uniformly
sampled starts and headings. They are **not** the incidence under a real controller.

Practical status: contract-compliant and potentially observable. Visible obstacle
edges provide relative geometry, while traits are exposed. A controller would still
need position history and a safety margin to line up two clear endpoints without
absolute coordinates. The small founder penetration and precision requirement make
this a secondary routing optimization, not a central survival strategy.

### 2. A blocked endpoint is silently redirected in deterministic 10° steps – proven, useful caution

The engine does not stop or shorten a blocked move. It tries headings in 10°
increments until the *same-distance endpoint* is clear. A 20-unit move from x=176
toward a vertical wall was redirected −20°; from x=190 it was redirected −80°. The
no-obstacle control moved straight. Full requested movement energy was charged in
all cases.

This can serve as crude obstacle following, but it is not a free-energy or extra-
distance exploit. The negative-first search order creates a deterministic side bias,
and a controller that assumes its requested heading was executed can accumulate a
large position error. Use observed edge geometry or observation history after a
blocked-looking move rather than dead reckoning through it.

### 3. Exact tangency is clear because every contact inequality is strict – proven, fragile

For an agent of radius 5, x=195 is clear beside an obstacle starting at x=200, while
x=`195 + 1e-9` is blocked. The fixture moved 20 units along the exact tangent without
redirection. The same strictness appears in interactions:

- A fresh radius-5 fruit crossed mid-move and left exactly 10 units from the final
  agent center was not collected. Ending at 9.999 units collected it.
- An active predator ending exactly 15 units from an agent did not kill it that tick;
  it killed the stationary agent on the next tick.

These are one-tick boundary effects, not stable refuges. Float-level alignment is too
fragile to rank as a primary tactic. For food, aim endpoints clearly inside the
contact radius. For predators, treat equality as no safety margin.

### 4. Collision uses a square-expanded rectangle, not exact circle geometry – proven, mostly a penalty

Obstacles are axis-aligned rectangles expanded by creature radius in x and y. A
radius-5 agent center at `(195.1, 95.1)` is blocked by the corner of an obstacle at
`(200, 100)` even though its Euclidean corner distance is 6.930. The square model
extends up to `5 × (sqrt(2) − 1) = 2.071` farther than exact circle-versus-rectangle
contact at a 45° corner.

This explains some surprising early redirects near corners. It is not a strategy
advantage by itself, though endpoint-only corner cutting can bypass the square and,
for a narrow range, the physical rectangle as described above.

### 5. Trees and other agents are non-solid – proven, ordinary behavior

An agent ended at a tree center, and two agents ended at exactly the same coordinates.
Only rectangular obstacles participate in movement collision. Overlapping agents can
therefore share a food endpoint or narrow route without blocking one another. Trees
should be treated as landmarks and fruit sources, not movement barriers.

### 6. Full wall and boundary tunneling are real but unavailable as a routine generated-map tactic

The known endpoint-only cases reproduce exactly:

- A founder sprint of 20 crosses a synthetic width-10 wall only at the exact
  `width + 2 × radius = 20` threshold. Width 10.001 redirects it.
- A max-trait sprint of 40 crosses a synthetic width-30 wall only at the exact
  threshold. Width 30.001 redirects it.
- From x=35, a max sprint of 40 crosses the 30-wide outer boundary, after which the
  post-collision world clamp puts the agent at x=5, inside the boundary. A 39.999
  control redirects and remains outside.

Across seeds 1, 7, and 42, the smallest generated internal obstacle dimension was
30.026. Because an agent needs obstacle dimension plus its 10-unit diameter, even a
40-unit cap trait cannot cross between opposite faces of any scanned generated
obstacle. Terrain only reduces the effective distance. The outer-boundary case also
requires exactly 40 on full-speed terrain.

Energy and inheritance further reduce practicality. The max-trait wall case succeeded
at energy 101, then fell to 89.9; at energy 99 the normal below-20%-energy rule capped
the move at walking speed 20 and it failed. A newborn with the usual 75 energy cannot
perform it before feeding, and sprint 40 cannot arise from one mutation of the
founder sprint 20. Entering a boundary is also not yet shown to preserve access to
food or provide a safe exit. Keep these as diagnostic engine bugs, not current policy
objectives.

## Source explanation

- `environment.py:496-557` charges movement, applies the start biome multiplier,
  checks only the endpoint, searches alternative headings, then clamps to bounds.
- `environment.py:781-806` expands every obstacle as an axis-aligned box and uses
  strict `<` comparisons.
- `environment.py:662-672` collects fruit only after movement and only for strict
  endpoint overlap.
- `environment.py:674-724` moves a non-resting predator and then applies strict
  endpoint predation contact.
- `biome.py:12-61` gives movement multipliers 1.0, 0.8, 0.5, and 0.3.

## Coverage, rejected hypotheses, and boundaries

- **Rejected:** generated walls can be crossed face-to-face at the current trait cap.
  The generated lower bound is just above the exact theoretical threshold.
- **Rejected:** a slightly sub-threshold wall or boundary move still tunnels. Controls
  at width 10.001/30.001 and distance 39.999 redirected instead.
- **Rejected:** path contact is enough to collect fruit or trigger predation. Both use
  strict endpoint contact.
- **Rejected:** collision is circular around rectangle corners. It is a square AABB
  expansion and therefore over-blocks corner space.
- **Rejected:** blocked movement is simply cancelled. It can rotate by as much as 80°
  in the tested fixture while preserving requested distance and energy cost.
- **Confirmed but low value:** exact fruit/predator tangency produces a one-tick edge
  case; it is too fragile for a core strategy.
- **Not tested as a policy:** observation-only acquisition and score improvement from
  deliberate corner cuts. The generated-map scan uses privileged map coordinates.
- **Not claimed:** literal exhaustiveness, hosted behavior, official permission beyond
  the documented one-action contract, or a sustainable boundary refuge.

All synthetic cases explicitly arrange geometry and traits. The seeded-map scan uses
real generated obstacles and terrain but still injects sampled starts for diagnosis.
No result depends on duplicate actions, hidden predator energy, disabled spawning, or
modified physics.
