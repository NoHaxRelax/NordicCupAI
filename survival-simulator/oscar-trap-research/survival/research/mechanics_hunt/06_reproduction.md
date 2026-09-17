# Investigator 6: reproduction, inheritance, energy, and trait caps

Tested 17 September 2026 against the unmodified vendored simulator at commit
`acfc31a4003a5f91bf11032a02cd98c178ddbd7e`. The executable evidence is
[`06_reproduction.py`](06_reproduction.py), and the complete measurements are in
[`06_reproduction.json`](../../results/mechanics_hunt/06_reproduction.json).

Run from the repository root:

```sh
survival/.venv/bin/python survival/research/mechanics_hunt/06_reproduction.py
```

The default run uses 12,000 exact-engine inheritance samples, 10,000 optimistic
lineage trials, and default 1600 × 1200 generated maps for seeds 1–9 and 42. It
makes no network calls and does not alter the vendor.

## Result in brief

The best contract-compliant takeaway is simple: **breed only after reserving the
100 energy plus all movement, turning, and passive costs, then inspect the
newborn's six reported traits immediately and select one current bottleneck at a
time.** The engine returns the child and its exact traits in the same response.
Trying to obtain several beneficial mutations in one child is much too
food-expensive for a baseline policy.

No new high-value reproduction exploit survived practical checking. The unusual
placement and energy states below are real engine effects, but they are hazards or
fixture-only curiosities rather than proven scoring tactics.

## Ranked findings

### 1. Birth is evaluated after movement and turning costs

This extends the already known strict `energy > 100` gate and 100-energy charge.
The action order in
[`environment.py`](../../vendor/survival-simulator/src/elements/environment.py)
is movement, turning, then reproduction.

- At energy 102, an idle birth succeeds and leaves the parent at 1.9 after the
  passive tick.
- At the same energy, requesting a 20-unit sprint first costs 5.5, so birth does
  not happen and the parent ends at 96.4.
- A pi-radian turn costs 0.5, so the same 102-energy parent still gives birth.
- A parent at 100.0001 gives birth but dies to the following 0.1 passive drain.
  Population remains one, so this is replacement rather than growth.

Magnitude: a controller should require at least `100 + movement cost + turn cost
+ next passive/old-age drain + safety margin`, not merely observed energy above
100. This is ordinary-observation, one-action-per-agent behavior.

### 2. Exact newborn traits are immediately observable; longevity is not heritable

Speed, sprint speed, max energy, hearing radius, vision radius, and cone angle
each independently attempt a 10% multiplicative mutation by a uniform factor in
`[0.5, 1.5]`. In 12,000 direct upstream births, 45.72% had at least one apparent
trait change, near the theoretical 46.86%.

The child's old-age threshold behaves differently: every child draws a fresh
`Uniform(60,120)` `max_age`. It is neither inherited nor included in the
observation response. The sample mean was 89.87 seconds with a range spanning
60.00–119.99. **A longevity-breeding policy cannot select a heritable longevity
trait in this engine.** This rejects a potentially expensive strategy direction.

The six heritable traits are all returned in the same response that first returns
the child. Parentage is not returned, so a policy that wants pedigrees must record
which parent requested the birth and associate the newly appearing ID itself.

### 3. Multi-trait selection is far more expensive than single-bottleneck selection

An optimistic Monte Carlo granted an immortal founder, unlimited food, no
predators, and independent children from the default founder. It used the exact
source mutation distribution.

| Target from a default founder | Median births | 90th percentile | Birth energy at median | Ripe-fruit equivalent at median |
| --- | ---: | ---: | ---: | ---: |
| Any visible improvement with no visible trait loss | 3 | 10 | 300 | 5.0 |
| Higher `min(speed, sprint_speed)` | 14 | 47 | 1,400 | 23.3 |
| Higher effective movement **and** max energy in the same child | 276 | 905 | 27,600 | 460.0 |

These costs exclude passive drain, overflow, travel, predation, and the energy
needed to keep parents alive, so they are lower bounds. This extends the prior
speed-search work with a multi-objective comparison. Prefer sequential retention
of useful descendants rather than waiting for a single perfect child.

### 4. Spawn placement is not reliably a 10–30-unit safe offset

In open central terrain, 6,000 births matched the source's radius-uniform 10–30
distribution, with mean radius 19.90. Near the arranged `(35,35)` boundary,
65.97% fell back exactly onto the parent and mean actual separation fell to 6.68.
The fallback happens when the proposed 20 × 20 clearance box intersects a wall.

This was not merely synthetic. In ten default generated maps, seed 8 produced a
child exactly on its parent; the other nine used nonzero offsets. All ten births
succeeded, all children appeared in the same response, and all parents/children
had the expected 49.9/74.9 post-tick energy.

Practical rule: breed in open space and away from predators. Do not depend on the
nominal 10-unit minimum separation.

### 5. Spawn clearance and collision geometry disagree at one side of obstacles

The birth clearance check treats `(x,y)` as the corner of a 20 × 20 box extending
right and down. Movement collision treats `(x,y)` as the center of a radius-5
creature. Seed 7 in the isolated fixture produced a child at
`(144.176, 111.641)` next to an obstacle ending at x=140: birth accepted the
location, while the centered collision check reported overlap.

Spawning also ignores agents and predators. An adversarial fixture placed an
active predator at the deterministic proposed child location; the newborn died
in that same world tick. This proves a hazard, not a normal acquisition rate, and
uses privileged geometry. There is no demonstrated refuge or scoring advantage.

### 6. Trait caps are upper-only and change visible mutation odds

The caps are speed 20, sprint 40, max energy 1000, hearing `chunk_size/4`, vision
`chunk_size`, and cone angle pi/2. At a cap, an attempted upward mutation is
clamped back to the parent value. Across 6,000 births per capped trait, about 95%
were visibly unchanged and 5% decreased; none increased. There are no explicit
lower floors, and positive traits can shrink toward zero across generations.

Consequences:

- A capped trait has no remaining upside but still has a 5% per-birth downside.
- A lineage with `max_energy <= 100` is reproductively sterile: fruit collection
  clamps energy to max energy, so it can never satisfy the strict `>100` gate.
- Exact observation fields, rather than renderer color, should drive selection.

### 7. A rare newborn can start above its own max energy

Newborn energy is fixed at 75 rather than clamped to inherited max energy. A
120-max-energy parent can produce a downward max-energy mutation below 75. The
seeded search found one at birth 101 with max energy 62.067. Artificially placing
a fruit on it caused “eating” to clamp energy from 75 down to 62.067.

This is real engine behavior but a fixture-only effect here. Reaching such a
low-max-energy parent from default founders requires prior adverse lineage
changes, and the affected child is close to or below the reproductive dead end.
It is a reason to cull, not a useful tactic.

### 8. Marginal-energy parent death skips the newborn's first update

When the 100.0001-energy parent gave birth and then died, removal from the agent
list skipped the appended child during that tick. The child remained age 0,
energy 75, and had no cached observations, instead of age 0.1 and energy 74.9.
This is the already known list-iteration family of bugs applied to birth timing.
It respects one action per agent, but saves only 0.1 energy and does not increase
population. It is strategically negligible.

## Evidence boundaries

| Evidence | Ordinary observation | One-action contract | Interpretation |
| --- | --- | --- | --- |
| Energy gate/order probes | Yes | Yes | Proven policy rule |
| Ten default generated-map births | Yes for decisions and returned traits; engine position used only for diagnostics | Yes | Small real-map validation |
| 12,000-birth inheritance and 6,000-birth cap samples | Traits would be observable, but births and food are free in the fixture | Direct engine fixture | Distribution evidence, not a viable policy |
| Boundary and obstacle placement probes | Exact position/obstacle state is privileged | Birth call is valid; geometry arranged | Hazard only |
| Predator-on-spawn probe | Predator location and RNG are privileged | One parent action | Adversarial possibility, no incidence estimate |
| 10,000 lineage trials | Traits are ordinarily visible | Abstract exact-distribution model | Optimistic lower-bound economics |
| Over-cap newborn fruit probe | Hidden state and artificial fruit placement | Not a gameplay run | Fixture-only curiosity |

No probe uses duplicate IDs, predicted newborn actions, malformed requests, or
another violation of the documented one-action-per-observed-agent rule.

## Rejected or bounded hypotheses

- **Breed for longer life:** rejected. `max_age` is independently redrawn and
  hidden, not inherited.
- **Every child starts 10–30 units from its parent:** rejected. Clearance failure
  falls back exactly onto the parent, including 1/10 small real-map checks.
- **Accepted spawn means collision-free circle:** rejected by the one-sided box
  mismatch, but no useful exploitation was found.
- **Caps make a lineage mutation-proof:** rejected. Upside disappears, while
  roughly 5% of births still reduce each capped trait.
- **All newborns start at or below their own energy capacity:** rejected in a
  low-max-energy fixture; strategically adverse.
- **A barely eligible parent can cheaply expand population:** rejected. It dies
  after paying the birth cost, leaving only a replacement child.
- **Wait for a child improving several major traits at once:** rejected as a
  practical baseline. The optimistic median joint speed-and-energy improvement
  cost was 27,600 birth energy before ecology.

## Recommendation to the coordinator

Treat reproduction as an energy-budgeted, observable selection process rather
than an exploit surface. Preserve viable parents, breed in open predator-free
space, reserve action costs before testing the gate, and retain children that
improve the current bottleneck without damaging essential traits. Do not invest
in longevity breeding, boundary clumping, low-max-energy states, or same-tick
parent replacement.
