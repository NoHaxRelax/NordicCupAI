# Mechanics hunt 09: population coordination, action order, and score horizon

Tested 17 September 2026 against the unmodified vendored simulator at commit
`acfc31a4003a5f91bf11032a02cd98c178ddbd7e`.

## Result

The strongest compliant finding is a coordinate-sharing primitive. When agent A
observes agent B, the reported `distance`, `angle`, and `rel_dir` define the exact
rigid transform from B's local coordinate frame into A's frame. The controller can
therefore relay B's fruit sighting to A without knowing either absolute position or
heading. The reconstruction error was `4.55e-14` in a nontrivial fixture. It also
recovered a fruit 110 units away through a wall and a fruit 260 units away through a
three-agent chain, beyond the receiver's 200-unit vision range.

This is a proven information advantage, not yet a robust policy advantage. Blindly
walking toward relayed fruit reduced mean score from 430.73 to 333.13 in a
three-seed, 600-second pilot. Using relay data only to turn and acquire direct vision
was better but volatile: across ten generated-map seeds it raised mean 600-second
score from 487.52 to 500.35 and mean survival from 494.72 to 501.65 seconds, winning
7/10 seeds. It also had large losses on seeds 3 and 8. Do not replace direct foraging
with it yet. Use the frame transform as shared-map infrastructure, with freshness,
occlusion, and path-safety checks.

All strategy runs issued exactly one action for every living observed agent and no
duplicate IDs. Diagnostics read engine state only to measure accuracy and outcomes.

## Ranked findings

### 1. Exact local-frame alignment enables collective sensing

For an observation by A of B:

```text
translation A<-B = distance * (cos(angle), sin(angle))
rotation    A<-B = angle + pi - rel_dir
```

Transform B's local object vector by that rotation, then add the translation. These
transforms compose along an observation graph, so a chain of nearby agents can relay
sightings farther than one agent's sensor range.

Evidence:

- Direct reconstruction error: `4.5496934130773415e-14`.
- Wall fixture: A had zero direct fruit observations, heard B through the obstacle,
  and reconstructed the fruit at exactly 110 units.
- Three-agent fixture: A had zero direct fruit observations and reconstructed a
  fruit at exactly 260 units through A-B-C.
- Generated maps contained many real opportunities. The nursery control had 829
  agent-ticks with no direct fruit but an observable connected peer with fruit across
  seeds 1, 7, and 42.

Ordinary versus privileged state: the transform uses only response DTO fields. True
positions were used solely to calculate diagnostic error.

Practical use: share sightings, align local maps, assign distinct targets, and let an
agent turn to acquire direct vision before committing to travel. Fruit observations
have no IDs, so cross-agent deduplication requires spatial clustering in a shared
frame.

### 2. Turning toward peer sightings is promising but has unacceptable tail risk

Generated-map results used default dimensions, obstacles, trees, fruit, predator
spawning, realistic founder energy, and the existing observation-only nursery policy.
Runs stop at death or 600 seconds.

| Seed | Nursery score at 600/death | Turn-only relay | Difference |
| ---: | ---: | ---: | ---: |
| 1 | 591.78 | 596.97 | +5.18 |
| 2 | 506.02 | 611.79 | +105.77 |
| 3 | 605.78 | 441.79 | -163.99 |
| 4 | 397.53 | 514.75 | +117.22 |
| 5 | 487.93 | 608.04 | +120.11 |
| 6 | 443.04 | 578.27 | +135.23 |
| 7 | 605.87 | 611.61 | +5.75 |
| 8 | 578.13 | 343.93 | -234.20 |
| 9 | 564.55 | 609.34 | +44.79 |
| 42 | 94.55 | 87.02 | -7.53 |
| **Mean** | **487.52** | **500.35** | **+12.83** |

The policies diverge and reproduction consumes the engine's shared RNG, so a common
seed does not preserve an identical later stream of births, trees, or predators. The
sample is small, local macOS rather than official Linux, capped at 600 rather than
3000 seconds, and was explored on these seeds. Treat the mean improvement as a lead,
not an unbiased performance estimate.

Why blind relay failed: a peer's cached observation can contain a fruit it consumed
in that same step, a wall can occlude the target, and straight-line movement has no
path planner. The probe excludes peer fruit sightings at distance 12 or less, but
that does not eliminate older stale reports or blocked routes.

### 3. Contested fruit is resolved by creation/list order, not request order

Two equal-energy agents landed equally close to the same 20-energy fruit. Agent 0
received it whether actions were submitted `[0, 1]` or `[1, 0]`; final energies were
118.9 and 98.9 in both cases. Agent-agent collisions do not exist, and all agent
actions finish before any fruit collection.

Practical meaning: reordering the response cannot direct contested food to the needy
agent. Coordinate endpoints so only the intended consumer overlaps the fruit. Older,
earlier-created agents otherwise win exact simultaneous contests.

### 4. Clustering multiplies contact loss in one predator step

Five agents with 100 energy each overlapped one active predator. All five died in the
same tick. Score became `-4.895`: +0.1 survival time minus a cumulative 4.995 eating
penalty. The predator began at 199/200 energy and ended capped at 200, but its energy
cap did not cap the score penalty.

Practical meaning: maintain separation during danger. A sacrificial or trapped agent
does not shield colocated peers, and a single contact event can erase several seconds
of survival score.

### 5. Same-tick ordering creates exact low-energy thresholds

Actions happen first, then passive drain/death, observation/fruit collection,
predators, and finally +0.1 score/time.

- A forest agent with 0.59 energy spent 0.5 to walk 10 units onto fruit, then died on
  the 0.1 passive charge before eating it. At 0.61 it survived and ended with 20.01.
- Fruit cannot fund a birth in the same tick. A parent at 99.9 overlapping fruit did
  not reproduce, then ate. A parent at 100.3 reproduced first and then ate.
- A doomed 0.55-energy agent could spend itself below zero before the predator phase,
  avoiding a 0.0045 score penalty and denying that residual energy. This is valid but
  too small to prioritize except when death is certain.

All of these are contract-compliant and use exposed own energy. Terrain changes the
passive term if a biome ever uses a non-default energy drain multiplier.

### 6. The local server loop awards one step beyond 3000

The official local server marks game over with `env.time <= 3000`. Starting a fixture
at 2999.8 yielded states at 2999.9 (`ok`), 3000.0 (`ok`), and 3000.1 (`game_over`).
An alive species therefore receives the final +0.1 time score and world interactions
through 3000.1 in this source.

This is a local source behavior, not a remotely verified competition guarantee. It is
not actionable before a policy can survive the full horizon.

### 7. Compliant request order matters for simultaneous births, but not ordinary movement

Swapping two one-action reproduction requests assigned child ID 2 and the first RNG
draws to the first requested parent's lineage. Both newborns were appended, aged to
0.1, and drained to 74.9 before the response. Swapping two non-reproduction movement
requests produced byte-for-byte equivalent position and energy snapshots.

The mapping is real but has no demonstrated expected advantage because future RNG is
hidden and draws are exchangeable. The cap-six nursery controller never proposed
multiple births in the same tick on seeds 1, 7, or 42, so a one-birth-per-tick wrapper
was an exact null result on those runs.

## Contract and fixture boundaries

Proven contract-compliant advantages:

- Frame alignment and multi-hop sighting relay from ordinary observations.
- Explicit food-consumer endpoint coordination.
- Spacing agents to avoid cumulative predator-contact losses.
- Planning birth and last-gasp movement against the actual within-tick order.

Promising but not established:

- Turn-only relayed-fruit acquisition. Positive mean on ten explored seeds, but large
  regressions and no 3000-second held-out validation.

Fixture-only or source-boundary effects:

- The late-time 3000.1 check begins from an artificial 2999.8 state.
- True positions are used only to verify transform error.
- The five-agent predator overlap is deliberately arranged to measure worst-case
  magnitude.

Rejected or low-value hypotheses:

- Request list order can choose the winner of a contested fruit: rejected.
- Ordinary non-spawn movement order changes interactions: rejected.
- One-birth-per-tick improves the current cap-six nursery policy: null in three seeds.
- Blind movement toward relayed fruit is beneficial: rejected by 0/3 pilot wins and a
  97.60-point lower mean score.
- Deliberate pre-predator energy burning is a material score tactic: mechanically
  real, but only +0.0045 in the realistic low-energy fixture.

Rule-breaking input quirks such as duplicate actions and actions for unseen newborn
IDs are outside this investigation and were not used by any policy or benchmark.

## Reproduction

From the repository root:

```sh
# Fast deterministic fixture suite
survival/.venv/bin/python survival/research/mechanics_hunt/09_coordination.py \
  --output /tmp/09_coordination-smoke.json

# Three-seed policy pilot recorded in the result JSON
survival/.venv/bin/python survival/research/mechanics_hunt/09_coordination.py \
  --full-map --seeds 1 7 42 --horizon 600 \
  --modes nursery one_birth relay

# Turn-only relay follow-up
survival/.venv/bin/python survival/research/mechanics_hunt/09_coordination.py \
  --full-map --seeds 1 7 42 --horizon 600 --modes relay_orient \
  --append-label relay_orient_pilot
```

Executable probes: [09_coordination.py](09_coordination.py)

Recorded evidence: [09_coordination.json](../../results/mechanics_hunt/09_coordination.json)

Baseline seeds 2–6, 8, and 9 use the already recorded nursery results in
`survival/results/benchmark-nursery-<seed>.json`, truncated to their 600-second sample
or final death. The turn-only held-out outputs themselves are embedded under
`generated_map_followups.relay_orient_heldout` in the result JSON.

## Remaining uncertainty

The next useful test is not a broader blind sweep. It is a frozen relay policy with
explicit report freshness, obstacle/path rejection, and target deduplication, tested
on untouched seeds at the full 3000-second horizon on Linux. The exact frame transform
needs no further mechanics validation; whether it improves final score safely remains
open.
