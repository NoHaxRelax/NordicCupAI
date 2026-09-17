# Survival simulator mechanics hunt

17 September 2026. Thirteen parallel local investigations against the unmodified vendored simulator at commit `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`: ten `gpt-5.6-sol` high-reasoning mechanics tracks and three `gpt-6-astra` medium-reasoning audits. No competition endpoint, evaluator, score submission, publication, remote target, renderer work, or vendor edit was used. This is broad adversarial coverage, not literal exhaustive proof.

## Executive result

The hunt found no credible contract-compliant instant-win exploit. It did find a useful set of controller corrections and several secondary tactics:

1. **Fix tick-order accounting first.** Movement and turning costs precede birth; passive death precedes fruit collection; old-age drain follows that death gate. Reserve action cost, the 100-energy birth cost, and the next upkeep before acting. Fruit eaten this tick cannot fund a birth until the next tick.
2. **Treat responses as mixed-time snapshots.** Predator positions are normally one move stale; consumed fruit and later-killed agents can remain in cached observations; a list-removal skip can make one agent's observations two predator steps stale. Deduplicate repeated edge records and suppress expected fruit ghosts.
3. **Exploit ordinary state that is actually exposed.** Own age, energy, traits, and biome are public. Old-age drain onset was classified without errors across 12,856 ordinary generated-map transitions. The private `max_age` threshold is hidden, but its onset is detectable.
4. **Use terrain-aware walking.** Slow terrain charges movement energy before applying its speed multiplier. Walking and dry detours are often much cheaper than sprinting through swamp or river. A move samples only its starting pixel, so narrow slow-terrain strips can occasionally be skipped at full speed.
5. **Improve orchard handling.** Dense early orchards replenish quickly but individual trees turn over around the late-50-second range. Timestamp fruit only when continuous coverage makes birth time credible, choose cluster endpoints, and ignore same-response post-contact fruit ghosts.
6. **Test scan-turn thrift more broadly.** Turning only every tenth stationary tick rescued seed 42 from 93.6 to 389.5 seconds in a three-seed 600-second screen, while slightly lowering scores on two runs already alive at the horizon. It is promising but not robustly established.

The best surprising mechanics beyond controller hardening are slight off-center gaze steering of a far predator, exact peer-to-peer coordinate transforms for sensory relays, rare founder-speed obstacle-corner cuts, and last-lineage emergency reproduction. Each is situational and needs policy-level validation before adoption.

## Ranked candidate list

| Rank | Candidate | Evidence and practical magnitude | Status |
| ---: | --- | --- | --- |
| 1 | Tick-order and observation-hardening patch bundle | Exact fixtures plus real-map occurrence. Prevents failed births, deaths before food, chasing consumed fruit, duplicated edge geometry, stale-agent membership, and underestimating predator approach by 15 or occasionally 30 units. | Proven, ordinary-observation, one action per agent. Implement as baseline correctness. |
| 2 | Terrain-aware routing and walking | A walk-10 costs 0.5 movement energy but realizes 10/8/5/3 units in fast/desert/swamp/river terrain. A 90-unit river sprint spent 57.6 energy versus 18.0 walking. Real seeds contained legal thin-strip skips, though privileged scans found them. | Proven physics; global route acquisition remains untested. |
| 3 | Orchard timing, cluster endpoints, and ghost suppression | Generated no-consumer maps produced 867–1,214 fruit births in 300 seconds and natural clusters of 3–4 fruits within 15 units. Known-fresh maturation adds exactly 40 energy, but blind waiting can lose ripe fruit or kill a low-energy agent. | Proven mechanics; observation-only policy benchmark still needed. |
| 4 | Stationary scan-turn thrift | In seeds 1/7/42 at 600 seconds, scores were 582.82/596.49/392.83 versus baseline 591.78/605.87/94.55. The large seed-42 rescue coexists with small losses on the other seeds. | Promising, ordinary-observation, exploratory three-seed result. |
| 5 | Peer sensory relay | Agent-to-agent `distance`, `angle`, and `rel_dir` define an exact frame transform. Fixtures relayed fruit through a wall to 110 units and over a three-agent chain to 260 units, beyond the root's 200 vision range. A turn-only relay pilot won 7/10 explored seeds and raised mean 600-second score by 12.83, but had two large regressions. | High-information mechanic and promising pilot, but tail risk remains. |
| 6 | Slight gaze-offset retreat | In an arranged walking fixture, exact facing died at 3.8 seconds while a 0.02-radian offset survived the 9-second horizon. Across nine comparable 60-second generated runs, alternating offset lost 8 agents versus 9 for exact gaze. | Mechanic proven; general score advantage not established. |
| 7 | Founder-speed obstacle-corner cuts | Among 108,772 sampled clear-endpoint sprint-20 moves on seeds 1/7/42, 13 crossed physical obstacle corners. Twelve examples replayed through the original move method. | Proven, compliant, rare and alignment-sensitive secondary routing optimization. |
| 8 | Emergency last-lineage reproduction | At energy just above 100, birth can happen before passive death or predation and sometimes leave a child without the parent's predation penalty. In the main 400-seed arranged fixture, a 100.05-energy birth preserved a child 62.5%; ordinary retreat won 100% in the open control. | Narrow fallback only; false positives are costly. |

No high-value lead in this ranked list remains completely untested. Items 4–8 have bounded evidence and explicit next tests, rather than being presented as settled improvements.

## Core controller rules

### Energy and birth

For a young agent at the normal 0.1-second tick:

```text
birth succeeds iff energy - movement_cost - turn_cost > 100
parent survives iff energy - movement_cost - turn_cost - 100 - passive_drain > 0
food is collected only after those checks
```

At energy 100.40, an idle birth succeeded, while walk-10 or a pi turn cancelled it. At 100.05, birth occurred but passive drain killed the parent, leaving only the child. A mover reaching fruit must finish the action with strictly more than 0.1 energy to survive until collection. Exactly 0.1 dies and leaves the fruit.

An older agent can pass the passive death gate, become negative from old-age drain, and then be rescued by already-overlapping fruit. This is an ordering quirk, not a safe operating margin. Estimate observed old-age drain and keep reserve.

### Observation hygiene

- Deduplicate exact `Edge.coords` pairs. On the first responses of seeds 1/42/2026, 151 edge entries represented only 32 per-response unique segments, so 78.8% were ray-hit duplicates.
- Treat predator distance as a pre-move value and reserve at least one 15-unit predator step. If the agent's age did not advance, its cache can be another full tick older, producing a measured 30-unit gap.
- Suppress fruit expected to have been contacted in the previous action. A default seed-1 agent returned a fruit at distance 7.0014 after its energy already showed that it had eaten it.
- Cross-check observed agent IDs against the top-level living status list. Later predation can leave a dead peer in another agent's cached observations.
- Interpret returned geometry in the post-turn egocentric frame. Movement used the pre-turn facing, then the turn was applied, then observations were computed.
- Use hysteresis at hearing, cone, and vision-range boundaries. The five-ray visibility polygon has small blind scallops, up to 1.711 units with founder vision and 7.686 at trait caps.

### Aging and reproduction selection

An idle young agent loses 0.1 energy per tick. After its private age threshold, the extra loss is `0.01 * age` per tick. Reviewer 13 classified 12,856 surviving-agent transitions across seeds 13/29/47 with zero errors, including 1,097 old updates and 14 onset detections. Eight list-removal skips were safely rejected through unchanged age.

The six heritable traits are returned immediately for a newborn. `max_age` is not heritable and is freshly drawn from 60–120 seconds. Optimistic mutation economics strongly favor one bottleneck at a time: the median search for improved effective movement was 14 births, while simultaneous movement and max-energy improvement was 276 births, before ecological costs.

## Interaction combinations

### Orchard navigation

Combine continuous-coverage fruit timestamps, cluster endpoints, terrain-aware walking, and one-response ghost suppression. Wait for maturity only when the fruit was newly observed under continuous coverage, reserve is safe, and predator risk is low. First sighting alone does not reveal ripeness because fresh and ripe fruit have identical payloads.

### Predator escape

Combine backward or strafing movement, a 15-unit stale-position margin, and a tiny gaze offset only when the predator is far enough to enter its pivot branch. Do not count down hidden predator energy: an active predator that depletes below zero still completes movement and contact before sleeping. Sleeping contact is harmless only until the wake tick.

### Colony information sharing

Peer observations support exact coordinate transforms without absolute positions. Deduplicate edges first, reject relayed fruit too near its source because it may already be consumed, and use relays initially for orientation or exploration rather than wholesale target override. The aggressive relay policy died earlier on all three screen seeds. The turn-only pilot won 7/10 explored seeds and raised mean 600-second score from 487.52 to 500.35, but lost 163.99 and 234.20 points on seeds 3 and 8. Freeze it before held-out testing rather than tuning further on those seeds.

### Elder replacement

Age-drain detection can trigger selective replacement, but the tested age-over-80 renewal rule was worse than baseline on seeds 1 and 7 and unchanged on 42. Do not add a population slot merely because an agent is old. Renewal should be gated by observed drain, colony food budget, safe spawn space, and an actual replacement need.

### Emergency birth

Use only when a last lineage is predicted to die and ordinary escape is predicted to fail. The child can be eaten in the same predator phase, spawn on its parent or near collision geometry, and begins below the founder sprint threshold. This is not a general evasion move.

## Proven but low-value or fixture-only effects

- A full-face sprint-40 tunnel crosses an exact width-30 wall but fails at width 30.001. All 240 internal obstacles scanned on seeds 1/7/42 had minimum dimension above 30, so generated walls could not be crossed face-to-face at the trait cap.
- The outer boundary can be entered only by the exact full-speed 40-unit setup followed by clamping. Food access, safe exit, and sustainable survival were not shown.
- Exact obstacle tangency is accepted; fruit and predator contact also use strict inequalities. Float-level alignment is too fragile for a core tactic.
- A blocked endpoint is redirected in deterministic 10-degree increments, up to 80 degrees in a fixture, while charging full requested energy. This is mainly a dead-reckoning hazard.
- Resting predator overlap is harmless for 34 ticks from zero energy and fatal on wake tick 35. Resting state and energy are hidden, so a single stationary observation cannot make this safe.
- One predator contact pass can kill every agent in an overlapping pile. Agents and trees are non-solid, but clustering is not protection.
- Negative-energy predation can add about 0.006 score, three orders of magnitude below one second of survival and not worth arranging.
- Fruit/list removal can skip elements for a few ticks, but natural timing did not make this a valuable tactic.

## Rule-breaking or malformed inputs

These are local validation gaps, not recommended strategies and not included in the ordinary-policy ranking:

- Repeated actions for one ID execute before one world update. One hundred walk-10 requests moved 1,000 units in one tick for 50.1 energy. Ten duplicates produced roughly 8.73–10 times normal displacement on seeds 1 and 42.
- A predicted newborn ID can act later in the same request list. This violates the observed-agent contract and is limited by the child's unseen random heading, movement-before-turn order, and 75-energy sprint gate.
- Local direct DTO construction accepts NaN, which corrupts state. Strict JSON may reject it, and no useful survival or score effect was established.

## Coverage and rejected hypotheses

| Track | Covered | Important rejection or downgrade |
| --- | --- | --- |
| 1. Actions/newborns | Duplicate scale, unknown IDs, newborn order, birth and food timing, finite-value gaps | Oversized single moves do not bypass the cap; omissions do not freeze upkeep. |
| 2. Collision | Endpoint geometry, tangency, redirection, contact, boundary clamp, real-map incidence | Generated obstacles cannot be crossed face-to-face at sprint cap. |
| 3. Observations | Cache phases, edge duplication, stale entities, hearing, vision scallops, coordinate semantics | Returned response is not one atomic post-step snapshot. |
| 4. Predators | Rest/wake, final lunge, target switching, gaze branch, multi-kill, spawn failure | Stationary gaze offsets do not save an agent; generated benefit was marginal. |
| 5. Fruit/trees | Maturity, rot, clustering, score, tree hazard, generated turnover | Blind 20-second waiting and permanent-orchard assumptions fail. |
| 6. Reproduction | Mutation distribution, caps, placement, trait visibility, lineage costs | Longevity breeding and multi-trait perfect-child search are poor investments. |
| 7. Death/aging | Passive gate, old-age order, natural list skips, death score | Intentional skip and negative-energy death tactics have negligible value. |
| 8. Terrain | Start-pixel sampling, energy per distance, sliver crossing, current, low-energy cap | River current and biome-specific passive drain are absent. |
| 9. Coordination | Fruit priority, shared RNG order, frame transforms, relays, horizon boundary | Request order does not reassign contested fruit; naive relay targeting underperformed. |
| 10. Adversarial | Cross-area emergency birth, stale responses, collision combinations, malformed input | Reproduction is not generally better than escape; duplicate short moves do not compose into tunneling. |
| Astra 11. Evidence audit | Independent reproduction of high-risk claims and controls | Exact wall tunneling was downgraded from general lead to knife-edge fixture. |
| Astra 12. Combinations | Aging, birth, fruit, predator, scan thrift and 12 full-map screens | Renewal alone was worse; combined gains were seed-sensitive. |
| Astra 13. Observability | Identical-payload controls and public-state inference | Fruit ripeness, predator sleep, and future age threshold are not inferable from one frame. |

## Reproduction and evidence

Run from the repository root with `survival/.venv/bin/python`. Each script writes or refreshes its paired JSON under `survival/results/mechanics_hunt/`:

```sh
survival/.venv/bin/python survival/research/mechanics_hunt/01_actions.py
survival/.venv/bin/python survival/research/mechanics_hunt/02_collision.py
survival/.venv/bin/python survival/research/mechanics_hunt/03_observation.py
survival/.venv/bin/python survival/research/mechanics_hunt/04_predator.py
survival/.venv/bin/python survival/research/mechanics_hunt/05_fruit_tree.py
survival/.venv/bin/python survival/research/mechanics_hunt/06_reproduction.py
survival/.venv/bin/python survival/research/mechanics_hunt/07_death_aging.py
survival/.venv/bin/python survival/research/mechanics_hunt/08_terrain.py
survival/.venv/bin/python survival/research/mechanics_hunt/09_coordination.py
survival/.venv/bin/python survival/research/mechanics_hunt/10_adversarial.py
survival/.venv/bin/python survival/research/mechanics_hunt/11_astra_evidence.py
survival/.venv/bin/python survival/research/mechanics_hunt/12_astra_combinations.py
survival/.venv/bin/python survival/research/mechanics_hunt/13_astra_observability.py
```

Detailed reports: [01 actions](../survival/research/mechanics_hunt/01_actions.md), [02 collision](../survival/research/mechanics_hunt/02_collision.md), [03 observations](../survival/research/mechanics_hunt/03_observation.md), [04 predators](../survival/research/mechanics_hunt/04_predator.md), [05 fruit and trees](../survival/research/mechanics_hunt/05_fruit_tree.md), [06 reproduction](../survival/research/mechanics_hunt/06_reproduction.md), [07 death and aging](../survival/research/mechanics_hunt/07_death_aging.md), [08 terrain](../survival/research/mechanics_hunt/08_terrain.md), [09 coordination](../survival/research/mechanics_hunt/09_coordination.md), [10 adversarial](../survival/research/mechanics_hunt/10_adversarial.md), [11 evidence audit](../survival/research/mechanics_hunt/11_astra_evidence.md), [12 combinations](../survival/research/mechanics_hunt/12_astra_combinations.md), and [13 observability](../survival/research/mechanics_hunt/13_astra_observability.md).

## Recommended next implementation test

Apply only the rank-1 correctness bundle to the existing nursery policy, then add candidates one at a time in this order: terrain-aware walking, orchard history/cluster endpoints, scan-turn thrift, gaze offset, and relay orientation. Run the existing ten seeds at the full 3,000-second horizon and report paired per-seed results, extinction count, population/energy traces, and runtime. Keep emergency birth behind a last-lineage plus failed-escape gate. Do not include duplicate actions, predicted newborn actions, NaN, full-face wall tunneling, or hidden-state sleep logic.
