# Overnight results (19 Sept 2026), survival simulator, C++ native engine + policy on Runpod

All numbers: nightsim fork (survival/nightsim = native engine + orchard policy in C++, bit-identical to the
Python policy), Linux EPYC pods, 3000 s horizon, paired comparisons on identical seeds. No competition API used.

## Phase 1: no predators (teammates handle predators)

| Measurement | Seeds | Result |
| --- | ---: | --- |
| r21s0c2 vs old best (r3s2c6) | 950 | survival 2391 vs 2353 s (+38 ± 14), score 2553 vs 2492 (+60 ± 14), fruit +22.6 |
| Late-game harvest (r21s0c2) | 450 | 92% of spawned fruit eaten before 1000 s, 62-72% after 2000 s |
| Rotted late fruit by nearest agent (r21s0c2) | 152 | 80-90% had no agent within 200 units during its life; 55% none within 400 |
| Perfect tree knowledge (oracle, diagnostic) | ~500 | +99 ± 30 s: the discovery headroom |
| 32 perturbations, 44 single-parameter probes, 48 late-game schedules | 96-128 each | none clearly better; late-game changes lean negative |
| Best screen winners re-run (L1c18, L1c45, s1a*, q_n0_60, q_sm0, q_brl450, combo) | 512 each | all within +-2 SE of r21s0c2 (winner's curse) |
| Vision selection (fit_vision 1.5/2/3, heir_select), faster sweep | 512 each | within +-12 s |
| Late birth control (1 birth/tick, lower late cap), tree-age inference (cell coverage, fruit) | 256-512 | no gain |
| Oracle split: trees within 120/200/300/400/800 of a member | 256-512 | +48/+17/+85/+122/+129 s; ages only +15; own vision range +75 |
| Protect known live trees from false 'dead' marks (oracle) | 256 | +51 ± 22; practical miss thresholds and wall-map occlusion do not capture it |

**Phase 1 conclusion (01:45):** r21s0c2 stays the best practical policy: 2393 s mean survival, score 2555 over
1465 seeds. The remaining headroom is detection of trees 200-400 units from agents (~+120 s with perfect local
knowledge), which no parameter, selection or bookkeeping change tested tonight captures.

Survival correlates weakly with the map's tree count (+0.1) and more with the colony's own state at 1500 s
(agents alive +0.38, fruit eaten 1500-2000 s +0.45).

## Phase 2 preview (parked until 02:00 per Oscar)

| Measurement | Seeds | Result |
| --- | ---: | --- |
| r21s0c2 with predators, no predator logic | 64 | 846 s, 226 agents eaten per game |
| + evasion (face, back away, sprint < 90) | 64 | 963 s, 116 eaten (score +282 ± 50) |
| Perfect-trap model: every predator trapped T s after spawn | 53 each | T=600: +205 s; 300: +481; 100: +922; 30: +1167 |
| One agent vs one fresh predator, 30 s (walk speed 10/15/20) | 288 per variant | caught ~60%/20%/8% with evasion; 79% without |

## Phase 2 (predators), 02:00-03:15: one problem at a time

| Step | Test | Result |
| --- | --- | --- |
| 1. One predator to the trap | 2337 scenarios per setting (true walls and poses, bait frozen 9 units inside a crevice, guide and predator placed with clear lines) | 78% delivered; 96% when the predator starts behind the guide, 80% beside, 53% between guide and trap. Lever: keep the predator inside its 60-unit hearing (band 45-70); bait never died |
| 2. Several predators | 1, 3, 6, 10 predators already held at the mouth, one more delivered | 94-98% delivered and all still held; capacity is not a constraint |
| Full games with predators (192 seeds) | no predator logic / evasion / evasion + trap machinery | 888 s / 923 s / 700 s: the trap costs ~220 s |
| Why: guide funnel in full games | 2206 guide episodes | 97% establish a chase, 13% reach the lane point, 1% reach the handoff; 87% of guides die en route, 79% of those walk slower than 12 (predator sprints 15) |

A standing bait alone costs ~170 s. Guides in full games fail for reasons absent from the scenario: agents walk ~10,
distances of 300-1000 units, no routing around obstacles, several predators, predators busy with other agents.

## Step 3: not losing agents to predators (03:15-04:10, full games with predators, 192 seeds per row, paired)

| Evasion | Survival | Score | Eaten per game |
| --- | ---: | ---: | ---: |
| none (r21s0c2 as is) | 864-888 | 720-728 | 215-232 |
| v0: flee within 200, shared alarms, sidestep 0.8 rad within 60 | 923-959 | 976-1014 | 59-60 |
| flee within 100 | 1120 | 1165 | 94 |
| flee within 80, own sightings only | 1239 | 1260 | 130 |
| + sprint within 40 | 1262 | 1283 | 129 |
| + sidestep 1.2 rad | 1364 | 1385 | 133 |
| + sidestep radius 80, 1.4 rad | 1444 | 1462 | 139 |
| + flee within 70, face within 80 (`pred_best_0400`; confirmed on fresh seeds 7000-7191) | 1507 | 1520 | 142 |

Lessons: the colony dies of the economy (starvation 290 per game), not of the kills (132); fleeing far, sharing
alarms, and running to a refuge (crevice pass-through: -280 to -545 s) all cost more than they save. The
cheapest local response wins: a short sidestep at 1.4 rad off the predator's heading (its direct chase turns at most
0.3 rad per tick), sprint only inside 40, face it only inside 80. Speed selection and population re-tuning under
predators: no gain. Perfect-trap model still promises +900 s if every predator were trapped within 100 s of its
spawn; the trap pipeline reaches that only in scenarios (step 1), not yet in full games.
