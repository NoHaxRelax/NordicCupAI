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

## After 04:30: decoys, keeper baits, and the pose-drift finding

| Test (192 games with predators, paired vs pred_best_0400) | Result |
| --- | --- |
| old agents walk toward nearby predators (decoy) | -6 s, -19 score; low-energy decoys -34 to -120 |
| flee only when I am the predator's closest visible agent; no births with a predator within 100-400 | all within noise |
| detection: sweep rate, hearing/vision selection, watch patience | noise (vision +33 ± 27) |
| keeper spawns bait children near the crevice rear (rear entry, obstacle-avoiding approach) | -414 to -506 s |

Kill profile (128 games): 140 kills per game, 34% of victims under 20 s old, median energy 72; per-agent kill
hazard per 100 s rises from 0.6% with 4 predators to 1.4% with 13, so the late game is kill-driven.
Pose finding: an early wrong family merge (about 80 units) can make the shared map drift to the wrong frame;
in normal play 1-5% of agents are more than 30 units off, in trap games whole families were 75 off, which
sends baits and guides to the wrong place. Fix this before any further trap work.

## 05:30-06:25: localisation fix and the trap re-test with exact positions

| Item | Result |
| --- | --- |
| anchor() wall-side bug (agent placed inside the wrong wall; 1140 units off in seed 7000) | fixed; agents >30 off 1-3% -> 0-2%; shipped configs +13 ± 18 (no predators, 512 seeds), +16 ± 16 (predators, 576) |
| keeper bait chain with exact positions (192 games) | -100 ± 24 s; more baits worse (-129..-183) |
| guides with exact positions | -93 to -250 |
| trap site chosen near the colony | -43 to -79 |
| final fresh-seed numbers, fixed policy, 768 seeds | no predators 2432 s / 2599; with predators 1533 s / 1543 |

## 06:00-07:10, follow-up items after the map fix (peer-relayed list 1-5)

| Item | Test | Seeds | Result |
| --- | --- | ---: | --- |
| 1 anchor fix ported to orchard.py | lockstep parity Python vs native (parity.py), 6 seeds x 800 s | 6 | zero divergent decisions |
| 2 trap next to the colony (sites <= 250 from the colony centre) | full games, trap_mode 1-3 | 192 | -43..-79 s |
| 3 relay guiding (hand-over to a fresh member ahead on the lane) | scenario 500-700 units, relay actually engaging (rel3) | 64 valid | delivered 92% vs 95% plain guide |
| 3 relay guiding | full games (f35, relay live) | 192 | -174 +- 28 s, same as the plain guide trap (-175) |
| 3 sprint-lead (guide sprints inside the hold band) | scenario (spr1) | 62 valid | 92% vs 89% (noise) |
| 3 wall-clear steering for guides | scenario (rel3) | 64 valid | 95% vs 95% |
| 4 turn-rate escape (committed heading 5/10/20 ticks, with/without facing) | escape grid + full games | 1152 / 192 | -409..-488 s vs the per-tick sidestep |
| 5 late-only trap (from 900/1200 s, 8-13 predators) | full games | 192 | -18..-96 s, 0.03 predators held |
| wall-clear steering for fleeing agents (pred_wallclear) | escape grid (esc4) | 1152 | kills 28.7% vs 27.8% (noise) |
| trap on demand: bait only within 20 s of a sighting, bait-only (trap_mode 2) | full games (f37) | 192 | -57 +- 27 s; 21 baits born per game (gate never binds) |
| trap on demand + guides (trap_mode 3) | full games (f37) | 192 | -140 +- 27 s, 0.06 held |
| trap on demand + guides only for predators within 300 of the trap | full games (f37) | 192 | -158 +- 28 s, 0.07 held; guide episodes 17 vs 23, deaths 7 vs 10 |
| wall-clear steering for fleeing agents | full games (f35 + f36) | 192 + 576 | +42 +- 27 on 192, then -22 +- 15 on 576; pooled -6 +- 13 (noise) |

Full-game trap funnel (f34/f35, per game): 21-26 guide episodes, 19-22 guide deaths before the handoff, 1.6-2.1
reach the handoff state, 0.04-0.24 predators held; kills per game 132 (no trap) vs 140-147 (trap). The scenario
(last 500-700 units, one predator, no bystanders) delivers 90-95%; the full game fails earlier: reaching the
predator, keeping its attention with closer colony members around, and the bait economics.

Relay scenario harness bug (rel1/rel2 were void): the intended guide was frozen during the warm-up ticks, so
the policy picked the relay agent as guide. Fixed in guide.py/guide_dbg.py (relay frozen until the guide is
chosen).

## 07:25-08:15, refuge design (chased agent holds inside a narrow gap; no guide, no bait child)

| Test | Seeds | Result |
| --- | ---: | --- |
| scenario ref1: agent 20-80 from a graded gap, predator 60-100 behind, walk 10/13/16 | 1920 valid | kills 41% -> 20% (r 100), predator held 146/200 ticks; walk 10: 68-74% -> 20-23% at <= 40 units |
| full games ref2: refuge r 60/100, all / slow-only, with posts near gaps | 128 | -146..-284 s; kills UP (141 -> 160-179) |
| ref3 death attribution | 96 | 113 attempts/game: 92 die en route, 10 holding, 0 exiting, 0.3 exits |
| ref4 clear straight path to the gap + sprint | 96 | attempts 20-36, route deaths 8-16; -28 +- 33 (r 100), -44 (r 60), -59 (slow-only); no sprint -125 |
| ref5 safe hold depth for wide gaps / open rears | 96 | -14 +- 38 (r 60), -67 (slow), -85 (r 150), -120 (r 100); hold deaths unchanged 6-14/game |

| ref6 kill-time diagnostic (belief vs truth, NIGHT_REFLOG) | 32 games, 394 deaths | holding deaths: pose error median 0, true distance to the hold point 1.7, predator at 11 => the map's gap is not real geometry; route deaths: pose error median 10 |
| ref7 refuge_verify (hold only while both faces are observed) | 96 | hold deaths 6.7 -> 2.4, aborts 6.4/game; -23 +- 34 (r 60), -28 +- 36 (r 100) |

Conclusion: the refuge is the first trap-family mechanism that is clearly positive in the narrow test, but in full
games it never beats evasion alone. Root causes are map consistency, not the refuge logic: about two thirds of the
map's narrow gaps are not real at the hold point, and runners' poses drift ~10 units on wall collisions.

## 08:30-08:50, the map itself (nightsim/mapcheck.py, 32 games, 600 s, with predators)

| Map code | Phantom face length | Missed face length | Sites per game | Sites valid | Top-1 site valid |
| --- | ---: | ---: | ---: | ---: | ---: |
| as used all night | 37.3% | 48.0% | 23.3 | 15% | 3% |
| wall_conflict=1 (opposite-side observations are conflicts, not twin faces) | 18.8% | 27.5% | 5.2 | 49% | 35% |
| + wall_min_n=20 | 14.5% | 24.7% | 4.9 | 56% | 52% |

Refuge on the corrected map (f38, 96 seeds): rg_60s_wc +13 +- 32 (50/96 wins), rg_100s_wc -17, rg_100v_wc -65.
| guide trap on the corrected map (f39) | 96 | old map -183 +- 39, corrected -180 +- 37, corrected + guides only within 300 of the trap -145 +- 36; held 0.07-0.18, 0.3 handoffs, 20 baits per game |
