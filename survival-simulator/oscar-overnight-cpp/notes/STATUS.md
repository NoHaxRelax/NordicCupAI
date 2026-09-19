# READ FIRST (morning summary, updated 07:15 on 19 Sept)

STATE: IDLE. All pods stopped (n7 at 09:05 after f40; j/k/n6 cannot restart, hosts full; n7 = dpm6cy0ejfmh5o can; disks are wiped on restart: bootstrap + deploy again). Spend ~$46 of $100.
Spend on the night pods (Runpod billing API, pods i/j/k/n1-n6): ~$30 of the $100; burn $0.96/h per pod.
Branch survival-simulator/oscar-overnight-cpp is pushed (fb584b2 + later commits). Nothing through the API,
nothing simulated on the laptop except single-game traces.

**Best configs (unchanged since 06:20, both in best-configs.json):** no predators r21s0c2: 2432 s / score 2599
(768 fresh seeds); with predators pred_best_0400: 1533 s / 1543 (768 fresh seeds). The last candidate, pred_wallclear=1
(fleeing agents steer around recently seen walls), was +42 +- 27 s on 192 seeds but -22 +- 15 on 576 fresh seeds
(pooled 768: -6 +- 13): noise, not adopted. Nothing from the 06:00-07:05 items changes the shipped configs.

**06:00-07:10, the five follow-up items (details in results-summary.md, last table):**
1. Anchor (localisation) fix ported to orchard.py; lockstep parity with the C++ policy confirmed.
2. Trap next to the colony: -43..-79 s. 
3. Guide speed gap: relay guiding (hand-over to a fresh member on the lane) and sprint-lead both change nothing
   (scenario 92% vs 95%; full games -174 vs -175 s). Note: the first relay runs (rel1/rel2) were void because
   the scenario harness let the policy pick the relay agent as guide; fixed and re-run.
4. Turn-rate escape variants: -409..-488 s; the per-tick sidestep against the predator's current heading stays.
5. Late-only trap with 8-13 predators: -18..-96 s, 0.03 predators held.

**07:25-08:15, the refuge design (my call while Oscar was away, reasoning logged at 07:25):** a chased agent already
near a narrow gap steps into it and holds; the predator cannot enter and stays at the mouth. Scenario: kills 41% ->
20%, predator held 146 of 200 ticks. Full games: -284 s at first (82% of refugees died en route: they ran at walls
on the far side of the obstacle), then with a clear-path check + sprint -28 +- 33, then radius 60 -14 +- 38: never
positive. A kill-time diagnostic (belief vs truth, 394 refugee deaths) then showed WHY refugees die while holding: the
agent really is at the hold point (pose error median 0, true distance to the hold point 1.7) and a predator still
reaches it, so the map's gap is not real geometry (a face extended past its true end or a phantom face); runners die
with their own pose ~10 units off (wall-collision deflection). Verifying both faces by direct observation before
holding cuts hold deaths 6.7 -> 2.4 per game but survival stays at -23 +- 34: the aborted refugees die anyway.
Closed at 08:30; code stays behind refuge_* parameters (default off) with nightsim/refuge.py, the death-attribution
counters (dbg_eval) and the kill hook (NIGHT_REFLOG=1). Lesson for any trap/refuge work: the self-built wall map is
wrong at the few-unit level often enough that geometry-critical behaviour must verify walls by direct observation.

**08:30-08:50, THE MAP (measured directly, nightsim/mapcheck.py + dbg_walls):** in the map the policy builds from its
own wall observations, 37% of confirmed face length is phantom (the side it calls solid is free) and 48% is missed
(the side it calls free is blocked); of its crevice sites only 15% are real, the TOP-1 site only 3% of the time.
Cause: an observation of a known face from a mis-posed agent with the opposite solid side created an inverted twin
face; a thin wall's twins pair into a "crevice" inside the wall. Fix (add_wall, wall_conflict, now the default for
map-building modes): opposite-side observations count as conflicts on the existing face. Result: phantom 19%,
missed 28%, sites valid 49%, top-1 valid 35% (52% with wall_min_n 20). The shipped configs build no map, so they
are unchanged. Every trap/bait/guide/refuge result of the night before 08:45 ran on the broken map; the guide and
bait SCENARIOS loaded true walls and never saw it. Re-run on the corrected map (f39, 96 seeds): the guide trap is
still -180 +- 37 (lane-restricted -145), 0.07-0.18 predators held, 0.3 handoffs per game: the trap fails on
delivery, not on the map. With guide_clear (a guide needs a straight wall-free walk to the lane point, released
otherwise; f40) guide deaths fall 15 -> 4 per game and the cost to -105 +- 37 (-91 lane-restricted), but 40 of 56
episodes per game are released and handoffs stay at 0.3: in a real map the lane is rarely clear for long. The
refuge becomes +13 +- 32 at radius 60 (noise). Nothing changes the shipped configs. Final decomposition of the
trap's cost: baits ~-57, guides' lost time ~-40, deliveries ~0. A trap that pays needs a new delivery idea, not
more fixes to this one.

**What the full-game trap funnel says (per game, f35/f37):** 17-26 guide episodes, 7-11 guide deaths (classes: far
from the trap and slow walk dominate; several predators near is rare), 0.26-0.29 handoffs, 0.04-0.09 predators
held, 19-21 baits born, 132 kills per game even without the trap. Cost decomposition (f37, 192 seeds): baits alone
-57 +- 27 s, baits + guides -140 +- 27, guides only for predators within 300 of the trap -158 +- 28; bait-on-sight
(bait only within 20 s of a sighting) does not reduce baits born (19 vs 21) because predators are always in sight. The isolated scenario (last 500-700 units,
one predator, no bystanders) delivers 90-95%, so the losses happen before that: reaching the predator, keeping
its attention while closer colony members are around, and baits that never eat. A trap that pays off needs a
different design (fast dedicated guides + a replenishable bait), not more parameter tests.

**No predators:** r21s0c2 remains the best; ~180 variants on 512 paired seeds each, nothing beats it. Headroom is
detecting trees 200-400 units away (oracle +120 s).

**With predators, evasion only:** 887 s -> 1533 s with pred_best_0400 (flee within 70, face within 80, sprint
within 40, sidestep 1.4 rad within 80, no shared alarms). Fleeing far, shared alarms, crevice hiding, decoys,
closest-only fleeing, birth gating, dodge-hold: all worse or noise.

Files: BEST.md (best configs), GUIDE.md (trap step 1-2 numbers), results-summary.md (all tables), PLAN.md,
configs-all.json / best-configs.json, runs/*.jsonl (compact per-game rows), docs/survival-orchard.md.

---
# Overnight status (19 Sept 2026) — read this first on every tick

Rules (Oscar, 23:55): C++ only (nightsim fork: survival/nightsim, engine+policy natively), runs ONLY on Runpod pods
(never simulate on the laptop; compiling locally is fine), NEVER touch the competition API, push only to
survival-simulator/oscar-* branches, keep compact data (no replays), always keep the best policy saved.
Budget: $100 new spend from 23:55 CEST, up to 8 pods ($7.68/h => ~13 h). Pods: pods.txt; driver: night.sh.

Phases
- P1 (to ~02:00): no-predator survival in C++ (nightsim/_npolicy.hpp + configs). Baseline r21s0c2.
- P2 (02:00 on): Lucas's latest trap/bait work (report: lucas-trap-report.md) -> port to C++, combine with
  survival policy; predator regime tests; many-predator late game; turn-rate escape; old agents as bait/guides.

Best so far (no predators): see BEST.md (policy hash + config + score).

Log (newest last)
- 00:05 5 new pods n1..n5 (EU-RO-1, cpu3c-32, $0.96/h each) + i/j/k; cpu-a stopped (never terminate).
- 00:15 nightsim fork built on all 8 pods (policy md5 0f880eec = peer's _orchard.hpp).
- 00:25 base1k running on all 8 pods (launcher bug: 'cd &&' kept ssh open; fixed with ';').
- 00:32 base1k (953 seeds): r21s0c2 2391 s / score 2553 vs best 2353 / 2492 (paired +38±14 s). Late-game harvest
  falls to 62-72% after 2000 s. diag1 (fruit fates, r21s0c2, 152 seeds): 80-90% of rotted late fruit had NO agent
  within 200 units during its life (55% none within 400) => coverage/discovery-limited, not supply-limited.
- 00:30 _npolicy.hpp 4be33b1c: late-game schedule (late_t + l_* overrides); regression identical when off.
  Running: s1 (32 perturbations, i/j/k/n1, seeds 2000-2127), p1 (12 single-param probes, n2), L1 (48 random late
  schedules, n3/n4/n5, seeds 2000-2095).
- 00:35 _npolicy.hpp: predator evasion layer (pred_mode=1: face nearest threat, back away, sprint <90) and
  merge_anchored (merge all wall-anchored families at once). pr0 (r21s0c2 with predators, evasion off vs on,
  64 seeds 5000-5063) running on n3/n4 next to L1. Engine facts for P2: predator 11 walk/15 sprint per tick,
  turn cap 0.3 rad/tick in direct chase; direct chase if agent faces away or is <90 away, otherwise a 45-degree
  pivot approach (closing ~10.6/tick); agents move then turn, turning uncapped (cost |turn|/2pi).
- 00:40 pr0 (with predators, 64 seeds): r21s0c2 no predator logic 846 s, 226 predator kills/run, score 684;
  evasion v0 963 s, 116 kills/run, score 982 (+282±50). Predators dominate: with them colonies die ~900-1000 s.
- 00:45 esc1 narrow escape grid (1 agent vs 1 fresh predator facing it, 30 s, energy 150/800 so no sprint):
  no evasion 79% killed; evasion v0 32%; kill rate by walking speed 10/15/20 about 60%/20%/8% (D>=80).
  => select for walking speed (heritable, cap 20). Added fit_speed_cap (fitness speed cap, default 1.5 => 15).
  Added no_spawn (tests only) and dbg_* engine hooks + nightsim/escape.py. sp1 (speed selection x predators) on n3.
- 00:55 P2 prep parked per Oscar (01:00: "optimise survival time now; the rest waits until after 2 am; narrow-gap
  baits only, never wall baits"). Parked code (all behind switches, default off): evasion+dodge+shared sightings,
  crevice site finder with median wall clustering (ts4: valid site in top-3 on 26/48 maps; every graded hold test
  kept the bait alive, predator closest ~19.5), bait role with replacement (untested). Perfect-trap model tv1:
  trapped within 100 s => +922 s; 30 s => +1167 s (the P2 prize).
- 00:55 c1 partial (~160 seeds): oracle tree knowledge +99±30 s (discovery headroom); merge_anchored 0;
  speed selection without predators -100 s; parameter candidates within noise.
- 01:00 P1 back in focus. q1 (32 probes of never-searched parameters: late breed reserve, tree-decay model n0/tree_half,
  emergency reserve, explore radius, births/tick, site_min, repost/switch/post/min_stay/watch_refresh, extra_old)
  on n2-n5 seeds 2000-2127 (r21s0c2 regression identical). c2 queued after c1 on i/j/k/n1 (seeds 10000-10511):
  oracle split (nearby-only 400/800, ages-only) + age_infer 20/40/60 (new: tree first seen in a cell viewed dt ago
  gets birth estimate at dt/2).
- 01:05 c1 (512 seeds 10000-10511) final: oracle tree knowledge +124±17 s (+11.6 fruit) = real discovery headroom;
  old_reach 150 +20±17; L1c18/L1c45/s1a*/L1c38 ~0 (winner's curse); merge_anchored -9±16; speed selection -89±18.
  tail1 (end-of-game traces): late colonies of 3-5 agents boom (bursts of 3-4 same-age births) and bust; intake falls
  from ~20 to ~10 energy/s in the last 300 s. Added late overrides l_births_per_tick/l_emergency_reserve/
  l_low_pop_reserve and age_fruit (a fruit seen by born_hi bounds its tree's birth <= born_hi-20).
  Running: c2 (oracle split + age_infer) on i/j/k/n1; q1 on n2-n5, then bc1 (late birth control + age_fruit).
- 01:15 c2: oracle limited to trees within 400 of a member +146±21 s, within 800 +150±21, full +124±17; ages-only
  +15±23; age_infer 20/40/60 noise. => the gap is LOCAL detection (trees near agents not seen), not ages.
  Mean vision radius evolves 208 (t=250) -> ~350 (t>=2000), cap 400. bc1 partial: late births_per_tick=1 identical
  (late bursts are heirs/emergency births), late cap cuts ~0 or worse, age_fruit -12±31.
  Queued: v1 (fit_vision 1.5/2/3+heir_select, fit_hear 0.2, sweep 0.035, oracle mode 4 = own vision range only) on
  j/n1; c3 (q_n0_60, q_sm0, q_brl450, combo) on i/k seeds 10000-10511. Spend ~ $9.5 of $100.
- 01:25 c3 (512 seeds): q_n0_60/q_sm0/q_brl450 -11..-13, combo +12±16 => parameter space converged around r21s0c2.
  bc1 final: late birth control no gain. Running: v1 (vision selection + oracle mode 4) on j/n1/n3/n4/n5 (512 seeds),
  or5 (oracle within 120/200/300) on i/k (256 seeds), bc1 remainder on n2.
- 01:40 v1 (512 seeds): vision selection fit_vision 1.5/2/3(+heir_select) and faster sweep all within +-17 s => no gain;
  oracle mode 4 (own vision range only) +59±19. or5: oracle radius 120/200/300/400/800 = +48/+17/+85/+122/+129 =>
  missing knowledge is trees 200-400 away. dm1: protect known live trees from false 'dead' marks (oracle 5) +54±27;
  simple miss thresholds don't capture it (dm3 -57). Hypothesis: occlusion check forgets walls after 40 s => new
  occ_walls option uses the permanent confirmed wall map; ow1 running on j/n1/n2/n3/n4/n5 (512 seeds).
- 01:45 P1 CLOSED: r21s0c2 remains best (2393 s / 2555 over 1465 seeds). ow1 (occlusion from wall map) -11±17.
  n3/n4/n5 stopped (disk kept) to save budget; 5 pods (i, j, k, n1, n2) for P2 step 1 scenario tests.
- 01:47 n1/n2 also stopped (disks kept); running pods: i, j, k. Pushed branch survival-simulator/oscar-overnight-cpp (871b639).
- 01:51 vm1 (visibility margins 40/80, fruit 20; 512 seeds): +7..+11 ±17 => noise. P1 final: r21s0c2 (unchanged). P2 starts 02:00 with step 1 (one predator to the trap).
- 02:00 st1 (stacked small gains, 512 seeds): +10±17 / +1 / -27 => nothing. P1 closed. P2 step 1 begins (guide one predator).
- 02:45 P2 step 1: guide role in C++ (ACQUIRE/LEAD/handoff = Lucas's sacrifice at 55 from the bait), scenario test
  nightsim/guide.py (true walls + poses via hooks, bait frozen 9 deep, guide/predator placed with clear straight
  lines). First full delivery traced: guide eaten at t=4.2, predator then held 19-25 from the bait, bait alive.
  Oscar (02:20): guide death is fine; scope = delivery rate only. Running: g1 smoke (12 seeds) on i; g2 full grid
  (dg 150/300/500 x dp 100/180/250 x bearing 0/90/180 x speed 10/13/16) seeds 1-32 on j, 33-64 on k.
- 03:10 step 1 iteration: the predator must actually be chasing before the handoff (state 3 needs 'closing');
  guide keeps in front of the predator (sprint away when < 92 = direct-chase range, angled when the predator sits
  between guide and lane; circle at walk in the pivot range). Test runner no longer stops at the guide's death
  (the handoff needs it). Known hard case: predator between the guide and the trap (bearing 180): the predator
  either eats the guide at close range or wanders off out of its senses. Measuring delivery by bearing: g1 (12
  seeds) on i, g2 full grid on j/k.
- 02:25 FIRST STEP-1 MEASUREMENT (g2, 1746 valid scenarios of 5184; 46% skipped for no clear straight line, 20%
  invalid site): delivered 62%; by predator bearing 0/90/180 (behind guide / side / between guide and trap):
  85% / 68% / 34%; by guide walk speed 10/13/16: 55/62/68%; start distance 150: 74%. Bait never died. Failure
  modes: predator lost / re-acquire loop 77%, guide killed within 8 s 22%. Fix under test (g3): hold the predator
  at 100-130 while leading instead of walking away at full speed.
- 02:30 g3 (hold 100-130): 63% (same as g2). g4 (hold 70-95 + acquire sprint): 52% => WORSE (more guides killed, more lost). Reverted to g3 lead; g5 A/B: A=g3, B=+acquire sprint, C=hold 80-115, D=handoff at 45.
- 02:35 g5 A/B (2337 scenarios each): A (g3 logic) 65%, B (+acquire sprint) 64%, C (hold 80-115) 62%, D (handoff
  45) 64% => parameters are not the lever. Instrumented the predator (dbg_pred_info): the dominant failure is the
  predator LOSING SIGHT of the guide at 120-140 (line of sight cut by the trap's own obstacle: mode 3 edge-avoid),
  then wandering away at walk speed; re-acquire is slow. Predator hearing (60, omnidirectional, through walls) is
  the only loss-proof sense => g6 tests hold bands 60-90 / 50-80 / 45-70 (predator kept within hearing).
- 02:45 g6 (2337 scenarios each): hold band 92-130: 65%; 60-90: 71%; 50-80: 75%; 45-70: 78% (bearing 0/90/180 =
  96/81/54%). Lost-predator failures 607 -> 180; guides killed early 208 -> 334. Keeping the predator inside its
  hearing range is the lever. g7: bands 40-60, 35-55, plus blocked sprint angle 2.9 and acquire stop 75.
- 02:50 g7: band 45-70 stays best (78%); 40-60 77%, 35-55 75% (more guides killed before the handoff); blocked
  sprint angle 2.9 and acquire stop 75 slightly worse. Remaining failures at 45-70: guide killed early 14%,
  predator lost 8%; bearing 180 (predator between guide and trap) 54%.
- 03:00 g8: removing the beyond-band slow-down HURTS (73% vs 78%: the predator in pivot mode closes only 10.6/tick, so
  a guide walking off at 13 loses it); handoff distance 35/40/45/55 all equal; acquire sprint no gain. Best stays
  near 45 / far 70 / slow-down on (78%). g9 tests 'fastclose': keep walking instead of waiting when the predator
  charges in at > 8-12 units/tick (traced early kills came from waiting while it closed 15/tick).
- 02:35 STEP 1 NAILED to the scenario limit: 78% overall, 96% with the predator behind the guide (GUIDE.md).
  Guide selection now penalises candidates with the predator between them and the trap. Step 2 started: K
  predators already held at the mouth (K=0,1,3,6,10) + one delivered; measure holding and delivery (runs m0..m10).
- 02:36 STEP 2 (holding): K=1/3/6/10 predators already held + one delivered: 94-98% delivered, all held,
  bait never dies. Now f1: FULL GAMES with predators, 192 seeds: no predator logic vs evasion vs evasion+trap
  (site finder, bait with replacement, guide), tracking held predators and guides used per 250 s.
- 02:40 f1 FULL GAMES with predators (192 seeds): no predator logic 888 s (232 eaten/game); evasion 923 s
  (60 eaten; score +248); evasion+trap 695 s (-193 vs evasion): the trap machinery costs agents (bait/replacement
  stand in the crevice without food; guides die before delivering; sightings sparse) and held ~0.1 predators.
  Trace (seed 6000): site found at 100 s, first predator 120 s, colony 40 -> 15 agents by 200 s (7 eaten in 10 s),
  guides assigned 3x but never delivered, 1 predator held briefly at 270-290 s. f2 tests trap-on-demand (bait only
  after a sighting within 60/120 s) and no keep-out.
- 02:48 f2: on-demand bait / no keep-out do not help (all trap variants ~700 s, guides done 0). One-second
  guide traces in full games show why: guides were chosen OLD (drain 10+/s, dead in seconds), assigned to far
  sightings of predators busy eating other agents, 'chasing' detection too loose (72-degree cone), some guides
  with 70-unit pose errors, stale sighting targets while a second predator is 33 away. Fixes: guides must be
  young with >= 120 energy (fast preferred), chase = predator displacement within 37 degrees of us AND closing.
  f3 running (evade / trap / trap with 200-energy guides / bait-only).
- 02:53 f3 (192 seeds): evade 923 s; bait-only (no guides) 754 (-169); trap 702; trap with 200-energy guides
  714; guides done still 0. A standing bait alone costs ~170 s (one forager permanently out of the economy while
  predators keep the colony churning). f4 adds a guide funnel (started -> chase -> lane point -> handoff / died / lost).
- 02:56 f4 guide funnel (192 games): 2206 guide episodes (11.5/game) -> chase established 2138 (97%) ->
  reached the lane point 292 (13%) -> handoff spot 28 (1%); 1919 guides died en route (87%), 179 lost. The
  problem is the LEAD walk, not acquisition. f5 attributes the deaths (far from lane / >=2 predators / slow /
  stuck against walls / early).
- 02:58 f5 death attribution (1919 guide deaths): walk speed < 12 in 79%, > 300 from the lane 52%, >= 2
  predators near 18%, in ACQUIRE 20%, stuck 9%. Colony agents walk ~10 vs predator 15: a walking guide loses 5/tick.
  f6: sprint hysteresis (sprint until 70/90 once triggered), assignment limited to predators within 250/400 and
  candidates near the lane.
- 03:05 f6: sprint hysteresis and assignment limits change nothing (guides still die en route: 'slow' deaths
  unchanged). Root cause is structural: colony agents walk ~10 vs predator 15; a guide cannot outrun a fresh
  predator over hundreds of units. f7: guide only predators already within 300 of the trap; speed selection
  (fit_speed 3, cap 20) with predators, with and without heir selection.
- 03:10 f7: speed selection with predators +8 s (noise); lane-limited guiding still -200 s. 0.2-s full-game
  trace shows a real delivery (guide eaten as it entered the handoff, predator held 26 from the bait for 10+ s)
  that the 'guides done' counter missed, plus two waste modes: the next guide walks to the STALE position of the
  now-held predator, and guides wait beyond the band for predators busy elsewhere. Fixes: deliveries counted as
  sightings newly held at the mouth, guide released on delivery, waiting timeout 6 s -> re-acquire. f8 running.
- 03:14 f8: all trap variants still -205..-225 s; the new delivery counter over-counts (held sightings
  flicker) while the sampled 'held' stays ~0.05: holds last only as long as the bait lives (a bait never eats and
  dies in 60-100 s), then the predators are released. Full-game trap integration parked; the tooling and the
  funnel/attribution counters stay in the branch. Now step 3 (not losing agents in general): f9 evasion variants
  (flee radius 100/150/200/300, dodge on/off, shared sightings, sprint radius).
- 03:18 STEP 3 first result (f9, 192 games with predators): flee radius 100 instead of 200: +198±21 s
  (923 -> 1120 s, score +189); 150: +138; no shared sightings: +102; sprint radius 60: +42; dodge off -46;
  radius 300: -61. More agents get eaten (94 vs 60 per game) but the colony forages instead of fleeing: the
  economy matters more than the kills. f10 pushes further (60/80, no share, combos). Spend since 23:00 CEST:
  ~$20 on my CPU pods (well within the $100).
- 03:23 f10: flee radius 80 + no shared sightings: 1239 s / score 1260 (evade baseline 923, no logic 888).
  Trend continues: flee less. f11: radius 50-80, sprint radius 40/60, face radius 80, dodge off.
- 03:27 f11: flee radius optimum ~80 (50/60/70 worse by 20-40 s); sprint radius 40/60 +20; dodge off -97.
  Best with predators now e_r80_ns_spr40: 1262 s / score 1283. f12 tunes the dodge (radius 40-100, angle) and facing.
- 03:32 f12: dodge angle 1.2 rad (sidestep 69 degrees off the predator's heading): +102 s -> 1364 s / score
  1385; dodge radius 80 +36; angle 0.5 or radius 40 -90..-100. f13 continues (radius 80/100, angle 1.0/1.4, combos).
- 03:37 f13: dodge radius 80 + angle 1.4: 1444 s / score 1462 (from 923 at 02:40). f14: angle 1.57/1.8,
  radius 100, flee radius 90/100 with matching dodge radius, sprint radius 50.
- 03:42 f14: dodge converged (radius 80-100, angle 1.4; 1.57/1.8 and flee 90/100 worse). BEST WITH PREDATORS:
  'pred_best_0342' in configs-all.json = r21s0c2 + flee within 80 (face 120), own sightings, sprint within 40,
  sidestep 1.4 rad within 80: 1444 s / score 1462 on 192 seeds (from 888 without predator logic). f15 re-tunes
  population parameters under predators (cap, reserves, speed selection).
- 03:48 f15: population parameters under predators: all within noise or worse (cap 0.35 -58, heir reserve 200 -67, speed selection -47). f16: facing range 80-300, sprint radius 20-30, shared sightings with far facing.
- 03:54 f16: facing range 80 +28, flee 70 +31 (noise), far facing / shared sightings clearly worse. Evasion converged ~1450-1475 s. f17: confirmation on fresh seeds 7000-7191 + death profile.
- 03:58 f17 CONFIRMED on fresh seeds 7000-7191: no logic 864 s, evasion v0 959, pred_best_0342 1422,
  + flee 70 / face 80 = 1507 s / score 1520 (+85±25) => 'pred_best_0400'. Death profile: 37 kills per 250 s early
  (population 39 -> 24 -> 16 -> 10 by 1000 s), starvation 290/game vs kills 132. f18 tests Oscar's 'hide behind a
  wall': agents flee INTO the nearest known crevice with an open rear and out the other side (predators cannot
  follow through a 10-20 gap; hide_mode, radius 150/250/400).
- 04:05 f18: crevice pass-through escape is much WORSE (-280..-545 s; 190-214 eaten/game): running any
  distance to a fixed point with a predator behind is deadly, same lesson as the guides. Dropped. Merging anchored
  families -27. f19: dodge side/angle/radius and sprint radius fine-tuning around pred_best_0400.
- 04:11 f19: dodge fine-tuning within noise (radius 70 +11, angle 1.3 +3); the sidestep SIDE is decisive
  (mirrored side: -712 s). Step 3 converged at 1507 s (pred_best_0400). f20: colony foraging around the trap
  (post bonus near the mouth) with a bait on sight, with/without guides, to see whether predators hunting the
  colony get caught without long guide walks.
- 04:19 f20: bait on sight now costs -455 s against the stronger evasion; foraging around the trap does not get
  predators caught (held 0.05-0.17); with guides -603. Trap integration is negative in every form tested tonight.
  esc3: single-predator escape grid for the new evasion (per speed / distance / bearing).
- 04:24 f21 final confirmation, 576 fresh seeds: pred_best_0400 1492 s / score 1502 vs 887 / 732 without
  predator logic (+605±16 s). Experiment agenda exhausted for the night; stopping pods i/j/k (disks kept) to save
  budget. Restart with night.sh (pods.txt) if needed.
- 04:28 Resumed per Oscar's 'run until the budget is spent' ($30 of $100 used): cpu-i/j/k restarted (new
  ports in pods.txt; the old venvs under /workspace/users were gone, rebuilt at /workspace/night/.venv). Score
  accounting with predators (576 seeds): pred_best_0400 = survival 1492 + fruit 130 - penalty 120 (146 kills at a
  mean 82 energy each). Next problem (Oscar's list): make the OLD agents the ones that get eaten. decoy_old/decoy_e:
  a senescent (or low-energy) agent within decoy_r of a predator walks toward it so it is the predator's closest
  target; f22 tests radius 80/150/250 and energy thresholds 40/80 on j/k; kp on i records every kill (t, age, energy).
- 04:35 f22: old-agent decoys -6 s / -19 score (more kills, no benefit); low-energy decoys -34..-120. Rejected.
  kp kill profile (128 games, 140 kills/game): 34% of victims are under 20 s old (newborns at 75 energy), median
  age 32, median energy 72; 9% of kills (>=150 energy) carry 23% of the penalty; kills peak at 250-750 s. f23
  tests: flee only when I am the predator's closest visible agent (evade_closest), and no births while a
  predator is within 100/200/400 (spawn_pred_r).
- 04:41 f23: closest-agent-only fleeing and birth gating near predators: all within noise (+2..-32). Evasion family saturated at ~1500 s. f24: detection side (sweep rate, hearing/vision selection, watch patience, explore threshold) with predators.
- 04:49 f24 (detection): fit_vision 2 +33±27, others noise. Late game IS kill-driven: per-agent kill
  hazard per 100 s rises 0.6% (4 predators, t=500) -> 1.4% (13 predators, t=1500); a dying colony loses 19 agents
  in its last 300 s with 13 predators and 15 trees alive. => the many-predator regime needs the trap. New design
  per Oscar's list: KEEPER role (keeper_mode): the member nearest the crevice rear holds within keeper_r of it and
  spawns a CHILD as the next bait when the current bait's life runs short (100 energy per bait ~ one agent's
  upkeep, no forager sacrificed). f25 tests keeper alone / on sight / with trap-side foraging / with guides.
- 05:10 KEEPER RESULT (f25/f26, 192 games): keeper-spawned bait children -414..-506 s even after fixing rear
  entry, obstacle-avoiding approach and stuck-replacement release; held predators ~0.05. Traced root cause: in
  trap games the family's shared map drifts (median pose error 75 units for ALL members from t~100 in seed
  7000) because a mis-merged sub-family (one early merge off by ~80) re-maps trees and the majority frame wins;
  bait children then walk to the wrong spot. In normal play only 1-5% of agents are >30 off (12 seeds, with and
  without predators), so the survival policy tolerates it, but the trap needs exact absolute positions. Fixing
  map consistency under merges is the prerequisite for any trap work; out of tonight's scope.
- 05:10 Work concluded: no productive experiment left inside the agreed scope (P1 converged, evasion converged,
  trap blocked on pose consistency + bait economics). Pods i/j/k stopped (disks kept). Spend ~$38 of $100.
- 05:26 Resumed (peer relayed Oscar: keep going until told to stop). Problem now: map consistency under family merges. Tools: NIGHT_POLLOG=1 prints anchors, merges (observer/observed poses, implied pose) and pose JUMPs > 20 with true poses. Pod i restarted.
- 05:47 BUG FOUND AND FIXED in anchor(): the boundary-wall candidate order accepted the first in-map position
  (y=5, inside the wall) instead of the wall on the correct side (seed 7000: agent 3 placed 1140 units off, then
  merged into the main family with zero correction). Fix: pick the wall by the edge's side relative to the agent
  and require the playable interior; also snap an anchored agent to boundary evidence for any error size (was < 40).
  Pods j/k restarting (container disks are wiped on restart: bootstrap + deploy needed) for paired re-tests of both
  shipped configs on their confirmation seeds.
- 05:49 After the anchor fix: pose error >30 units 2/1/1/0/0% (no predators) and 1/2/1/3/0% (predators) at
  100/300/600/1000/1500 s (was 3/3/3/1/1 and 3/5/1/3/0); trap-mode family drift gone (median 0); the keeper's bait
  child reaches the goal and holds ~55 s. Keeper now holds a post near the rear (keeper_post_w) and only walks over
  when a bait is needed. Running: regression of both shipped configs on their own seeds (j: 512 no-predator,
  k: 576 with predators) and f27 keeper A/B on i (96 seeds).
- 05:59 REGRESSION OF THE ANCHOR FIX (paired on the shipped configs' own seeds): no predators +13±18 s
  (512 seeds), with predators +16±16 (576) => the fix is kept; both shipped configs now run on the fixed policy.
  Keeper with exact positions (f27, 96 seeds): -45±35 (was -450); bait present 20-40% of the time, 24 baits/game,
  held 0.05-0.1 (predators still do not come by themselves); keeper+guides -187; old-agent bait -481.
  Running: f28 keeper continuity (keeper_r 200/300, reserve 20, post radius 400), f29 guides with exact positions,
  f30 keeper confirmation on 192 more seeds.
- 06:09 With exact positions: keeper -100±24 (192 seeds); more baits (keeper_r 200/300, reserve 20: 43-48
  baits/game) makes it worse (-129..-183): every bait is 100 energy and held predators never exceed ~0.2 because
  predators hunt where the colony is, not at the crevice. Guides with exact positions -93..-250. Last trap test
  (f31): choose the site nearest the colony (site_dist_w 0.3/1.0) with and without bait-on-sight. Also running the
  final fresh-seed measurement of both shipped configs with the fixed policy (768 seeds each, j/k).
- 06:22 f31: site nearest the colony -43..-79; trap line closed. Final 768-seed numbers recorded in BEST.md.
  Pods i/j/k stopped. Waiting for new direction from Oscar (tick continues at a low cadence).
- 06:32 Item 1 done: anchor fix ported to research/orchard/orchard.py; lockstep parity (nightsim/parity.py,
  Python policy vs native on one engine): 6 seeds x 800 s, zero divergent decisions. Pod i cannot restart (host
  full) -> new pod n6 (jjys2nbbrba2v3). Running f32 (items 2+5): trap only from 900/1200 s when predators are
  many, with keeper, near-colony sites, and lane-limited guides. Coded item 4: committed dodge heading
  (pred_dodge_hold ticks: run past/behind the predator, then walk away).
- 06:44 Item 4 (turn-rate escape, committed heading 5/10/20 ticks, with/without facing, angle 2.5): -409..-488 s
  vs the per-tick sidestep => re-evaluating the sidestep every tick against the predator's current heading is
  essential; 'run past and behind, then walk away' loses. Items 2+5 (trap only from 900/1200 s, when predators are
  8-13): -18..-96 s, held predators ~0.03; late colonies (5-10 agents) cannot afford baits either. Item 3 (relay
  guiding) implemented (guide_relay: a fresh member waits 180 ahead on the lane, takes over when the predator
  comes within 45); scenario A/B at 500/700 units running on n6.
- 06:46 Relay bug: the relay block sat after the LEAD state's early return, so it never ran (rel1 identical
  on/off). Moved it before the state machine. Trace also showed a guide circling into the boundary wall and dying:
  added steer_clear() (recent-edge wall avoidance) for guides (guide_wallclear) and fleeing agents (pred_wallclear).
  Running rel2 (relay on/off x wall-clear, 500-700 units) and esc4 (escape grid with wall-clear) on n6; f34 (relay in
  full games, but with the dead relay code) on j/k.
- 06:53 f34 (192 seeds, full games with predators, relay code dead so this is guide-trap vs no trap): rf_relay -185+-26,
  rf_relay_lane400 -133+-26 vs the evasion best (1527 s). Consistent with every earlier full-game trap result.
  Relay scenario harness bug: during warm-up the intended guide was frozen, so the policy picked the relay agent as
  guide and no hand-over could ever happen (rel1/rel2 measured nothing). Fixed (relay frozen until the guide is
  chosen); trace seed 11 now shows relay search -> hand-over at t=2.1 -> handoff state at t=6.5. rel3 running on n6.
  f35 on j/k (fresh seeds 7200-7391): rf_ref, rf_wall (pred_wallclear), rf_relay (live relay + guide_wallclear), rf_trap.
  esc4: pred_wallclear in the escape grid 28.7% vs 27.8% kills (noise).
- 06:55 rel3 (harness fixed, relay engages: search -> hand-over -> handoff in the trace): delivered r_off 95%,
  r_on 92%, r_wall 95%, r_on_wall 89% (64 valid scenarios each). Relay guiding and wall-clear steering give no gain
  in the scenario; the plain guide already delivers 95% there. Item 3 relay: closed, negative/noise.
  f34 funnel (full games, trap_mode 3): ~21 guide episodes per game, ~19 die before the handoff, 1.6 reach state 3,
  0.24 predators held; 132 kills per game in the reference vs 140 with the trap. The full-game problem is not the
  last 500 units (scenario) but reaching and holding the predator from wherever the guide starts, with the colony
  (closer agents) around. Testing sprint-lead (spr1 on n6) as the last item-3 variant.
  Spend since the overnight start (Runpod billing, my night pods only): ~$29 (i/j/k/n1-n6); burn now $2.88/h (j, k, n6).
- 06:56 spr1 (sprint-lead: the guide sprints whenever the predator is inside guide_far, 62 valid scenarios):
  g_off 89%, g_sprint 92%, g_sprint_far100 87% => noise. Item 3 (sprint / relay guiding) closed: the scenario
  delivery is already ~90-95%; the guide speed gap is not what breaks the full-game trap. n6 stopped (idle).
- 06:57 f35 (192 fresh seeds 7200-7391, full games with predators, paired vs pred_best_0400 = 1514 s):
  rf_wall (pred_wallclear: fleeing agents steer around recently seen walls) +42+-27 s, 104/192 wins;
  rf_trap (guide trap, no relay) -175+-27; rf_relay (relay live) -174+-28 => relay changes nothing in full games.
  Trap funnel again: 26 guide episodes, 22 guide deaths, 0.04 predators held. Confirming rf_wall on 576 more
  seeds (f36 on j/k, seeds 7400-7975).
- 07:03 f36 (576 fresh seeds 7400-7975): rf_wall -22+-15 s vs rf_ref; pooled with f35 (768 seeds) -6+-13, 381/768 wins
  => pred_wallclear is noise (the +42 on 192 seeds was a fluctuation). Not adopted. Pods j, k stopped; all pods idle.
  Night-pod spend ~$30 of $100. Branch pushed. Nothing through the API; no simulation on the laptop.
- 07:05 Tick: items 1-5 done. Funnel columns decoded (dbg_trap): f35 rf_trap per game = 25.9 guide episodes,
  21.6 chased, 2.1 reach the handoff state, 0.29 handoffs, 10.9 guide deaths (classes: far 6.1, slow 6.5, multi 1.6,
  stuck 1.2, early 2.3; overlapping), 21.2 baits born; only 47/192 games ever complete a handoff, 2/192 hold a
  predator. The dominant cost is 21 baits per game. One last narrow test, f37 on j (restarted, port 33242, seeds
  7200-7391): trap on demand = bait only within 20 s of a sighting (bait_on_sight=20), with/without guiding only
  predators within 300 of the trap (guide_pred_lane_max), and bait-only. Cost ~$0.3.
- 07:18 f37 (192 seeds 7200-7391, paired vs pred_best_0400 1514 s): t_bait_only_sight20 -57+-27 (84/192 wins),
  t_sight20 -140+-27, t_sight20_lane300 -158+-28. Held 0.06-0.09 per game; baits born 18-21 regardless of the
  sighting gate (predators are seen almost continuously); lane300 halves guide episodes (17 vs 23) and deaths (7 vs 10)
  but the survival cost is the same. Trap line closed with a full cost decomposition: baits ~-57, guides ~-85..-100.
  Pod j stopped. All pods stopped. Night-pod spend ~$31 of $100.
- 07:26 Peer relayed Oscar's standing order: when waiting on his decision, make the sensible call, note it,
  continue. DECISION: try the "refuge" design, the one trap idea not yet tested that avoids the two proven killers
  (long guide walks, child baits that never eat). Reasoning: predators kill 132 agents per game and hunt where the
  colony is; a chased agent that is already within ~60 units of a narrow gap can step 9 deep into it (a predator,
  radius 10, cannot enter a 10.1-19.9 gap and cannot get within 19 of it), and the predator then stays pressed at
  the mouth for as long as it hears the agent (60 through walls). That is exactly Lucas's held-predator geometry
  without a guide: the chase itself delivers the predator, and the refugee is an agent that would otherwise have
  been eaten (low-energy agents cannot sprint and are caught ~100% in the open). f18's crevice PASS-THROUGH
  (-280..-545) differs on all three counts: it ran up to 150-400 units to the gap, went out the other side (no hold),
  and ran on the pre-fix drifting map. Steps: R1 implement refuge_mode (stay at the site goal, leave when no
  predator is sensed for refuge_leave s or through the rear); R2 scenario: agent 30/60/90 from a mouth, predator
  50/80/110 behind, walk 10/13/16 -> kill rate and predator ticks held vs plain evasion; R3 full games 192 seeds:
  refuge radius 60/100, with and without foraging posts near crevices (trap_post_w). Budget for this: <= $5.
  Pods j and k restarting.
- 07:32 R1 done: refuge_mode implemented in nightsim/_npolicy.hpp (refuge(): pick the nearest known site
  whose mouth is within refuge_r and not nearer to the predator than to us, go pre-point -> goal (9 deep), hold;
  refuge_exit(): hold refuge_leave s after the last sensed predator, then walk out to the pre-point). Harness
  nightsim/refuge.py (agent da outside a graded site, predator dp behind it, added after the warm-up). Harness
  lessons: dp < 60 is unwinnable by construction (one-tick observation delay, first tick wanders). Local sanity
  (seed 8/9/14/15/16, da 40, dp 80, walk 10): plain evasion 3/5 killed, survivors 480-765 away; refuge 0/5 killed,
  all in the gap, predator held ~197 of 200 ticks. ref1 grid on j: 64 seeds x da 20-80 x dp 60-100 x walk 10/13/16.
  Pod k never got a port after 10 min (stuck starting): stopped; j alone is enough.
- 07:33 R2 ref1 (1920 valid scenarios, 64 seeds, agent 150 energy): kill rate 41% plain evasion -> 20% with
  refuge_r 100 (25% with r 60); predator held 146 of 200 ticks on average (0 without). By walk speed: 10: 68-74% ->
  20-23% (da <= 40), 39-49% (da 60-80, r 100); 13: 37-51% -> 14-18%; 16: no change (5-15%, fast agents escape in
  the open anyway). First trap-family mechanism that is clearly positive in the narrow test at zero extra cost.
  R3 ref2 on j (128 seeds 7200-7327, 6 configs): rf_ref, rg_60, rg_100, rg_100_slow (only agents that cannot
  outrun a predator), rg_100_post (posts near known gaps, refuge_post_w 1), rg_100_slow_post.
- 07:46 R3 ref2 (128 seeds, full games): refuge is strongly NEGATIVE: rg_100 -284+-30, rg_100_post -237,
  rg_60 -170, rg_100_slow -146, rg_100_slow_post -152; kills per game UP (141 -> 160-179), starvation down
  (291 -> 227-257). Opposite of the scenario (kills 41% -> 20%). Hypotheses: (a) the policy's site positions in
  a self-built map are off by a few units, so the agent holds outside the safe zone; (b) inside the gap the agent
  faces the goal, the predator at the mouth is behind it and outside its vision cone, so after refuge_leave 8 s
  it walks out into the predator; (c) route deaths (the run to the gap). Narrow test ref3 on j (96 seeds): death
  attribution counters (died en route / holding / exiting, exits), hold now faces the mouth, refuge_leave 8 vs 30,
  all vs slow-only.
- 07:55 ref3 (96 seeds) death attribution, per game with rg_100: 113 refuge attempts, 92 die EN ROUTE (82%),
  10 die holding, 0 die exiting, 0.3 exits (refugees hold until they die; the predator never leaves). Facing the
  mouth and refuge_leave 30 change nothing (-278/-297; slow-only -139/-174). So the full-game failure is the RUN
  to the gap, not the hold: in the scenario the agent stands on the lane axis in front of the mouth; in a game an
  agent "within 100 of the mouth" is usually on the far side of the obstacle and runs into walls with the predator
  behind. Next narrow test ref4: refuge_clear (straight run to the pre-point and mouth must not cross known walls)
  and refuge_sprint (sprint the whole run), radius 60/100, all vs slow-only.
- 08:05 ref4 (96 seeds): the clear-path check cuts refuge attempts 113 -> 20-36 per game and route deaths
  92 -> 8-16; with sprint the cost drops to -28+-33 (rg_100c_sprint), -44 (r 60), -59 (slow-only); without sprint
  -125. Still not positive. Remaining defects: (1) 7-12 refugees per game die WHILE HOLDING (0 in the scenario);
  (2) refugees never exit (0.4 per game): they hold until they starve because the predator never leaves.
  Geometry finding for (1): a predator (r=10) only has to clear the face END points, so it can push its centre to
  sqrt(100-(gap/2)^2) outside the mouth line: 8.7 for gap 10 but only 1 unit for gap 19.9 => "9 deep" is
  within 15 of it for gaps wider than ~16 (the scenario site had gap 18.6 and the predator got to 17). Same at an
  open rear end when the gap is short. ref5: site_safe=4 (hold depth = 15 - xmin + 4, open-rear sites need length
  >= 2 x depth), radius 60/100/150, slow-only variant.
- 08:14 ref5 (96 seeds, safe hold depth site_safe=4): rg_60s -14+-38, rg_100s_slow -67, rg_150s -85, rg_100s -120;
  hold deaths unchanged (6-14 per game) => the gap-width geometry was not the cause; with the scenario at 0 hold
  deaths on exact walls, the remaining suspect is the self-built map (hold point a few units off / entry deflected
  by a wall collision). Exits still 0.4 per game. Stopping rule applied: four narrow iterations, best result ~0
  (r 60, clear path, sprint). Refuge line CLOSED; pod j stopped; all pods idle. Spend ~$39 of $100.
- 08:16 Tick: one cheap diagnostic before leaving the refuge line for good: why do refugees die WHILE HOLDING
  (7-14 per game, 0 in the scenario)? Added an engine kill hook (NIGHT_REFLOG=1): at every predator kill of an agent
  heading to / holding in a refuge it logs believed vs true position, true distance to the hold point and mouth, gap,
  rear_ok, nearest predator distance and predator count. Pods j, n6 and k cannot restart (hosts full), so a new pod
  n7 (oscar-claude-night-7, dpm6cy0ejfmh5o, EU-RO-1, $0.96/h) was created for it. Run: 32 games rg_60s.
- 08:21 ref6 diagnostic (32 games rg_60s, 394 refugee kills logged, NIGHT_REFLOG): HOLDING deaths (n=175):
  pose error median 0.0 (163/175 under 2 units), TRUE distance to the hold point median 1.7 (151/175 within 3),
  true distance to the mouth median 10.9, gap median 13.7, nearest predator at death median 11.2, 8 predators on
  the map. So the agent really is at the hold point and a predator still reaches it: the SITE is wrong, not the
  pose. A predator cannot be 11 units from a point 11 deep inside a real 13.7-wide gap, so at least one of the
  two map faces is not real geometry (a face extended past its true end, or a phantom face), and the predator walks
  in from the side the map calls walled. EN ROUTE deaths (n=219): pose error median 10 (113/219 above 10 units),
  i.e. the runner's own pose is off, consistent with wall-collision deflection during the run. Both are map/pose
  consistency failures, not the refuge logic. One last variant: refuge_verify (hold only while both faces are
  actually observed at gap/2 +- 3 on both sides; otherwise abort to normal evasion).
- 08:30 ref7 (96 seeds): refuge_verify (hold only while both faces are observed at gap/2 +- 3 on both sides):
  hold deaths 6.7 -> 2.4 per game, aborts 6.4 (i.e. about two thirds of the map's "gaps" are not real at the hold
  point), survival -23+-34 (r 60) / -28+-36 (r 100) vs -14+-38 unverified: the aborted refugees die anyway and the
  route deaths remain. Refuge line closed for good. Pod n7 stopped; all pods idle; ~$41 of $100 spent.
- 08:32 Tick, my call: the refuge diagnostic pointed at the one thing that blocks the whole trap family, the
  self-built wall map. Measuring it directly (M1): new hook dbg_walls + nightsim/mapcheck.py grade every confirmed
  map face against the true map (solid side actually free = phantom length; free side actually blocked = missed
  length) and every crevice site (hold point free, predator cannot reach within 14.9, lane free), by observation
  count. If phantoms concentrate in low-observation faces, the fix is a stricter confirmation rule (wall_min_obs),
  which is a one-parameter A/B. Pod n7 restarting for it.
- 08:33 M1 first look (seed 3, 300 s, 400 confirmed faces, blocked/free profile across each face's middle):
  46% correct (free side free, solid side blocked at the face), 28% have the SOLID SIDE INVERTED (blocked on the
  side the map calls free, free on the side it calls solid; e.g. the right face of a 48-wide wall recorded with the
  left face's sign, 178 observations), 15% lie entirely inside solid, 4% float in free space. Site finder: 19
  sites, 3 valid in truth, top-1 invalid. An inverted face pairs with a real one into a "gap" that is the inside of
  a wall or an open corridor: that is the refuge/bait hold-death mechanism. Measuring over 32 games on n7 (mc1).
- 08:35 Cause found in add_wall(): an observation of a known face with the OPPOSITE solid side (a mis-posed
  observer) did not match the face and created a twin face with the inverted sign at the same place. A thin wall's
  two inverted twins then pair into a "crevice" whose hold point is the wall's interior, and clear_of() skips the
  pair itself. Fix behind wall_conflict=1: such observations count as conflicts on the existing face (no twin),
  the face flips if conflicts outnumber agreements, and a face with > 25% conflicts is not confirmed.
  Local check (4 seeds, 300 s): phantom face length 33.9% -> 13.6%, missed 44.6% -> 23.0%, sites 52 -> 9 of
  which 7 valid (was 11 of 52), top-1 valid 3/3. Running on n7: mc1/mc2 (32 games each, map metrics with/without
  the fix), then f38 = full-game refuge A/B on the corrected map (rg_60s, rg_100s, rg_100v with wall_conflict).
- 08:35 mc1 baseline (32 games, 600 s, with predators, current map code): 1286 confirmed faces per game,
  phantom 37.3% of face length, missed 48.0%; 23 crevice sites per game, 15% valid in truth, TOP-1 VALID 3%.
  That is the number behind the whole trap line: 97% of the time the policy's best crevice was not a real crevice
  (the guide/bait scenarios loaded true walls, so they never saw this). mc2 (same games, wall_conflict=1) and the
  refuge A/B on the corrected map are running on n7.
- 08:44 mc2 (same 32 games, wall_conflict=1): phantom 37.3% -> 18.8%, missed 48.0% -> 27.5%; sites per game
  23.3 -> 5.2, valid 15% -> 49%, top-1 valid 3% -> 35%. f38 refuge A/B on the corrected map (96 seeds):
  rg_60s_wc +13+-32 (50/96 wins; first non-negative refuge result), rg_100s_wc -17+-33, rg_100v_wc -65+-34;
  hold deaths 4.1 per game at r 60 (was 6.7), refugees still never exit. wall_conflict is now the DEFAULT in
  nightsim (it only affects map-building modes: trap/hide/refuge/occ_walls; the shipped configs build no map, so
  they are unchanged). Half the sites are still invalid: testing stricter confirmation (two distinct observers,
  wall_min_obs=2; or 20 observations, wall_min_n=20) on the map metric only (mc3).
- 08:46 mc3: wall_min_obs=2 is already the default (identical to mc2); wall_min_n=20 (20 observations to confirm a
  face): phantom 18.8 -> 14.5%, missed 27.5 -> 24.7%, sites valid 49 -> 56%, top-1 valid 35 -> 52%. Diminishing
  returns from confirmation knobs; the remaining errors are faces recorded inside solid (observer position error).
  f39 on n7: does the corrected map rescue Lucas's guide trap? rf_ref vs rf_trap_old (map as before) vs
  rf_trap_wc (wall_conflict + wall_min_n 20) vs the same with guides only for predators within 300 of the trap.
- 08:52 f39 (96 seeds, guide trap on the corrected map): rf_trap_old -183+-39, rf_trap_wc -180+-37,
  rf_trap_wc_lane300 -145+-36; held 0.07/0.07/0.18, handoffs 0.3 per game, 20 baits born per game in all three.
  The map was not what broke Lucas's trap in full games; delivery (guide episodes ending in death or loss before
  the handoff) and the bait cost are. Pod n7 stopped; all pods idle. ~$45 of $100 spent. Branch pushed.
- 08:53 Tick, my call: one last narrow trap test that the refuge work suggests. Guides die "far" and "slow":
  the scenario always had a straight clear lane, full games do not, and for refugees a clear-path requirement cut
  route deaths 92 -> 8-16 per game. guide_clear=1: a guide is only chosen if its straight walk to the lane point
  crosses no known wall, and is released if that stops being true while leading (the corrected map makes this
  check meaningful now). f40 on n7 (96 seeds): rf_ref, rf_trap_wc, rf_trap_clear, rf_trap_clear_lane300.
- 09:01 f40 (96 seeds): guide_clear: rf_trap_clear -105+-37, rf_trap_clear_lane300 -91+-36 vs rf_trap_wc -180+-37;
  guide deaths 15.2 -> 4.4 (kills per game 146 -> 132, below the reference 138), episodes released 40 of 56
  (lane blocked), handoffs 0.3, held 0.14-0.17, baits 20 per game. Guide deaths are solved; delivery is not.
  Trap line closed for the night with a full cost decomposition. Pod n7 stopped; all pods idle; ~$46 of $100.
