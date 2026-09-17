# Trapper development plan (Oscar's order) and progress

Work the steps in this order. Wall trap and narrow-gap trap are developed in parallel because
luring and bait replenishment are shared. The society keeps foraging and breeding throughout.
"Validated" means measured on repeated runs, not a single run (the engine is not reproducible).

## Phase A: full knowledge of the game state (oracle world, development only)

| Step | Goal | Wall (backup) | Gap (primary) |
| --- | --- | --- | --- |
| 1 | Bait one predator into the trap while society continues; two agents cooperate | fixtures 3/4 (guide dies at the front by design); real games not measured since the refocus | fixtures 4/4 (guide enters, becomes bait, survives); flyby past a staffed mouth validated on the arranged passage; real games: 16 of 198 guide-led deliveries delivered (14 guides alive), refuge runs give most holds (112 entries per 32 games), held 5 % of predator-time |
| 2 | Replace the bait (agent 1) | implemented, not validated | replacement stages outside the far mouth and enters after the old bait walks out (33–38 of 61–77 replacements in place per 32 games); reserve slot behind a dying bait tried, no measurable gain |
| 3 | Guard (agent 2) protects the bait | implemented, not validated | not needed |
| 4 | Several predators in the same trap | not validated | up to 3 predators held at one mouth (`max_held_one_station`) |

## Phase B: no full knowledge

| Step | Goal | Status |
| --- | --- | --- |
| 5 | Fuse the agents' observations into a full game-state estimate | estimator exact (positions, headings), predators within one step; manager runs on it; agent server serves it |
| 6 | Redo steps 1–4 on the estimate | same protocol runs on the estimator (agent server serves it): 8 seeds x 2 give 619.7 vs 612.3 for the society (median +5.8), held 1.7 %; refuge runs work, guide-led leads do not (3 of 181 delivered) because estimated predator positions are 8–15 off (p90 up to 44) |

## Refocus (Oscar, 18 September ~00:30)

Narrow-gap traps first; wall traps stay in the code as a backup (`site_kinds=('wall','gap')`) to be
tested after Phase A and B are done on gaps. Reasons: no guard is needed, a guide that dies in front
of the mouth still hands the predator to the bait, and relaxed gap requirements give a site on almost
every map. Replacement baits enter from behind (the far mouth).

Gap-site rules now: passage width 10.5-19.5 (strict engine tests), passage length >= 30, bait depth
chosen so the bait is > 16.5 from any point the predator can reach from either mouth, straight
approach >= 100 clear for the predator. Survey over 60 generated maps: 59/60 have a gap site
(mean 6.1 per map); walls: 50/60.

Endgame on an arranged passage (`scripts/trapper/gap_endgame.py`, bait inside, guide brings one
predator down the axis): the "flyby" (guide sprints along the obstacle face when the predator is
within 45) hands the predator to the bait within 3 ticks and the guide survives; the sacrifice (guide
stands at 25 out) also hands it over, guide dead. Bait depth 5 or 9.4 both survive 40 s.

## Overnight work log (18 September 2026)

Entries are appended as milestones complete; each names the commit and the evidence file.

- **23:00** Wall deliveries in real games diagnosed with per-tick capture traces: guides died in the
  OPEN phase (society flee at low energy), after rest/wake transitions (predator speed assumed
  walking), in rivers (3 per tick) and to second predators while deferring. Fixes: energy-based
  speed prediction, own sprint flee, biome-aware planning, early abort on a second predator,
  transfers only to agents with sprint energy. walls-v3 -> v4: captures 38/59 -> 6/31, score 613.7 -> 617.2
  (`results/trapper/batches/v4-*.json`).
- **00:30** Oscar: gap traps first, walls as backup. Relaxed gap detector (59/60 maps), flyby
  endgame validated on the arranged passage (`scripts/trapper/gap_endgame.py`: predator held within
  3 ticks, guide alive), far-mouth staging and swaps, lifetime-based bait choice. Commit e4883be,
  pushed to `survival-simulator/oscar-trapper`.
- **01:10** gap-v1 (seeds 1-8, oracle): trapper 621.4 vs society ~615, held 5.6%, deliveries 36
  (1 delivered, 12 aborted for a second predator, 10 lost, 7 captured). Two speed bugs fixed
  (grid rebuilt every tick, failed A* every tick): 600 s games now ~60-140 s wall.
- **01:40** gap-v2 (seeds 1-8, both modes): trapper beats society on 6/8 seeds (+2 to +18) but
  seed 1 went extinct (-178): prestaffed baits and successors starved 15 agents. Staffing is now
  conditional on a loose predator within 420 of the mouth or the colony. Added the "hearing tap":
  a guide with sprint energy runs into hearing range (60) of a predator that lost it, instead of
  failing "cannot intercept" (the most common loss).
- **02:20** gap-v3: 3 of 12 colonies extinct. Cause found: overridden agents never spawned (the
  society's birth requests were dropped: 200-900 requests per game vs ~50 births). Births now pass
  through except within 35 of a passage. gap-v4: births normal, no trapper-only extinction.
- **02:50** Hold timelines showed a station fed a new dying (senescent) bait every 30 s without ever
  holding anything, and "holds" counted while the bait was still walking in. Speculative staffing
  off; a hold needs the bait at its slot; held stations get young well-fed replacements.
  gap-v5 (12 seeds): no extinctions, held 8.8% of predator-time (49 holds, 30 s each, up to 3
  predators at one mouth), score neutral (mean -1.0 vs society, 6/12 wins). Commit f968bd3.
- **03:20** Reserve slot restored for gaps: the replacement enters behind a dying front bait and
  moves up when it dies (no target loss); deliveries limited to turns <= 35-55 degrees and leads
  <= 650 to stop guide deaths (10 per 12 games). Batch gap-v6 running.
- **04:20** A/B over 16 seeds x 2 (see `docs/trapper.md`): wide-turn deliveries without the
  reserve slot are best (617.4 vs society 612.1; median -0.6) and are now the defaults; the
  narrow-turn + reserve variant was worst (608.4). Guides now give up a lead when their sprint
  reserve is gone. Commit aa8f497 pushed. Final 16x2 oracle batch and an 8-seed estimator batch
  running on the PC (`final-v7`, `final-v7est`).
- **04:50** final-v7 (16 seeds x 2, oracle): 619.6 vs society 612.1 (+7.5 mean, +0.8 median,
  9/16 wins), held 8.7 %, no extinction, 19 delivered (16 guides alive), guide deaths 4 (was 21).
  First configuration with a positive median. Commit dd7933e + docs.
- **05:10** Seeds 17-24 (one run each, Mac): trapper 616.2 vs society 618.6 (median -5.7): over
  24 seeds the score effect is neutral within noise. Resting predators at the mouth now count as
  held (holds were being cut at every wake). Estimator batch (8 seeds): score +6 median but holds
  1.6 %; cause: stale predator tracks aborted 125 of 193 deliveries. Fixed with a 1.5 s recency
  filter; `final-v8est` (8 seeds x 2) running.
- **06:10** Estimator: duplicate predator tracks found (2-3 tracks per predator) and fixed
  (prediction-gated association, 3 s expiry, dedupe within 30); a frame-merge KeyError fixed.
  `final-v9est`: 619.7 vs 612.3 (median +5.8, 7/8 wins), held 1.7 %, deliveries 3/181. Guide-led
  leads do not work on estimated tracks; refuge runs do. Testing deliveries off / short on the
  estimator (`est-nodeliver`, `est-shortlead`). Commits ed32785, 363bd0a.
