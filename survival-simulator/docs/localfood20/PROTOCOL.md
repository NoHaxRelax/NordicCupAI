# Local food: 20 new families

Parent: frozen local_food winner, docs/late250/pod-0/local_food-winner.json.
Branch: codex/localfood20. Earlier results and the replay server stay unchanged.

Same protocol as late250: 20 families × 32 trials × 200 shared checkpoints, each
250 seconds before baseline extinction. This time the opening policy is local_food.
Fresh training seeds 14001–14200. Complete Linux fork snapshots preserve engine RNG,
policy RNG, learned map, odometry and all agent memory. Every baseline continuation
must reproduce final score AND extinction time exactly before tuning starts.
Reject censored or too-short games rather than silently changing seed selection.
Objective: mean additional score until extinction or the 3000-second horizon;
not artificially truncated at the original death time. One seeded configuration,
five random warmup trials, 26 GP expected-improvement trials. Inputs scaled to [0,1],
GP objectives standardized, public scores kept in game units.

Every family retunes fit_vision, fit_hear, fit_energy and fit_speed along with its
specific parameters (6–7 dimensions). These existing weights rank breeding
candidates and help select heirs. A weight expresses preference among eligible
parents; it cannot manufacture a trait or force ineligible agents to reproduce.
32 evaluations in that dimensionality give a broad search, not convergence proof.

Families:
1. pulse_scan: x degrees every n ticks while moving or idle, unless a predator is visible.
2. idle_pulse: periodic scans only when stationary.
3. fast_travel: larger travel turn limit plus faster ordinary sweeps.
4. late_pulse: periodic scans after a tunable elapsed-time threshold.
5. sparse_pulse: periodic scans below a tunable living-population threshold.
6. late_retirement: stop feeding elderly agents only after a time threshold.
7. late_small_colony: lower population capacity only late and below a population threshold.
8. late_food_radius: different food reach late.
9. late_breed_reserve: change breeding reserve late.
10. small_colony_rescue: change reserve when both time and population triggers hold.
11. risk_one_tick: search escape actions when a one-tick approximation predicts danger.
12. risk_two_ticks: same with two ticks.
13. behind_predator: reward positions behind the observed predator, beyond hearing range.
14. behind_two_tick: combine behind preference, gaze and two-step risk estimates.
15. wall_risk: combine observed-wall escape with two-step risk estimates.
16. late_risk: activate two-step avoidance only after a time threshold.
17. fast_scan_risk: periodic scanning, larger travel turns and two-step avoidance.
18. expanded_local_food: extend the food/tree/post ranges beyond prior winning boundaries.
19. cluster_local_food: cluster food sites around the local_food baseline.
20. trait_selection: breeding eligibility/heir thresholds plus the four trait weights.

The short lookahead is an observation-only approximation, not a call into the real
engine: current own biome for movement, observed recent walls, all currently seen
predators, a conservative 15-unit direct chase per tick, capture distance 15 plus a
tunable margin. It does not know predator energy, unseen agents, or destination
biomes. Candidate actions include standing, walking and sprinting across16 headings.
It can reduce risk but offers no survival guarantee. Behind preference uses public
relative predator heading. Existing map learning is retained: orchard shared groups,
landmark odometry, boundary anchoring, food and exploration memory. This is NOT
Nikolaj's separate Python global-map module; predator sightings remain own-agent.

Phase changes use observable elapsed time and total living-agent count only. They
never use the baseline's future extinction time. Families with phase knobs keep
local_food early-game parameters, except breeding fitness weights which are retuned
throughout the game. Forecast/scan features have independent activation thresholds.
All optional features default off and must preserve the old baseline result.

After freezing ALL winners, evaluate each from game start on 1000 fresh paired maps
15001–16000. Include local_food, scheduled_breeding and original baseline: 23,000
full games. Keep per-map score, paired confidence intervals, policy/CPU/wall time
per tick and death counts (predation versus energy depletion, including aging).
This checks transfer to complete games; tail training cannot establish early-game
benefit or reliably identify the best onset threshold before all checkpoints.

Use ten existing Oscar CPU pods, 32 workers each, two families per pod. No new pod
provisioning, no spending cap currently, never stop these pods after use. Source
hashes and build manifests accompany raw results. Push protocol, code and results.
