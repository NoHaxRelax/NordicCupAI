# BO search space for lower-tail harvest reliability

Optimize **overall mean + mean of the lowest 10 scores**, both over **1,000
complete games**. Sort each candidate's own scores again; do not reuse a fixed list of
the baseline's worst seeds. Proposed bounds below are experiment ranges, not
validated optima. Keep extra drain actions at zero during this search.

```text
objective = mean(scores) + mean(sort(scores)[0:10])   # exactly 1000 games
```

This supersedes both the original worst-100 and subsequent worst-10-only
objectives. Both terms have weight 1; dividing the sum by 2 would give the same
ranking. This rewards both overall performance and catastrophic-run prevention;
it does not itself prevent overfitting to the search seeds. Log the worst-100 mean,
overall mean, minimum, and harvest failures as secondary diagnostics. Ten
tail observations make the objective noisier, so retain the common search seeds
and validate finalists on a fresh 1,000-seed cohort. Partial trials must never
be ranked or reported as complete 1,000-seed measurements.

Use the same fixed 1,000 world seeds for every search trial and a fixed policy RNG
independent of world seed. Confirm finalists on a separate, untouched 1,000-seed
set. The already-inspected 40001–41000 and 50001–51000 cohorts are development
data now; do not call either an untouched final holdout after tuning against it.
Use the C++ engine and compiled C++ controller. The current observation tracker
is still Python; a fully native simulation-policy loop is not yet implemented.

## Parameters to expose first

`New` means a named runtime setting is needed in C++; the current value is
hardcoded. `Existing` means a native base-policy configuration key already exists.
`Controller` means a HarvestPolicy constructor argument already exists.

| Parameter | Current | Suggested search | Status / purpose |
|---|---:|---|---|
| `early_population_target` | 8 | Derived, see constraints | New; enough survivors and harvest participants |
| `early_population_slack` | — | integer 0–6 | New BO coordinate above the minimum participant budget |
| `early_breed_until` | 0.5 s | 0.2–10 s, log | New; duration of the opening population boost |
| `early_parent_reserve` | 25 energy after birth | 15–80 | New; replace the hardcoded pre-birth threshold of 125 |
| `early_food_until` | 60 s | 20–180 s | New; end of immediate/early fruit collection |
| `early_ripen_wait` | 0 s | 0–5 s | New; separate opening food policy from later conservation |
| `ripen_wait` | 15 s | 8–20 s | Existing; later fruit ripening wait |
| `hungry_margin` | 5 | 2–20 | Existing; hunger override for waiting |
| `fruit_reach` | 200 | 120–450 | Existing; maximum fruit assignment distance |
| `low_pop_reserve` | 200 | 110–240 | Existing; birth threshold when population is low |
| `breed_reserve` | 200 | 130–300 | Existing; regular earlier-game birth threshold |
| `breed_reserve_late` | 200 | 150–400 | Existing; regular later-game birth threshold |
| `births_per_tick` | 3 | integer 1–4 | Existing; ordinary birth-rate limit |
| `cap_hard_min` | 2 | integer 4–10 | Existing; floor on the tree-informed population cap |
| `cap_min` | 4 | integer 6–14 | Existing; minimum nominal population cap |
| `cap_max` | 20 | integer 14–36 | Existing; maximum nominal population cap |
| `cap_mult` | 0.3 | 0.15–0.60 | Existing; population capacity relative to estimated resources |
| `tree_slots` | 1 | 0.5–2.5 | Existing; population capacity per known living tree |
| `min_free_agents` | 6 | integer 4–10 | Controller; survivors reserved before a sacrifice |
| `engage_range` | 45 | 25–60 | Controller; maximum harvest commitment distance |
| `max_harvest_ticks` | 2 | categorical 1, 2, 3 | Controller; bounded retry budget |

Start with food, population and harvest settings jointly: more births without
food, or a harvest reserve above the population cap, can defeat an otherwise good
setting. For a smaller first study, fix `early_breed_until=0.5`,
`early_ripen_wait=0`, `births_per_tick=3`, `cap_mult=0.3`, `tree_slots=1`, and
`max_harvest_ticks=2`; search the remaining high-priority coordinates first.

## Second-stage settings

These are already exposed unless marked new. Open this group after the first
stage has identified a viable population/food region; then jointly refine the
parameters with evidence of sensitivity rather than opening every dimension at
once.

| Parameter | Current | Suggested search | Reason |
|---|---:|---|---|
| `tree_half` | 600 s | 300–1,200 s, log | Resource-decline estimate driving population capacity |
| `cap_tree_slack` | 1 | 0–6 | Stops sparse tree observations shrinking capacity too aggressively |
| `reserve_t0` | 600 s | 100–900 s | Start of birth-reserve transition |
| `reserve_transition_duration` | 1,200 s | 200–1,800 s | New search coordinate; derive existing `reserve_t1` |
| `emergency_reserve` | 105 | 101–150 | Existing emergency births at very low population |
| `explore_min` | 60 | 25–100 | Energy needed for movement toward a watch location |
| `explore_energy` | 200 | 100–300 | Energy threshold for broader exploration |
| `explore_radius` | 450 | 200–700 | Exploration distance |
| `tree_reach` | 420 | 200–650 | Tree assignment radius |
| `watch_reach` | 500 | 250–750 | Watch-location search radius |
| `watch_patience` | 30 s | 5–90 s | Waiting versus moving on |
| `watch_refresh` | 60 s | 10–120 s | Reconsidering the watch location |
| `post_radius` | 30 | 18–45 | Position around a tree |
| `travel_turn` | 0.25 rad | 0.1–0.8 | Turning while travelling |
| `repost_every` | 10 s | 2–30 s, log | Reassignment frequency |
| `min_stay` | 15 s | 3–45 s | Commitment to a site |
| `switch_gain` | 100 | 10–300, log | Improvement required before switching sites |
| `site_min` | 5 | 0–15 | Minimum acceptable tree-site value |
| `dist_pen` | 0.1 | 0.02–0.4, log | Travel penalty in assignment |
| `heir_age` | 55 s | 35–75 s | Timing of replacement offspring |
| `heir_reserve` | 250 | 140–350 | Energy reserve for replacement offspring |
| `sweep_rate` | 0.03 rad | 0.01–0.12, log | Looking for food while stationary |
| `pred_r` | 70 | 45–100 | Evasion trigger distance |
| `pred_face_gap` | 10 | 0–50 | New coordinate; derive existing `pred_face_r = pred_r + gap` |
| `pred_sprint_r` | 40 | 20–60, bounded by `pred_r` | Sprint only near danger |
| `pred_dodge_r` | 80 | 40–110 | Dodge versus straight retreat |
| `pred_dodge_ang` | 1.4 rad | 0.8–1.8 | Dodge angle |
| `pred_turn_max` | 1 rad | 0.4–1.5 | Evasion turning limit |
| `harvest_probe_cooldown` | 1 s | 0.5–3 s | New; frequency of stationary perception probes |
| `harvest_probe_min_distance` | 50 | 50–65 | New; preserve the conservative lower distance bound |
| `harvest_probe_band_width` | 15 | 5–25 | New; derive upper distance bound |
| `harvest_facing_limit` | 0.6 rad | 0.25–0.8 | New; how directly the predator must approach |

`pred_dodge_r` has no effect above `pred_r` in the present flee branch. Express it
as a fraction of `pred_r` in [0.5, 1.0] to avoid searching an inactive interval.

## Optional categorical and conditional experiments

- `pred_face_threat`, `pred_evade_closest`: 0/1. Test whether evasion should face
  its threat and whether it should react only when that predator targets it.
- `feed_mode`: hungry/breed. This changes fruit-allocation priorities; keep it
  categorical and evaluate interactions with the birth thresholds.
- `heir_at_food`, `heir_needs_site`, `heir_select`, `old_eat_last`, `extra_old`:
  booleans. Only open heir-selection thresholds when `heir_select` is enabled.
- `idle_sweep`: boolean; `sweep_rate` remains relevant in other branches, so it is
  not globally inactive when this flag is off.
- `cluster_radius` and `spread_weight`: optional food-group distribution study.
  Start with clustering off; if enabled, search radius 50–200 and weight 0–2.
- `fit_hear`, `fit_energy`, `fit_speed`: relative offspring/agent selection
  weights. Fix `fit_vision=1` rather than varying all weight scales together.
- `wall_mode`: keep off initially. If explicitly enabled in a separate study,
  activate `wall_engage_r`, `wall_target_gap`, `wall_epsilon` and
  `steer_max_ticks`. Their mechanism and constraints differ from direct harvests.
- `dump_after_t`, `dump_food_site`, `dump_mult`, `nursery_bonus`: optional
  late-life/food-surplus birth experiments after the early tail is controlled.
- `lone_reach_mult`, `old_reach`, `heir_slack`, `select_min_young`: secondary
  allocation/replacement refinements.

Leave culling and unconditional late-game immobility disabled for the initial
reliability search. Represent optional infinite thresholds with explicit enable
flags and finite conditional ranges, not numeric infinity in the BO vector.

## Required configuration changes for a valid BO study

1. Route base-policy parameters to the native policy configuration, and harvest
   parameters to the compiled controller. The current `harvest_benchmark.py`
   calls `core.policy_init(0, {})`; its CLI does not forward the base-policy search
   vector. The controller separately exposes only its constructor parameters.
2. Expose the hardcoded opening population/food/probe constants in C++. Split the
   current bundled `harvest_ready` switch into independent opening births,
   opening fruit collection, and perception-probe switches for clean ablations.
3. Use one canonical, typed configuration. Reject unknown keys, invalid enum
   values, nonintegral integer settings, and infeasible combinations. Print and
   hash the fully resolved effective configuration for every trial. The current
   native parser can silently leave unused keys ineffective.
4. Remove duplicate search dimensions. `cap_mult` and `n0` appear as their
   product in the population estimate: fix `n0=80` and search `cap_mult`, or
   replace the pair with `initial_nominal_capacity`. Keep the derived cap and
   clipping parameters visible in the trial record.
5. Enforce `cap_hard_min <= cap_min <= cap_max`,
   `explore_min <= explore_energy`, and `reserve_t1 > reserve_t0`.
   Derive `reserve_t1 = reserve_t0 + reserve_transition_duration`.
6. Derive `early_population_target = min_free_agents + 2 +
   (max_harvest_ticks - 1) + early_population_slack`. This covers the farm, first
   predecessor, and possible retry predecessors. Still recheck the actual
   living-agent floor each tick; newborns never count before they exist.
7. Reuse that target in the opening boost and probe population gate. If a future
   opening cap is introduced, ensure it cannot undercut this target. Keep later
   capacity resource-sensitive; do not force the early population all game.
8. Keep reserve schedules ordered and bounded, and apply energy thresholds after
   movement/turn costs. Honor birth-rate limits consistently in the opening and
   ordinary birth paths; currently the opening boost is a separate code path.
9. Build once per source revision, configure per trial, and assert the compiled
   controller is loaded (`IMPLEMENTATION == 'cpp'`). Record C++ source hashes,
   binary hashes, compiler/build options, dependency versions and seed lists.

## Fixed correctness rules, not BO dimensions

- Minimum drain formula, predator recovery/wake constants, and horizon (3,000 s).
- `extra_drain_actions=0`; otherwise score can rise simply by increasing payload,
  obscuring reliability. Increase it only after selecting a reliable policy.
- The actual action/response budget, with atomic rejection before sacrifice.
- Immediate living-predecessor selection and the mutable-list skip-chain check.
- Confirmed-sleep requirements, observation freshness, identity association,
  stationary confirmation, and movement revocation of sleep certificates.
- Known walking/sprinting mechanics and their numerical matching tolerances.
- Score-confirmation accounting for simultaneous meals and the minimum cooldown
  needed to age out stale pre-harvest observations.

Treat observed early wakes, false sleeper filtering, or invalid sacrifice batches
as correctness failures requiring a code fix, not outcomes the optimizer may buy
with higher score. Record failed deliveries, zero-harvest games and colony losses
separately; choose any delivery-reliability constraint before looking at results.

## Trial record and final selection

Every scored trial needs exactly 1,000 distinct, completed seeds. Extinction is a
valid completed game; a crashed or timed-out worker is not a missing row to drop.
Use a declared failure score or mark the trial invalid and rerun it. Report the
objective together with minimum score, mean score, zero-harvest rate, harvest
delivery success, early wakes, false ignores, failed farms and runtime.

For the population/food hypotheses, also record population and collected fruit
energy at 10/30/60/120 seconds, time to first harvest, and starvation versus
predation deaths. These show whether a gain comes from actually feeding agents
or merely creating more short-lived offspring. Add diagnostic plumbing in C++;
do not feed simulator truth back into policy decisions.

Compare finalists using paired per-seed results and uncertainty on the worst-100
mean. A single minimum is a useful warning but is too narrow to replace the
requested objective. Freeze code, configuration and payload size before the
untouched final 1,000-seed validation.
