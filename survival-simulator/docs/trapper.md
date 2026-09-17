# Trapper: predator trapping on top of the society

Branch `survival-simulator/trapper`, package `models/trapper/`, scripts in `scripts/trapper/`.
Written 17 September 2026 from Oscar's and Lucas's earlier trap research (see the
`oscar-trap-research` folder on the `lucas-trap-slopsesh1` branch for the evidence behind the
geometry numbers). Everything below was measured locally with the unmodified simulator.

## What is in the package

| Module | Purpose | Status |
| --- | --- | --- |
| `world.py` | The `WorldState` views (agents, predators, rectangles, food) that every layer reads. Fields the estimator cannot know are `None`. | done |
| `oracle.py` | `WorldState` from the true engine state, for development only. | done |
| `estimator.py` | `WorldState` from ordinary observations: exact odometry, heading from any wall edge, absolute position from arena boundary edges, frame merging when agents see each other, rectangles from two perpendicular edges, predator tracks (heading from `rel_dir`) advanced one step with the predator model. | validated: exact poses on seeds 1–2 (see results) |
| `predator_model.py` | Exact clone of the engine's predator: what it observes, its decision (charge / pivot / edge / wander) and its movement with collision deflection. | validated tick for tick on 8,500 predator steps (`scripts/trapper/validate_predator_model.py`) |
| `sites.py` | Wall sites (30–35.5 thick, ≥70 long, clear 150-unit approach) and gap sites (11–19 wide, ≥55 overlap, bait 5 inside the mouth). | done; on 30 maps: wall site on 23, gap site on 24, either on 28 |
| `paths.py` | Grid A* with clearance and line-of-sight shortcutting. | done |
| `lure.py` | Guide-led delivery: attract → lead (facing the predator with an alternating gaze offset) → corridor → sacrifice at the front / enter a gap. | works in arranged fixtures for favourable geometry; not reliable in real games (see below) |
| `refuge.py` | A chased agent runs into the nearest reachable gap and becomes the bait. | works in real games; net effect on score not yet positive |
| `manager.py` | Stations, baits, successors, guards, deliveries, zone avoidance for the society. | done for gaps; wall staffing only with explicit deliveries |
| `policy.py` | `TrapperPolicy`: Oscar's society (`society_base.py`, verbatim copy) plus manager overrides, written back into the society's odometry. | done |

Scripts: `validate_predator_model.py`, `survey_sites.py`, `check_lead_dynamics.py`,
`test_delivery.py` (arranged fixtures), `run_game.py` (one game, metrics, optional Survival Lab
replay), `batch.py` (seeds × repeats in parallel, paired table), `trace_game.py` (every delivery
tick by tick), `check_estimator.py`.

Run from `survival-simulator/` with the uv environment at the repository root:

```sh
../.venv/bin/python scripts/trapper/validate_predator_model.py --seeds 1 2 3
../.venv/bin/python scripts/trapper/test_delivery.py --kind gap --cases 6 --record
../.venv/bin/python scripts/trapper/run_game.py --seeds 2 --seconds 600 --mode both --record
../.venv/bin/python scripts/trapper/batch.py --seeds 1 8 --repeats 2 --seconds 600 --workers 4 --label mytest
```

Replays land in Oscar's Survival Lab results folder (`survival/results/trapper/replays/`) when
his checkout exists; set `TRAPPER_RECORDER_DIR` / `TRAPPER_REPLAY_DIR` otherwise.

## Mechanics that drive every design decision

* A predator chases the nearest agent it observes (hearing 60 through walls, 60° cone to 250 with
  line of sight). Facing it from beyond 90 makes it *pivot* (45° off-line at sprint, i.e. it
  closes at ~10.6/tick, or 7.8/tick once it is down to walking); exact facing (`rel_dir == 0`)
  makes it charge straight, so a guide alternates a ±0.12 rad gaze offset.
* A walking guide (10/tick) therefore holds a sprinting predator at bay only while retreating
  almost directly away; any sideways component loses ground. This is why leading a predator to a
  trap that is not roughly "behind us" fails: the pair can only rotate slowly, and only while the
  predator is in its walking phase.
* Walking costs 5 energy/s, so a lead of more than ~20 s exhausts a 150-energy guide.
* Predators cycle sprint (~2.4 s from a fresh wake) → walk (7.2 s) → rest (3.4 s, no kills).
* Contact kills below 15 units. A gap of 11–19 admits an agent and not a predator; a bait 5
  inside the mouth is 15.0 from a predator pressed into the mouth and survives.

## What works

**Arranged deliveries (oracle world).** With one guide, one predator starting 260–300 away
and a prepared holder:

| Fixture | Cases | Delivered and held to the end |
| --- | ---: | ---: |
| Wall 30×100, predator ahead within ±45° of the approach axis (incl. one resting) | 4 | 4, in 5–9 s |
| Wall, predator to the side or behind the guide | 4 | 0 (the retreat cone cannot reach the corridor; guide starved or lost the predator) |
| Gap 15×100, guide enters and becomes the bait | 3 | 3, in 3–6 s, guide alive with 60–77 energy |

**Real games (oracle world, `refuge` on, explicit guide deliveries off).** Chased agents run
into the nearest gap when it is reachable before the predator closes; the first one inside is the
bait. Predators that followed them pile up at the mouth. Per-game holds of 14–16 % of
predator-time were reached on some seeds with no guide losses.

**Guide-led deliveries in real games** did not work: over six 600-second games, 225 attempts,
none held, 26 guides lost. The failure classes were (a) chasing wandering predators that cannot be
intercepted (131), (b) the trap lying outside the retreat cone so the pair drifts away from it, and
(c) obstacles cutting the predator's line of sight mid-lead. Fixes for (a) and (c) are in the code;
(b) is structural. A rest-window reversal (walk round the sleeping predator and wait 50–59 behind
it) was checked with the exact model and does not work: on waking it moves 15 units along its old
heading before its capped turn can bring it round, the guide drops out of hearing after one tick
and it wanders off. The remaining option is a **relay**: a second agent with sprint energy waits
about 59 units beside the predator's path outside its cone; the predator hears it, charges, the
relay sprints ~12 ticks in the wanted direction (about 60 energy), then faces it and leads. Each
relay can turn the pair by any angle; the manager already has the roles to host this.

## Gap-first protocol (18 September, Oscar's refocus)

Narrow gaps are the primary trap; walls stay in the code as a backup (`site_kinds`). The pieces:

- **Sites** (`sites.py`): width 10.5–19.5 (the engine's strict tests let an agent of radius 5 through
  above 10 and stop a predator of radius 10 below 20), passage length >= 30, bait depth chosen so
  the bait stays > 16.5 from any point the predator can reach from either mouth
  (`gap_depth`: for width w the predator's center stays sqrt(100 - (w/2)^2) outside a mouth),
  straight approach >= 100 clear. Each site carries a score (short passages, closed far mouths,
  extreme widths and short approaches are penalised) and staging/exit points outside the far mouth,
  off the axis on opposite sides. 59 of 60 generated maps have at least one site (mean 6.1).
- **Staffing** (`manager.py`): the best open station is kept staffed from `prestaff_time` on.
  Baits are chosen by remaining life (`_life_s`: energy over 1 + 0.1 x age once senescent):
  senescent agents first because the colony loses them anyway, then the oldest; a senescent
  bait needs 30 s of life on arrival, a healthy one 60 s. Baits walk in through the far mouth.
- **Replacement from behind**: a successor is called when the bait's life would not cover the
  replacement's walk plus 25 s; it stages outside the far mouth. The old bait walks out to the
  exit point when every held predator rests (or its life is under 8 s), then the successor walks
  in. A senescent bait never leaves to eat; it serves until it dies.
- **Delivery to a staffed mouth** (`lure.py`): LEAD as before, then CORRIDOR facing the predator
  (it pivots and closes 0.6 per tick) to the flyby point 14 out from the mouth; when the predator
  is within 45 the guide sprints along the obstacle face ("flyby") and the predator, now nearer
  the bait than the guide, takes the bait within a few ticks. If the guide dies at the mouth the
  predator takes the bait anyway. An unstaffed mouth is entered by the guide, which becomes the
  bait. Chased agents running for refuge do the same flyby when the holder is occupied.
- **Guide safety**: predator speed predicted from its energy (15 above 40, else 11; it wakes in
  the step its energy passes 100); inside 90 the guide sprints directly away with an
  obstacle-aware heading; `steer` tests clearance from the next position and falls back to the
  clear heading with the largest away component anywhere on the circle (running along a wall
  instead of into it); a second loose predator within 150 aborts the delivery; a transfer to an
  agent without sprint energy ends it.
- **Planner**: biome movement modifiers are path costs (river 3.3x, swamp 2x, desert 1.25x),
  cached per grid cell; grids are keyed on geometry; failed searches are cached for 2 s.

## Gap-first results (18 September, 600-second games, oracle world unless noted)

A/B over seeds 1–16, two repeats each (`results/trapper/batches/ab-*.json`, compare with
`scripts/trapper/compare_labels.py results/trapper/remote ab-society ab-v5 ab-v6 ab-noreserve`):

| Variant | Score | vs society (mean / median / wins) | Held | Delivered | Guide deaths |
| --- | --- | --- | --- | --- | --- |
| society only | 612.1 | – | – | – | – |
| wide-turn deliveries, no reserve slot (now the default) | 617.4 | +5.3 / −0.6 / 7 of 16 | 5.3 % | 16 of 198 (14 guides alive) | 21 |
| narrow turns, reserve slot | 608.4 | −3.8 / −3.2 / 4 of 16 | 5.7 % | 4 of 41 | 8 |
| narrow turns, no reserve | 615.7 | +3.5 / −4.6 / 5 of 16 | 5.7 % | 7 of 35 | 4 |
| **final-v7**: wide turns, no reserve, guides give up without sprint reserve (current defaults) | **619.6** | **+7.5 / +0.8 / 9 of 16** | 8.7 % | 19 of 189 (16 guides alive) | 4 |

The means are moved by single seeds where one colony went extinct (seed 5: +130); the medians
say "neutral" for the A/B variants and slightly positive for the current defaults (no extinction in
32 games, 160 holds of ~20 s, 91 of them ended by the predator leaving the mouth). Fruit gathering (score plus penalty) is identical (627 vs 628); the trapper pays a
higher eaten-energy penalty (19.7 vs 15.2) because the agents that die in trap roles carry more
energy. Holds last about 30 s and end when the bait dies or walks out (45 of 89) or the predator
leaves (25 of 89); up to 3 predators were held at one mouth. Most holds come from refuge runs
(chased agents entering a passage: 112 entries in 32 games), not from guide-led deliveries.

Estimator world (observations only). Four early runs held 6–12 % of predator-time; refugee
deaths dropped from 6–8 per game to 0–1 once the refuge margin was widened to 45 for estimated
tracks. The 8-seed batch `final-v7est` then scored 620.1 against 612.3 for the society (median
+6, 5 of 8 wins) but held only 1.6 %: 125 of 193 deliveries were aborted for a "second predator"
that was a stale track (a predator last seen seconds ago, carried at its predicted position).
Aborts, staffing gates, wild-threat checks and mouth choices now use only predators observed
within the last 1.5 s (`WorldState.recent_predators`). The estimator also carried two or three
tracks for one predator (observers in different frames, re-detections after a gap); tracks are now
associated with their predicted position, expire after 3 s and are deduplicated within 30, and a
frame-merge crash (a frame merged twice in one tick) is fixed. Fresh predator estimates are
then within 8–15 (median) and 15–44 (p90) of the truth. Batch `final-v9est` (8 seeds x 2):
619.7 against 612.3 for the society (median +5.8, 7 of 8 wins), held 1.7 %; 181 guide-led
deliveries with 3 delivered (leads need the gap to ±5, which estimated tracks cannot give), holds
come from refuge runs. Without the oracle, refuge runs and flybys are the working mechanism;
`est-nodeliver` and `est-shortlead` test whether guide-led deliveries should be off or short on
the estimator.

Two defects found on the way that any override policy must respect: overridden agents must keep
the society's spawn decision (dropping it stopped all breeding by role agents and starved three
colonies), and the A* grid must be cached on geometry (a per-tick rebuild made games 30x slower).

## Results before the refocus (600-second games, seeds 1–8, two repeats, oracle world)

| Batch | Mode | Runs | Score mean ± sd | Extinct | Predator deaths | Held fraction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| refuge-v1 (pushes around any staffed station, refuge on every flee) | society | 16 | ≈620 | 0 | ≈10 | – |
| refuge-v1 | society + traps | 16 | 602 ± 35 | 2 | 10.4 | 2.8 % |
| refuge-v2 (pushes only around held predators, refuge only in danger) | society | 16 | 615.1 ± 12.4 | 0 | 9.9 | – |
| refuge-v2 | society + traps | 16 | 615.8 ± 10.7 | 0 | 9.0 | 2.4 % |

refuge-v1 harmed the colony through constant pushes near gaps that sit next to its food and
through refugees idling inside empty traps; refuge-v2 is neutral at 600 s (it still had a bug that
made a fresh bait leave as "idle" at once, fixed afterwards).

**2000-second games (`long-v3`, seeds 1–6, two repeats, idle bug fixed):**

| Mode | Runs | Score mean ± sd | Survival mean | Extinct before 2000 s | Predator deaths | Starvation deaths | Held fraction | Max held at one station |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Society | 12 | 1048 ± 219 | 1039 s | 12/12 | 23.0 | 69.0 | – | – |
| Society + refuge traps | 12 | 967 ± 245 | 967 s | 12/12 | 26.2 | 61.1 | 3.5 % | 0 |

Every colony, with or without traps, went extinct between roughly 700 and 1500 s. The refuge
version scores about 80 points lower on average (within one standard deviation, but consistent
with the 600 s neutrality: it does not help). Predators were held for only 3.5 % of predator-time
and never more than 0 at one station, so the mechanism is not engaging often enough to matter,
while its costs (agents inside traps not foraging, energy spent running) are real.

Same-seed runs are not reproducible on this engine (object sets iterate in memory order), so
compare means over repeats, never single runs.

**Estimator (observation-only world).** `check_estimator.py` on seeds 1 and 3 over 450 s:
absolute localization at t = 0.1 s (an agent sees an arena boundary), median and 95th-percentile
agent position error 0.0 (transient maxima 11–71 before an edge fix), heading error 0.0,
75–82 rectangles recovered of 80 (a few spurious), predator track error median 4–10 units and
95th percentile 15 (one predator step: the one-step prediction assumes 100 energy and flat
terrain). A 600-second game on seed 2 with `--world estimator` found three gap sites from the
estimated map, tracked predators and ran refuge flights end to end, so the manager works without
the oracle.

**Server.** `scripts/trapper/agent_server.py` serves the policy on the estimator at `/predict`
(handles the lowercase types and missing `sim_time` of the platform's test sample, and detects
new games). Smoke test: 400 ticks over HTTP at 7 ms per tick, worst 47 ms.

## Honest assessment and next steps

1. At 2000 s the trap version does not beat the society (see the table). Colonies die around
   1000 s from a mix of starvation and predation. To make traps pay, the number of predators held
   must rise from ~3 % to a large share of the population, which means (a) keeping a bait in a gap
   permanently near the colony's food so every arriving predator ends up at that mouth, and
   (b) feeding that bait by rotation through the far mouth. Measure `max_held_one_station` and
   `held_fraction` first; score only follows once holds are common.
2. Gap crowds are the cheapest containment: one bait can hold many predators and needs no guard.
   Bait rotation needs a free mouth; when both mouths get predators, wait for a rest window
   (implemented in `manager._gap_approach`) or accept that the bait stays until it dies.
3. Walls need a sacrifice (an old or senescent agent is free) and a guard on the back side; the
   controllers exist (`lure.py`, `manager` guard role) but are switched off until deliveries are
   reliable. Turn them on with `--params '{"explicit_deliveries": true}'`.
4. The estimator is the path to the real server and already carries the full policy:
   `agent_server.py` runs `TrapperPolicy(world='estimator')`. Validate on the platform's
   validation queue before trusting local results (Linux vs macOS set ordering differs).
5. The society itself loses ~40 agents per 600 s to starvation (newborns); that is Oscar's
   open problem in the society line and dominates the score more than predators do at this stage.
