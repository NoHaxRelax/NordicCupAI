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
(b) is structural and would need either relays between agents or a rest-window reversal
manoeuvre (walk round the sleeping predator and re-attract it from the other side).

## Results so far (600-second games, seeds 1–8, two repeats, oracle world)

| Batch | Mode | Runs | Score mean ± sd | Extinct | Predator deaths | Held fraction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| refuge-v1 (pushes around any staffed station, refuge on every flee) | society | 16 | ≈620 | 0 | ≈10 | – |
| refuge-v1 | society + traps | 16 | 602 ± 35 | 2 | 10.4 | 2.8 % |
| refuge-v2 (pushes only around held predators, refuge only in danger) | society | 16 | 615.1 ± 12.4 | 0 | 9.9 | – |
| refuge-v2 | society + traps | 16 | 615.8 ± 10.7 | 0 | 9.0 | 2.4 % |

refuge-v1 harmed the colony through constant pushes near gaps that sit next to its food and
through refugees idling inside empty traps; refuge-v2 is neutral at 600 s. refuge-v2 still had a
bug that made a fresh bait leave as "idle" immediately (fixed afterwards), so its held fraction
understates the mechanism. The 2000-second batch `long-v3` (seeds 1–6, two repeats) in
`results/trapper/batches/` is the first run with the fix and a horizon where predator pressure
matters; read its summary before drawing conclusions.

Same-seed runs are not reproducible on this engine (object sets iterate in memory order), so
compare means over repeats, never single runs.

**Estimator (observation-only world).** `check_estimator.py` on seeds 1–2 over 200 s: absolute
localization at t = 0.1 s (an agent sees an arena boundary), median and 95th-percentile position
error 0.0, heading error 0.0, 70–85 rectangles recovered of 80 (a handful are spurious).
A 600-second game on seed 2 with `--world estimator` found three gap sites from the estimated map,
tracked predators, and ran refuge flights end to end, so the manager works without the oracle.

## Honest assessment and next steps

1. The 600-second window is where the society already copes (4–8 predators). Trapping should
   pay off at 900–3000 s with 8–26 predators. Run `batch.py --seconds 2000` on seeds 1–8 with two
   repeats (about an hour with four workers) before judging the strategy.
2. Gap crowds are the cheapest containment: one bait can hold many predators and needs no guard.
   Bait rotation needs a free mouth; when both mouths get predators, wait for a rest window
   (implemented in `manager._gap_approach`) or accept that the bait stays until it dies.
3. Walls need a sacrifice (an old or senescent agent is free) and a guard on the back side; the
   controllers exist (`lure.py`, `manager` guard role) but are switched off until deliveries are
   reliable. Turn them on with `--params '{"explicit_deliveries": true}'`.
4. The estimator is the path to the real server: the same manager runs on `EstimatedWorld`
   instead of `OracleWorld`. Check `scripts/trapper/check_estimator.py` output (localization
   time, position error, rectangles recovered) and then switch `TrapperPolicy(world_source=...)`.
5. The society itself loses ~40 agents per 600 s to starvation (newborns); that is Oscar's
   open problem in the society line and dominates the score more than predators do at this stage.
