# Guard against capture from stale predator sightings

The new sprint benchmark exposed guide 36 on PC seed 730951 at 172.9 seconds.
It had energy 481.2, sprint speed 18.32 and grassland terrain. The old search
said its move was safe. Native replay with all original actions reproduces
the death, with **zero position error** before the fatal tick. An alternative
ordinary-observation escape action survives that tick.

The DTO sighting was 31.78 units away, but the predator had already moved and
was actually 17.29 units away. The old forecast repeated the predator's previous
velocity; during this turn that projected it to the wrong place. Public chase
rules applied to the observed heading and updated guide position reproduce
the actual missing move in this case without reading hidden engine state.

Replacing every nominal prediction with this projection saved the isolated
tick but regressed a full dev-seed game: **433.60 score / 413.9 seconds**,
one delivery arrival from 28 assignments, and one premature sprint-available
capture (replacement bait 22). This broad version is not the chosen strategy.
Its exact source is recoverable by applying `guide-public-lag-experiment.patch`
to commit `1fcdb57`; source hash and results are in the adjacent manifest and
summary. The patch hash was checked against the manifest.

The narrower implementation preserves nominal guiding decisions, then checks
the proposed next move for capture after projecting the missing predator move
with public physics. If any sampled speed/terrain/rest/pivot outcome captures
the guide, it reruns the three-tick search using projected predator states.
It does not impose an extra fixed predator clearance. The guarded version
also saves guide 36 in the exact native counterfactual, with the guard triggered
and zero pre-change replay position error. This proves only that tick; unknown
targets, map errors, terrain borders and later actions still limit safety.

Artifacts: `guide-36-native-escape-counterfactual.json`,
`guide-public-lag-counterfactual.json`, `guide-lag-guard-counterfactual.json`.
`scripts/replay_guide_forecast.py` reconstructs guide memory from ordinary DTOs,
restores the previous forecast sample, then changes only the fatal search.
The original failure is viewable at `http://localhost:9082/?time=171.9`.
Every frame through 174 seconds is retained in
`logs/entrapment-iteration/sprint-failure-730951-frames`.

## Replacement avoidance exemption

The broad pilot exposed another avoidable death: replacement 22 at 156.2 s.
It had once visited the rear waypoint, but its route subsequently went around
to the exposed front. The permanent `entered_rear` latch still suppressed
avoidance of nearby trap occupants. Restoring normal avoidance saves the exact
fatal native tick; see `replacement-22-native-escape-counterfactual.json`.

`core.py` now clears that latch and replans toward the rear if the estimated
position crosses out past the trap's front plane. The hearing exception remains
available when entering from behind. This fixes the stale exemption, not the
underlying reason that the route took a detour around the crevice.

Separate full dev-seed replays are running with the narrow lag guard alone
and with the rear-route reset. Both keep every frame. Do not report their
interim scores as completed benchmarks. The original PC forest/swamp failure
is also being recorded on its original machine, to inspect the repeated
border crossings with exact state reproduction. No paid compute was used.

## Completed guard-only pilot

Seed 1883894846: **805.35 score / 752.0 seconds**, normal extinction,
295.60 seconds wall time, all frames retained. **Zero premature sprint-
available captures**, two delivery arrivals from 25 assignments, and
82.8 estimated bait-gap seconds. The prior 44-unit handoff with isolated
tracking scored 842.83 / 791.6 seconds with one arrival from 17 assignments
and 52.5 estimated gap seconds. This is a specific safety repair with native
counterfactual evidence, not an established full-game performance improvement.
The separate rear-route reset comparison remains in progress.

The rear-route reset replay completed with the **same 805.35 score, 752.0-second
lifetime, 82.8-second estimated bait gap and zero premature sprint captures**
as the narrow guard alone. The reset was not exercised on this trajectory.
It is a repair of the stale exemption exposed in the broad pilot, not a
measured gain on the new default trajectory. Oscar's branch was rechecked at
`4542793`; recent additions concern his separate C++ guiding/holding experiments,
not new best survival parameters.
