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

Separate full dev-seed replays completed with the narrow lag guard alone
and with the rear-route reset; results follow below. Both keep every frame.
The original PC forest/swamp failure was also reproduced on its original
machine. No paid compute was used.

## Completed guard-only pilot

Seed 1883894846: **805.35 score / 752.0 seconds**, normal extinction,
295.60 seconds wall time, all frames retained. **Zero premature sprint-
available captures**, two delivery arrivals from 25 assignments, and
82.8 estimated bait-gap seconds. The prior 44-unit handoff with isolated
tracking scored 842.83 / 791.6 seconds with one arrival from 17 assignments
and 52.5 estimated gap seconds. This is a specific safety repair with native
counterfactual evidence, not an established full-game performance improvement.
The rear-route reset comparison is recorded below.

The rear-route reset replay completed with the **same 805.35 score, 752.0-second
lifetime, 82.8-second estimated bait gap and zero premature sprint captures**
as the narrow guard alone. The reset was not exercised on this trajectory.
It is a repair of the stale exemption exposed in the broad pilot, not a
measured gain on the new default trajectory. Oscar's branch was rechecked at
`4542793`; recent additions concern his separate C++ guiding/holding experiments,
not new best survival parameters.

## Forest/swamp failure remains after the narrow guard

PC seed 204871, guide 160 at 547.9 seconds, was reproduced with zero position
error. Public DTO distance was 24.72; native evaluator distance was 19.22.
Walking moved only 5 units in swamp. The guard instead selected STOP, and
that exact counterfactual also died. It reported 63 of 81 sampled scenarios
capturing the guide. Thus the guard recognizes risk but its all-unsafe
fallback is inadequate; the bug is not fixed by recognizing danger alone.
Artifact: `guide-border160-guard-counterfactual.json`.

The recorded failure is visible at `http://localhost:9083/?time=545.5`, with
every frame through 549 seconds retained in
`logs/entrapment-iteration/sprint-border-204871-frames`. This is a partial
replay of a longer game, not an extinction at 549 seconds.

## Earlier terrain-risk guard probe (not adopted)

Extending the narrow lag guard from its immediate step to all three planned
steps, with river slowdown starting on either future step, regressed the
dev seed to **546.75 score / 519.4 seconds** (current default: 805.35 / 752.0).
There were two delivery arrivals from 17 assignments and 48.8 estimated
bait-gap seconds. This isolated probe does not contain the separate unsafe-
fallback repair. It is not adopted. Exact source is preserved in
`guide-terrain-guard-experiment.patch`, applicable to `8764dce`; adjacent
manifest and summary record the frozen source and result. All frames are in
`logs/entrapment-iteration/guide-terrain-guard-20260919`.
