# Predator trapping: completed 10,000-game experiment

All 10,000 C++ games completed successfully on eight Runpod CPU pods. The
experiment confirms both moving confinement and completely blocked predators.
Escape is not guaranteed by the current movement logic.

The follow-up [analysis of every trap location](predator_stuck_spots.md) adds
geometry requirements, pose recurrence, independently reconstructed collision
outcomes, counterexamples and rules evaluated on separate maps.

All eight compressed archives were downloaded and checked against their remote
SHA-256 hashes before deleting the pods. The final API audit confirms all eight
pod IDs are deleted. No workers from this experiment remain running.

## Experiment and results

Seeds 0–9,999 each started with 100 predators and no agents, and ran for 600
simulated seconds at 0.1-second timesteps. A finding requires every position
over a full 60-second interval to remain within 15 units of the interval's first
position. Every rolling interval is considered; only the first finding per
predator is counted. Native ambient spawning remained enabled.

| Measurement | Result |
| --- | ---: |
| Completed games / failed games | 10,000 / 0 |
| Games with at least one finding | 9,969 |
| Initial predators | 1,000,000 |
| Total predators, including ambient births | 1,001,351 |
| Predators observed for at least 60 seconds | 1,001,083 |
| Predators meeting the confinement criterion | 248,090 (24.78% of eligible predators) |
| Moving confinement: no active tick rejected every candidate | 231,516 (93.32% of findings) |
| Every active tick rejected every candidate | 16,139 (6.51%) |
| Mixture of successful and completely blocked movement | 435 (0.18%) |
| Later left the original 15-unit test circle | 326 |
| No departure observed before game end | 247,764 |

Follow-up ends at 600 seconds, so the last row does not mean permanently stuck.
229,109 findings had at least another 60 seconds of observation after detection.

## What the evidence says about the mechanism

The dominant pattern is repeated movement near obstacle edges. 227,570 findings
(91.73%) used a 90-degree turn on at least 95% of active ticks. 228,992 findings
(92.30%) returned near their position from four active moves earlier on at least
90% of eligible comparisons. The latter also includes shorter repeating paths,
such as two-step oscillations; it does not establish four distinct vertices.

The engine explains why these cycles need not break spontaneously:

- Edge avoidance is deterministic. Its turn magnitude is
  `pi / max(closest_visible_edge_distance - predator_radius, 2)`, so clearance
  of two units or less produces a 90-degree turn.
- Random heading jitter is sampled in the wandering branch. A predator that
  keeps seeing edges can stay in avoidance without receiving that jitter.
- A blocked requested move triggers a search through alternative directions.
  The accepted movement direction can differ from the heading change, which
  still follows the original requested turn. This can sustain a small cycle.
- Biome movement penalties change the step length and therefore the size of
  the repeating path. Resting pauses movement without independently choosing
  a new escape direction.

Of the moving-confinement findings, 228,249 used at least one successful fallback
move during the detected interval; 3,267 did not. Fallback is strongly associated
with these findings but is not required by every observed cycle.

Spawn overlap is a separate contributor. Predators with legal initial placements
accounted for 215,417 findings, so initial overlap does not explain the main
pattern. Among eligible predators, 23.50% of legal spawns and 38.77% of overlapping
spawns qualified. At interval onset, 16,135 of the 16,139 fully blocked predators
overlapped an expanded collision rectangle. These are descriptive associations,
not a proof that overlap alone is sufficient or necessary for every blocked case.

The 9,993 unflagged control windows contained 39 predominantly quarter-turning
predators and 45 with the repeating-position pattern. Controls use the first
unflagged predator per game, so they are not random or matched samples.

## Controlled replays

Two Windows pilot cases were replayed from complete world/RNG checkpoints for
another 120 seconds. Each baseline was checked against its saved event before
testing changes. These are case-specific interventions, separate from the
unmodified Linux batch.

| Intervention | Seed 1, predator 84: moving cycle | Seed 1, predator 32: fully blocked |
| --- | --- | --- |
| Unchanged | Stays within about 4.67 units | Remains motionless |
| Heading +1 degree | Exits after 10.8 seconds | Remains blocked |
| Heading -1 degree | No exit observed | Remains blocked |
| Position one unit to the right | Exits after 2.8 seconds | Remains blocked |
| Remove a nearby relevant obstacle | Each of the three closest removals permits exit | Removing overlapping obstacle 12 permits exit after 0.3 seconds |
| Remove a distant obstacle, as a control | No exit observed | Remains blocked |

The first case demonstrates sensitivity to heading and geometry. The second
demonstrates a collision obstruction that the tested heading changes cannot
resolve. Together they support testing two separate fixes: preventing or
recovering invalid overlapping placements, and detecting/recovering movement
cycles in avoidance. No movement fix was applied during this measurement run.

These experiments do not establish universal necessary and sufficient conditions
for every trap. The saved logs and intervention script support investigating
additional cases without repeating the full batch.

## Saved data and tools

- [Merged analysis](../runs/predator-stuck-diagnostics-10000/diagnostic_analysis.json)
  contains counts, group comparisons, example event paths, source hashes, shard
  verification metadata and complete seed coverage.
- [Merged findings CSV](../runs/predator-stuck-diagnostics-10000/diagnostic_findings.csv)
  contains all 248,090 findings, including heading-pattern counts, fallback and
  blocking counts, geometry clearances, detection time and later escape.
- [Raw dataset directory](../runs/predator-stuck-diagnostics-10000/) contains
  eight compressed archives totalling about 17.51 GB. Each includes per-game
  maps/biomes, predator catalogs, approach and confinement traces, controls,
  runtime/build metadata and the deployed source. Keep the archives compressed
  and extract only cases needed for inspection.
- [Pod deletion audit](../runs/predator-stuck-diagnostics-10000/pod-cleanup.json).
- [Moving-case replay](../runs/predator-stuck-diagnostics-replay/seed1-p84.json)
  and [blocked-case replay](../runs/predator-stuck-diagnostics-replay/seed1-p32.json).
- [Diagnostic schema and commands](predator_stuck_diagnostics.md) explain the
  retained fields, native build, replay and optional HTML case viewer.
- [Scanner](../scripts/predator_stuck_scan.py),
  [C++ implementation](../scripts/predator_stuck_cpp/_stuck.cpp),
  [replay tool](../scripts/predator_stuck_cpp/replay.py), and
  [report merger](../scripts/predator_stuck_cpp/merge_reports.py).

Each event includes up to 30 seconds before confinement and the full 60-second
interval, with movement decisions and rejected candidates at each native tick.
Later escape is monitored through game end; the full later trajectory is not
stored for every predator. All predators have lifetime motion counters. The
merged `sum_of_predator_longest_blocked_active_runs` field is a sum of individual
maximum run lengths, in active ticks, not the maximum of the population. Older
per-shard reports call this summed field `longest_blocked_active_run`.

## Validation and limits

Every pod passed native/Python parity checks on two 65-second games before
starting its shard, including predator state, RNG state and recorded movement
endpoints. Detector checks and Windows parity also passed. Per-shard analysis
validated event duration, sample counts, confinement radius, lifetime tick
accounting and trace existence. The merger verified identical configuration and
simulation-source hashes, zero failures, matching CSV counts, and exactly one
copy of every seed from 0 through 9,999.

The replay evaluator was added locally after launching the distributed scan;
archive source and build metadata identify the exact deployed version. The
native simulation physics was unchanged. Floating-point trajectories can differ
between platforms; the replay tool rejects a mismatched baseline.

This detector intentionally measures confinement to the requested circle. It can
miss a larger repeating path: an 11-unit square has a 15.56-unit diagonal from
its starting corner. Findings also depend on no agents being present; pursuit
can change the movement branch. Biome counts are not exposure-adjusted risk
estimates. The optional HTML viewers were generated but browser preview was
unavailable, so their visual presentation has not been checked in a browser.
