# Predator trapping diagnostics

The full run is complete: 10,000 games, zero errors, all eight archives verified
locally and all eight Runpod workers deleted. See the
[results report](predator_stuck_results_2026-09-19.md) for findings and data links.

The experiment runs seeds 0–9,999, no agents, exactly 100 successful initial
predator spawns per game, and 600 simulated seconds at the native 0.1-second
timestep. A finding is the first 60-second interval whose 601 positions all
remain within 15 units of its starting position. Ambient births remain enabled.

The compiled engine records diagnostics passively. Logging consumes no random
numbers and does not change movement decisions. Before starting each Runpod
shard, `verify.py` compares two 65-second instrumented games against the Python
engine, including every predator state, RNG state, confinement event, and the
endpoint reconstructed from logged movement decisions.

## Data retained

For each flagged predator, the compressed event file contains:

- All 601 confinement samples, including heading, energy and resting state.
- Up to 30 seconds of approach before the interval, plus every transition in
  the interval: 901 diagnostic rows when enough history exists.
- Pre- and post-movement state and the biome before moving.
- The decision branch: rest, edge avoidance, random wander, or prey pursuit.
- Visible-edge count, local-obstacle count, the selected edge and closest point
  in the observation frame, its clearance and angle.
- Requested movement, relative direction, turn, biome-scaled step distance,
  and absolute attempted movement direction.
- The number of candidate moves, a bitmask of rejected candidates, the accepted
  candidate index, first blocking obstacle ID, and bounds-clamping flag.
- Exact obstacle geometry at interval onset and detection: overlapping obstacles,
  nearest signed L-infinity clearances to expanded collision rectangles, and
  distance to the world boundary.
- Follow-up through the end of the game: first departure from the test circle,
  maximum later distance from its anchor, and final position.

Every game also stores its full obstacle map and compressed biome grid. Every
predator, including unflagged predators, has lifetime counters for rest, edge
avoidance, wandering, blocked moves, fallback successes and distance travelled.
One unflagged predator per game contributes a full final diagnostic window as
a descriptive control. The control is the lowest-index unflagged predator, not
a random or matched sample.

This is incident-focused logging rather than a recording of every tick of every
predator. Seed, source hashes, platform build metadata, map, birth records, and
the native engine allow full-game replay when additional detail is needed.

## Reading the movement log

The event embeds its schema in `diagnostic_format`. Row tick `t` describes the
transition from `t-1` to `t`. Birth rows have mode `-1`.

Candidate 0 is the original movement. Candidates 1–36 use the native fallback
order: a duplicate original direction, then -10°, +10°, -20°, +20°, through
-180°. Bit `n` in `rejected_candidate_mask` indicates rejection of candidate `n`.
`accepted_candidate=-1` with an active decision means every candidate failed.
The heading still follows the original turn, independently of which fallback
direction moved the predator. Obstacles are square-expanded by predator radius
and tested with strict inequalities. Obstacle IDs index `map.json`; IDs 0–3
are world boundaries.

Analysis distinguishes completely blocked movement, a mixture of blocked and
successful movement, and confinement while continuously moving. It counts turns
at the 90° limit and near-returns after four active moves. Stationary predators
are excluded from the four-step moving-cycle classification.

The requested radius can miss larger repeating paths. For example, four 11-unit
steps separated by 90° turns form a square with a 15.56-unit diagonal, exceeding
the 15-unit circle anchored at a vertex. The control-window cycle counts can
therefore include a repeating path that does not meet this particular detector.
The experiment retains the requested radius rather than silently changing it.

## Controlled replay

Build the native extension, then replay a saved case:

```powershell
& "..\NordicCupAI\survival-simulator\.venv\Scripts\python.exe" survival-simulator/scripts/predator_stuck_cpp/build.py
& "..\NordicCupAI\survival-simulator\.venv\Scripts\python.exe" survival-simulator/scripts/predator_stuck_cpp/replay.py <event.json.gz> --seconds 120 --output <replay.json>
```

The replay first verifies that its baseline checkpoint exactly matches the
recorded event. It then copies the complete world and RNG state for each
intervention: small heading changes, position offsets, or removal of one nearby
obstacle. The geometry variant rebuilds collision and visibility indices.
Invalid overlapping starting positions are explicitly marked. Escape is tested
at every native tick; replay path previews are sampled once per second.

Replays are separate experiments. The 10,000-game batch retains the original
physics. A perturbation that frees one predator establishes a result for that
case, not a universal cure. Floating-point trajectories can differ across
platforms, so replay rejects a mismatched baseline instead of interpreting it.

## Pilot findings

For Windows seed 1, 20 predators qualified by 600 seconds. One was completely
blocked after spawning inside an expanded collision rectangle. The other 19
kept moving inside the test circle and spent at least 95% of their active ticks
turning by 90°. None left its test circle before the game ended.

Predator 84 illustrates the moving trap. From its detection checkpoint, the
unchanged world remained within approximately 4.67 units for another 120 seconds.
A +1° heading perturbation escaped after 10.8 seconds; -1° did not. Moving the
starting position one unit to the right escaped after 2.8 seconds. Removing any
one of its three closest obstacles also allowed escape. This supports a
geometry- and heading-sensitive movement cycle, rather than every attempted
move being blocked. These results are in
`runs/predator-stuck-diagnostics-replay/seed1-p84.json`.

Predator 32 remained motionless under all tested heading and one-unit position
perturbations. Removing its overlapping obstacle freed it within 0.3 seconds;
removing either of the other two nearest obstacles did not. See
`runs/predator-stuck-diagnostics-replay/seed1-p32.json`.

`view_case.py <event.json.gz> --output <viewer.html>` creates a self-contained
step viewer with the approach, obstacle geometry, candidate directions, and
movement state. Examples are `movement-loop.html` and `blocked-spawn.html` in
the replay directory. Browser preview was unavailable during this session, so
the generated viewer has not had a visual browser check.

## Runpod execution and analysis

Eight pods process disjoint seed ranges: one 32-vCPU pod handles seeds 0–3,636;
seven 8-vCPU pods handle 909 seeds each. Their combined rate is $2.72/hour while
all are running. Each has a local watchdog that stops it within two hours of
creation; loss of the local machine or API connectivity can prevent that guard
from acting. The worker is deleted after its completed archive is retrieved and
its SHA-256 verified.

`runs/predator-stuck-runpod-launch/fleet.json` records pod IDs, ranges, resources
and connection endpoints. Per-shard setup logs and parity reports are retained.
Full archives and compact reports are collected under
`runs/predator-stuck-diagnostics-10000/`. Archives remain compressed to avoid
duplicating the large trajectory dataset locally.

`scripts/predator_stuck_cpp/analyze.py <output-directory>` produces
`diagnostic_analysis.json` and `diagnostic_findings.csv`. The analysis validates
event duration/sample counts, lifetime tick accounting and trace presence. It
compares spawn-overlap groups, moving and blocked confinement, turn/cycle
patterns, unflagged controls, and observed recovery. It does not interpret
failure to escape by 600 seconds as proof of permanent trapping.
