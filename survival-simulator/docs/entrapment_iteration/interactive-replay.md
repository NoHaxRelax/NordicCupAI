# Interactive C++ replay loop

Baseline checkpoint: `entrapment-interactive-baseline-20260919` (`3a2391e`).
Experimental candidates are preserved separately under
`entrapment-research-candidates-20260919`; they are not the default policy.

The first interactive replay uses seed 1883894846, ordinary energy, native
predator spawning, and observation-only policy inputs. Requested horizon
3000 seconds; natural extinction at 626.8 seconds. Score 632.18. All 6,268 frames
are saved locally under `logs/entrapment-iteration/fast-recorded-seed1883894846`.
This is one inspection game, not an estimate of mean performance.

Run from repository root (current machine's pinned C++ checkout):

```sh
.venv/bin/python survival-simulator/scripts/fast_entrapment_game.py --fastsim /home/Ucals/projects/NordicCupAI-fastsim-inspect/survival-simulator --seed 1883894846 --seconds 3000 --out survival-simulator/logs/entrapment-iteration/NEW-RUN
.venv/bin/python survival-simulator/scripts/entrapment_viewer.py survival-simulator/logs/entrapment-iteration/NEW-RUN --port 9068
```

Use a new output directory for every iteration. The C++ checkout is pinned
to `696bbd86272c27d9edba56238d3ec3b9469faaf9`; its compiled extension is
Python/platform-specific. Build with that checkout's fastsim/build.py when
needed. Recorded manifests preserve policy and engine source hashes.

The viewer must remain in a persistent foreground terminal session. A child
backgrounded from a transient shell exited and made the first link fail.
Reload the page after recording completes to obtain the full frame range.

## Direct overlapping bait replacement (2026-09-19)

Exploration continues until our trap detector finds a site. Normal agents then
use Oscar's upstream tuned Orchard module and best configuration (a7a7d634),
with movement overrides for predator avoidance and dedicated bait/guide roles.
Normal reproduction is no longer suppressed merely because an agent is old
or currently avoiding a predator.

Replacement bait walks through the rear entrance directly to the bait spot.
There is no entrance standby role. `--bait-overlap 20` sets the target overlap
in seconds, using conservative travel and remaining-life estimates; actual
overlap is not guaranteed. Stalled replacements are released for reassignment.
Bait travel uses wall pathfinding and the normal-agent predator avoidance filter,
discarding the route after an avoidance move so it is recalculated. On the final
rear approach, observed predators within 40 units of the bait are exempted;
other observed predators still trigger avoidance. This proximity classification
is an estimate, not proof of capture. Arrived bait stays in place.

The same-seed replay is stored locally at
`logs/entrapment-iteration/interactive-overlap20-avoidance-v2-20260919`, served on port 9069.
Every frame is retained. Its manifest includes the JSON survival configuration
as well as Python and C++ source hashes. This single game is for visual inspection,
not a reliability benchmark.
