# Write your own predator guide

Edit **`survival-simulator/models/entrapment/my_guide.py`**, specifically:

```python
def guide(bait, edges, agent, context, memory):
    return {
        'move_distance': 10.0,
        'move_direction': 3.141592653589793,  # backward
        'turn_angle': 0.0,
    }
```

The guide follows a cached **A\*** route to the handoff point while looking toward
the nearest sensed predator. `models/entrapment/guide_pathfinding.py` keeps routes 11 units
from walls (predator radius 10 plus margin), so agent-only gaps are excluded.
It smooths grid turns only along collision-clear segments. Its fixed internal
frame survives changes in the guide's position and facing direction.

Route cost is distance, not terrain-aware travel time: the function receives
all edges but only the guide's current biome. A feasible route does not guarantee
that the predator follows around corners. The guide retains the simple sprint
below 100-unit predator distance / walk otherwise rule. When contact is lost it
waits `LOST_WAIT_TICKS` (default 5, or 0.5 seconds), then walks back toward the
last observed predator position without turning. The remembered position
uses the fixed map frame, so turning does not corrupt it. If the exact point
cannot be reached with the route clearance, it tries the guide's last contact
position. At the search destination it holds. In every mode, an observed predator
centres the guide's vision on its bearing; without an observation the guide keeps
its heading. Delivery resumes when the following check permits it. This uses no
hidden predator tracking. No-route cases hold. This is not a proven
capture strategy; the remaining guidance logic is yours to improve.

`models/entrapment/guide_steering.py` applies a local movement selector to every observed-
predator action, including recovery. It prioritizes survival, then staying in
the predator's hearing circle or unobstructed vision cone, then the requested
route movement. It samples walking, sprinting, and stationary alternatives;
checks walls, terrain slowing, the low-energy sprint cap, and observed predators;
and centres gaze on the observed predator after accounting for our movement.
It uses a 48-unit safety distance (15-unit contact radius plus two possible
15-unit predator steps and a 3-unit margin). Contact targets are conservatively
55-unit hearing or 235-unit sight within a 25-degree half cone. These are local
estimates from potentially stale observations, not guarantees: unknown predator
motion, hidden predators, future terrain changes, and finite energy remain
limitations. If contact and survival conflict, survival wins; if no safe sampled
move exists, it favors affordable movement with the greatest separation.
Replay debug includes `steering.safe`, `clearance`, and `contact_error`.

`models/entrapment/predator_following.py` implements `predator_is_not_following`, imported
by `my_guide.py`. It compares consecutive observed predator poses to the moves
the native chase code could produce toward the guide. It accounts for direct
chase, watched pivots, the 11/15 energy-dependent speed caps, all four terrain
speeds, native collision-turn ordering, and the one-predator-movement delay in
observations. Two consecutive incompatible moves trigger the same recovery
behavior as lost sight; `memory['debug']['following_check']` shows the evidence.

The state starts as `not_following=False`. Compatible active translation switches
it to False; two incompatible movements switch it back to True. First sightings,
resting, gaps in observations, and ambiguous identity preserve whichever state
was already established. Missing contact still starts search behavior independently
of the latched following state. It does not read hidden target IDs, energy, terrain, or
rest state. Compatible movement does not prove that the predator targets this
guide. Exact vision occlusion is not used to reject candidate moves. Tolerances
and the mismatch count are constants at the top of the helper.

## Run and view

Commands below run from the **repository root**, using its existing virtualenv.

```bash
# Run one reproducible map, then serve the replay until Ctrl+C.
.venv/bin/python survival-simulator/scripts/guide_lab.py --seed 10224 --serve

# Open http://127.0.0.1:9056/ and select the printed run directory.
# Native frames: play/pause, arrow keys, frame slider, input/action inspector.
```

Keep the viewer server running and use a second terminal for further runs:

```bash
# Edit my_guide.py, then repeat the exact same encounter.
.venv/bin/python survival-simulator/scripts/guide_lab.py --seed 10224

# Different encounter on the same map; different bait site with --site 1.
.venv/bin/python survival-simulator/scripts/guide_lab.py --seed 10224 --encounter-seed 42

# Five sequential map seeds, 100 through 104. Single worker.
.venv/bin/python survival-simulator/scripts/guide_lab.py --seed 100 --maps 5

# A random seed, printed and saved so it can be repeated.
.venv/bin/python survival-simulator/scripts/guide_lab.py

# Serve existing recordings without running anything new.
.venv/bin/python survival-simulator/scripts/guide_lab.py --serve-only
```

Each run gets a unique directory in `survival-simulator/logs/guide_lab/`.
Nothing overwrites earlier runs. Refresh the server's directory listing to find
new recordings. `--seconds 120` changes the default 60-second limit;
`--width 960` increases the native replay resolution (default 640).
All recordings stay local under the existing ignored `logs/` directory.

## Thousand-map overview

Open <http://127.0.0.1:9057/> while the overview server is running. It shows
all 1,000 cases, outcome filters, seed/case search, survival, detection rates,
and selected-case details. Select a case and click **Load game replay** to
render every native frame locally from the frozen batch code. Two replay
workers run at most; generated replays are cached. No Runpod resources are
needed to use this viewer.

Restart the overview from the repository root:

```bash
.venv/bin/python survival-simulator/scripts/guide_batch_viewer.py survival-simulator/logs/guide_batch/runpod-20260918-1000
```

Results and limitations: [1,000-map evaluation](guide_batch_1000_results.md).
The bulk recordings preserve every policy input, action, and evaluation tick
in `ticks.jsonl.gz`; PNGs are generated on demand. Source hashes are checked
before the replay server starts.

## Inputs and action coordinates

Every point is expressed in the guide's **current local frame**:
origin at the guide; positive x forward, positive y right. Angles are radians
and positive angles turn clockwise in screen coordinates.

| Argument | Contents |
|---|---|
| `bait` | `(x, y)` of the actual bait agent, 5 units inside a predator-excluding gap |
| `edges` | List of every static edge as `((x1,y1), (x2,y2))`, including arena boundaries |
| `agent` | Ordinary simulator status: observations, energy, biome, age, speed, sprint speed, sensing traits |
| `context` | `tick`, `time`, `dt`, local `mouth` and `handoff` coordinates |
| `memory` | Your mutable dictionary, preserved between calls and reset per run |

The function receives no global coordinates and no hidden predator position,
target, energy, or rest state. Predator observations have native `distance`,
`angle`, and `rel_dir` fields. If a predator is no longer sensed, it disappears
from observations. There are no persistent predator IDs in the native DTO.
All edges and bait/handoff coordinates use **perfect static-map localization**,
an explicit convenience assumption for this lab. Stored local points need to
be transformed after movement/turning before reuse.

Return exactly `move_distance`, `move_direction`, and `turn_angle`, all finite
numbers. The harness supplies the guide ID and disables reproduction.
`move_direction` is relative to the **old heading**, and movement happens
**before turning**. Looking backward does not reduce movement speed.
Walking is `move_distance <= agent['speed']`; higher distances sprint, capped
at `agent['sprint_speed']`. The native low-energy sprint restriction still applies.

After the initial observation, the simulator's normal timing is preserved:
the input observations were computed before the previous predator movement.
Thus an observation is slightly stale; the harness does not refresh it with
privileged current predator positions.

## Scenario and results

- The map, biomes, obstacles, food, and trees are generated by the native simulator.
- A bait site is selected using only static geometry. Boundary-wall gaps count;
  the channel must also have rear access. Unsupported maps are reported, never
  silently rerolled as successful cases.
- The bait is predeployed, held still, reset to full energy every tick, and exempt
  from aging. It can still be eaten if the geometry fails.
- The guide is sampled in free space at least 250 units from bait. An awake,
  full-energy predator is sampled 80–160 units away and the guide faces it.
  Native sensing must confirm the predator is actually visible at tick 0.
  This is an arranged encounter, not a simulation of searching for a predator.
- The guide starts at full energy, age 0, and thereafter has normal energy,
  aging, terrain, food, and movement. No energy refill for the guide.
- The handoff point is outside the gap, at most 44 units from the bait. If a
  default predator catches a default guide there, its centre is within 59 units
  of bait, inside its 60-unit smell/hearing radius. Arriving near the bait does
  not itself guarantee capture; the guide still has to lead the predator there.
- The predator's native behavior, energy/rest, and ambient predator spawning
  continue after setup. The original predator is tracked only for evaluation.

`summary.json` reports capture time, whether the predator actually sensed the
bait when it caught the guide, and the guide's final energy/alive status. A
surviving guide is allowed. `delivery_proxy_pass` means the original predator
stayed within 40 units of the bait while sensing it continuously for 10 seconds
(`--hold` changes this). **This is a debugging proxy, not a proof of causal
guiding, physical entrapment, or full-game retention.** After guide death the
lab observes for up to `hold + 10` seconds, within the total time limit.

## Debugging

Set `memory['debug']` to any JSON-compatible value to see it beside each frame.
Your input and action are saved verbatim; the separate evaluation inspector
contains hidden ground truth which is never given to your function.

```bash
.venv/bin/python survival-simulator/scripts/guide_lab.py --seed 10224 --break-at 20
```

At the debugger prompt, `p inputs`, `p memory`, and `s` inspect/enter the policy;
`c` continues. You can also put `breakpoint()` directly in `my_guide.py`.
For VS Code, use this runner as the Python launch program with the desired seed
in `args`, then put breakpoints in `my_guide.py`.

Exceptions and invalid actions produce a `policy_error` result, with traceback
and the failing input preserved. Every completed tick and the initial state have
a native PNG and an input/action JSON file. The final frame has no next action.
`ticks.jsonl.gz` is the same per-tick data in one compressed file. Each run also
saves the exact policy, A* and following-check helpers, runner, and selector source and simulator file
hashes. Keep any additional helper modules versioned too if your policy imports them.
