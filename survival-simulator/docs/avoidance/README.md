# Orchard corner avoidance

Work branch: `survival-simulator/orchard-corner-avoidance`. Trapping is disabled
in these experiments; the existing trapping implementation is preserved.

`models/avoidance/policy.py` is the first Python prototype, using Nikolaj's
WorldEstimator and Oscar's Orchard. It supports plain avoidance, nearest
corner, and a fixed corner with fewer observed trees (an output proxy, not
knowledge of unseen fruit). Its strict variant requires an observed exit
within ten degrees. It has not demonstrated reliable directional exits.

`nightsim/` imports Oscar's native engine and Orchard from
`origin/survival-simulator/oscar-overnight-cpp` at `4542793`.
The initial import comprised __init__.py, _nengine.cpp, _npolicy.hpp,
build.py and run.py. Our C++ additions are in
`models/avoidance/native_corner.hpp`, with parameter/state/debug hooks in
those native files. The C++ policy currently uses Oscar's observed shared
map and boundary anchoring, not a port of Nikolaj's estimator. It assumes
the standard public 1600 x 1200 arena. Python prototype uses Nikolaj's map.
Native oracle, trap, hide, refuge and freeze options are all zero in the
corner comparison configs. Native AState contains the same own-agent fields
and cached sightings as the ordinary DTO; no predator/world state is passed
to the steering function.

The 16-game native baseline, seeds 1–16, averaged **1763.68 score**.
It took **58 seconds elapsed** on three local workers (mean worker runtime
9.58 seconds/game). This is a measured baseline, not a claim to have matched
another agent's 2000 mean. No paid compute used.

The Python dev-seed prototype scored 743.66 versus 1173.53 for ordinary
avoidance, and its safer strict revision scored 1044.80 with no confirmed
aligned exits. These are failed exploratory variants, not defaults to deploy.

The native paired comparison tests shared avoidance, nearest-corner ±10°,
and sparse-corner ±10°. Record alignment separately from score. An exit
outside tolerance, a lost sighting, a death, or budget exhaustion is not
success. `aligned` is an observed heading result at escape, not a promise
that a wandering predator will remain facing that way indefinitely.

Build: `python nightsim/build.py` (from survival-simulator).
Batch: `python nightsim/run.py --configs docs/avoidance/native-corner-configs.json --seeds 1-16 --predators --workers 3 --out logs/avoidance/results.jsonl`.

## Completed comparisons and tracking correction

On seeds 1–16, native shared avoidance scored 1583.63, nearest-corner v1
1597.74, and sparse-corner v1 1583.68. With deliberate gaze offset and the
original non-shared avoidance settings, nearest ±10 scored 1661.29,
nearest ±20 scored 1646.24 and sparse-corner ±10 scored 1571.52.
None beat the original 1763.68 avoidance baseline.

**The early directional counters are invalid.** A new nearest sighting could
replace the predator being steered. In the displayed seed-1 replay, all three
apparent successes switched targets; the original predators were roughly
44°, 47°, and 146° off target. See `target-switch-audit.json`. This is why a
provisional alignment count must not be presented as an achieved ±10° exit.

The current C++ controller matches consecutive sightings by position and
heading, rejecting ambiguous matches and jumps. A new inexpensive variant
only starts within 30° of the desired direction and spends at most two
seconds trying. A fresh paired comparison on seeds 17–32 is running.
This is research code; directional steering is not established as beneficial.

The full-game native recorder is `scripts/record_native_avoidance.py`.
It uses native `policy_act` and ordinary actions, retaining each tick for
`scripts/entrapment_viewer.py`. Example current review URL:
http://localhost:9092/?time=786.1 (historical target-switch failure).

## Target-locked results and verified replay

The fresh paired tests are complete:

| Seeds | Plain avoidance | Sparse-corner, ±10° | Difference |
| --- | ---: | ---: | ---: |
| 17–32 | 1481.53 | 1511.00 | +29.47 |
| 33–48, confirmation | 1637.33 | 1597.08 | −40.24 |

Across these 32 maps the sparse-corner variant is about 5.4 points worse.
This does not establish a score improvement. Keep the existing plain
avoidance configuration as the default. Nearest-corner steering scored
1420.99 on seeds 17–32, also below plain avoidance.

The corrected implementation made five **observation-based alignment
claims in 204 attempts** across those 32 sparse-corner games. One has been
independently checked against the original native predator; do not treat
all five as independently verified or claim reliable steering.

Verified example: seed 21, agent 294, 548.9–550.1 s. At exit the original
predator is 9.601° from the selected corner direction, the agent is 94.225
units away, and its bearing is 54.224° off the predator's heading, outside
the 30° half-cone. Native state is used only by the after-the-fact auditor.
`locked21-alignment-audit.json` contains the measurements;
`scripts/audit_replay_alignment.py` reproduces them from the full recording.

**Current review replay:** http://localhost:9093/?time=548.8 . Every tick of
the 1585.6-second game is saved (score 1611.5). The earlier port 9092 remains
a historical failed-counter example; use 9093 for the corrected behavior.

The four-map escape-angle/commitment sweep also failed to beat its plain
avoidance control. Results and exact configs are retained. No additional
Survival changes were adopted. All new compute in this worktree was local:
**Runpod spend $0**.

New native batch runs record source hashes, loaded binary hash, runtime,
seeds and configs in a companion manifest. The runner refuses stale builds
and refuses to append mixed revisions to an existing nonempty output.
