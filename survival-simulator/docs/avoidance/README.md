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
