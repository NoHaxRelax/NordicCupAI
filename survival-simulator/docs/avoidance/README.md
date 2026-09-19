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
