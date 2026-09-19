# Native entrapment controller

All per-tick decisions and simulation steps run in C++17. Python only starts
independent games and writes benchmark/replay files.

Build from the repository root:

```bash
.venv/bin/python survival-simulator/native_policy/nightsim/build.py
```

The final entrapment preset is `configs/entrapment-final.json`. The older
`with-predators-best.json` is an orchard/evasion baseline: its trap mode is OFF.

## Files

- `nightsim/_npolicy.hpp`: observed mapping, exploration, Oscar's orchard survival,
  shared predator avoidance, trap discovery, replacement bait and guide roles.
- `nightsim/entrapment_guide.hpp`: three-tick beam search for guide steering.
- `nightsim/_nengine.cpp`: native simulator, DTO conversion and policy entry points.
- `nightsim/_evaluation.hpp`: measurements after decisions; never policy inputs.
- `evaluate.py`: reproducible full-game batch runner and score histogram.
- `../scripts/native_policy_replay.py`: every-frame recording for the existing
  game-sprite viewer.

## Final behavior

Energetic agents explore observed-map frontiers until their group finds a site;
food collection and reproduction continue. A site needs a 10.1–19.9-unit gap,
at least 20 units of overlap, a clear approach and a verified radius-5.01 rear
route. Bait stands 5 units into the gap. Replacement dispatch accounts for route
length, conservative terrain speed, aging and energy, with a 20-second target
overlap. Replacements enter immediately and may take nearby ripe fruit when
arrival and survival margins permit.

A currently detected pursuit can become a guide. Its search samples three
future ticks, predator speeds, pivots and terrain slowdowns. Distance 100–120 is
a preference, not a collision rule; capture uses the actual 15-unit threshold.
A following guide can stand still within 44 units of bait on the front side.
Guide deaths there are intentional sacrifices; others are premature captures.
Non-guides avoid shared observed predator detection regions, including the next
planned movement. Incoming bait is exempt only after reaching its rear entrance.

## Provenance and limits

The native foundation came from Oscar's `oscar-overnight-cpp` branch, imported at
`31743d1`. This is a C++ **adaptation**, not a bit-for-bit translation of the prior
Python combined controller. It uses Oscar's observed mapping/exploration rather
than Nikolaj's Python explorer, and one active guide/trap per connected group.
The guide beam search is ported from `models/entrapment/guide_lookahead.py`.

Policy inputs are ordinary observations, public agent traits, and memory.
`oracle_trees` is rejected by the native API; parked-predator debugging is also
rejected in the evaluation loop. Bait has normal energy and aging. Hidden
positions are read only by the evaluator/recorder. Standard map dimensions are
1600 by 1200; custom dimensions are unsupported by this controller's pathfinder.

Observation lag, uncertain localization, unobserved walls and competing targets
remain limitations. Finite lookahead does not guarantee survival. If groups with
different occupied traps merge, secondary bait remains stationary but only the
primary site receives replacement scheduling. Corners are not implemented in
this final native variant. No 95% delivery or zero-gap claim is implied.

`evaluate.py` freezes source/configuration hashes and writes every seed, including
failures. Arrival, proximity for 30 seconds, and confirmed permanent retention
are distinct metrics; the last is not currently measured. See
`../docs/benchmarks/native-evaluation-method.md` for commands and definitions.
