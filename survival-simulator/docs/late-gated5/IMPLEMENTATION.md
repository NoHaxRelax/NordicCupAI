# Implementation notes

- models/late_activation.hpp: time/population trigger and reversible, sticky or one-episode state. Uses only policy-call count and the number of agents in the public request.
- fastsim/_orchard_policy.cpp: parses a second parameter profile using late_ keys and selects the profile before any policy work each tick. The original predator controller remains unchanged.
- scripts/late5_config.py: five combinations and six-dimensional BO spaces; preset provenance in source-presets.json.
- scripts/preflight_late5.py: 352 full games on one 32-worker pod, default and always-active variants plus base; BO cannot start until every mean is <20 seconds.
- scripts/run_late5.py and late5_worker.py: durable shared queue, score optimization, raw timings and runtime eligibility, then frozen paired evaluation.
- scripts/report_late5.py: completeness checks, score/runtime intervals, paired gains, activation rules and BO evolution.
- scripts/watch_late5.py: console progress.

The optimizations preserve floating-point safeguards: finished native engine 3ccd187, policy optimization d6486e4, sharing optimization bc3069d. No compiler fast-math, PGO or host-native flags are enabled. The engine task measured those options and rejected them.

Verification: exact policy actions across two full games with activation delayed beyond the horizon; exact immediate-activation versus equivalent ungated late profile on a third full game. Total 43,484 checked action ticks. Trigger unit checks cover time, population, AND/OR, exact threshold and all persistence choices. The observation-only policy boundary check passes. This is finite regression evidence, not a proof for every possible map.
