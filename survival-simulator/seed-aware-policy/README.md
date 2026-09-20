# Seed-aware survival policy research

Active target: average score at least 2200 on at least128 fresh paired confirmation seeds. Not achieved. This is an experimental checkpoint, not a production controller.

Based on Lucas branch `survival-simulator/lucas-experimental`, commit `6bef2ccd3f129feaa9c2913d077987b072711784`. All simulation and policy ticks execute in C++. Python only compiles or orchestrates. No evaluation API calls.

The benchmark supplies exact current engine/model state after a configurable delay (180 simulated seconds in the recorded experiments). This is a conditional information upper bound, not proof of unknown live-seed recovery or an immutable future spawn schedule.

## Build and run

Ubuntu24.04: install g++, python3-dev, NumPy, pkg-config, nlohmann-json3-dev and libcpp-httplib-dev. Then `python3 build.py`. Run `bench/policy_bench configs/orchard.json SEED MODE 3000 OUTPUT_DIR 180`. Every completed run saves metrics and a state-only native replay.

## Current evidence

- Mode8: exact localization, static obstacles and current tree/fruit state. Development baseline.
- Full map orchard mode8 beat observation-only orchard on17/20 seeds: mean1617.71 versus1377.07.
- Fresh128-seed run: mode8 mean1747.90; modified allocation20 mean1776.77, paired gain28.87 with95% bootstrap interval[-38.16,96.43]. No promotion.
- Mypc24-seed batch: mode8 mean1823.47; routing23, routing+reach24, widerreach25 were worse.
- Modes26–28 test exact nearby predator poses outside public field of view, with wall/resting ablations. Preliminary16-seed evidence only.

Numerical portability is under investigation: a same-seed full baseline and its pre-activation action hash differ between the previous Runpod runtime and mypc NumPy1.26.4. Do not pool cross-host runs as deterministic replications. Within each frozen batch, paired modes use the same engine/runtime and pre-activation hashes are checked.

Recorded evidence includes all assigned seed results; large replays remain in the local experiment archive, not Git. Frozen historical experiments were retained separately. The default baseline remains mode8 until a fresh confirmation supports a replacement.
