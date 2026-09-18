# Integrated predator-trapping baseline

Merged `origin/survival-simulator/entrapment-9059-benchmark` at `6bb9b85a`
into `challenge1_nikolaj`. This is the integrated MVP combining Nikolaj's
observation-based exploration/map, Oscar's Orchard, and the trap coordinator.
The earlier `entrapment` branch contains the guide experiments; `oscar-trapper`
is a separate implementation and was not mixed into this one.

The incoming branch is authoritative. All 39 files listed in
`entrapment_native_seed0/manifest.json` match its exact SHA-256 hashes,
including the shared `src/core.py`, `src/elements/environment.py`, and
`src/utils/simulation.py`. Those three replace the older no-predator switches
and diagnostic hooks. `.gitattributes` preserves the required LF source bytes
on Windows. Existing uncommitted work was left in place and checked against
a pre-merge inventory.

## Run the merged policy

From `survival-simulator` in PowerShell, after the current Orchard run finishes:

```powershell
.\.venv\Scripts\python.exe -u scripts\entrapment_benchmark.py --out logs\entrapment-local --count 1 --seed-start 0 --workers 1 --seconds 3000
```

This verifies the frozen sources and runs one native game with natural
predator spawns. The controller RNG uses seed 0 independently of the world
seed. Results go to `logs/entrapment-local/000000/result.json`; progress is
written to `000000/progress.json`, and recorded trajectories are retained.
Completed cases are skipped when the same command is run again; an interrupted
case restarts from the beginning. Choose another output directory for a fresh
evaluation or a changed horizon.

The merged policy lives in `models/entrapment_policy.py`. Use the new
`scripts/entrapment_*` entrypoints: the old `local_playground.py`, BO scripts,
and diagnostics belong to the previous policy and expect its modified engine.
The independent `oscar-orchard-research` folder and its vendored engine remain
available; the merge does not change the currently running Orchard evaluation.

## Evidence and limits

34 focused upstream tests passed: integration, guide delivery, pathfinding,
steering, predator following, and trap geometry. All 39 frozen source hashes
were verified. No additional full simulation was started during the user's
running evaluation.

The branch's saved seed-0 reference reports 15 delivery arrivals, 23 overlapping
bait replacements, and 11 predators continuously near bait for at least
30 seconds. It went extinct at 1,073.6 seconds. These are historical MVP results,
not new measurements or proof of permanent capture; see
[the baseline report](entrapment_9059_baseline.md).

The incoming controller uses observations and simulation time; evaluator world
state is used only for recording and metrics. It retains its original static
arena assumptions. The separate Orchard process-isolation/boundary-inference
adapter is not applied to this frozen baseline, so that stricter audit does
not automatically carry over. The benchmark command above also avoids the
single-game research runner's world-seed-dependent controller randomization.
