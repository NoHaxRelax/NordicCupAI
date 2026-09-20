# Optimization scripts review

Review date: 19 September 2026. Revision: `282446421263e7bb11081d692f0ac443b99f9e20`.

The highest-value work is to restore the evaluator's simulator contract, prevent incompatible cached experiments from being reused, and make the search measure late survival. Faster sampling will only help after those issues are addressed.

The main tuner has a sound foundation: one journal coordinator, per-episode locks, atomic checkpoints, separate training and validation seeds, a mandatory baseline comparison, and exclusion of interrupted episodes from scored trials. Preserve these safeguards.

## Coverage and verification

The repository inventory contains **642 tracked Python files and 2 shell scripts**. All 642 Python files parsed successfully. Detailed review covered `tune_policy.py`, `tuning_evaluation.py`, the three top-level benchmarks, laptop/HPC launch instructions, active configuration and controller paths, simulator integration, tests, guide/entrapment batch runners and reports, orchard sweeps and their harness, and representative frozen trap-research runners. The drone and medical directories contain evaluation clients rather than the simulation parameter optimizer; they were screened for scope. Historical policy variants, saved experiment data, and every archived research runner were not individually behavior-tested.

Checks used Windows, Python 3.12.12, and the installed versions matching `requirements-tuning.txt`: Optuna 5.0.0, NumPy 2.3.5, SciPy 1.16.3, Pydantic 2.12.4, Pygame 2.6.1, and Shapely 2.1.2. Imports were verified to resolve to this checkout's simulator.

| Check | Result |
| --- | --- |
| Parse every tracked Python file | 642 passed |
| Tuning unit tests | 17 passed, 1 error |
| Full `survival-simulator/tests` unittest discovery | **521 passed, 3 errors**, 117.4 seconds |
| Guide-batch stale-cache probe, with subprocess execution blocked | Accepted a cached result with different world and encounter seeds |
| Direct evaluator cache probe | Accepted a same-seed/same-horizon result with different supplied configuration dictionaries; callers must currently provide identity protection |
| Simple-policy search-space probe | Accepted `policy_mode="simple"` while constructing `SimpleCoordinator`, which uses a different configuration space |
| Initialization-only profile, default native world, seed 1 | 15.739 seconds under cProfile; 14.180 seconds in biome rendering; no episode steps executed |

No optimization campaign, remote compute job, or competition evaluation was launched. This is a report-only change. Performance proposals below are opportunities to measure, not claimed speedups.

The three test errors are:

1. `test_harvest.PredatorFreeTests.test_disables_initial_later_and_explicit_spawns_and_survives_reset`: unexpected `predators_enabled` constructor argument.
2. `test_run_diagnostics.DiagnosticsTests.test_recording_does_not_change_simulation_trajectory`: `Environment` has no `event_sink` attribute.
3. `test_tuning.TuningTests.test_episode_checkpoint_restores_world_policy_rng_and_metrics`: unexpected `predators_enabled` constructor argument.

Reproduce the suite from `survival-simulator` with the tuning dependencies installed:

```text
python -m unittest discover -s tests -p test_tuning.py -v
python -m unittest discover -s tests -v
```

## Priorities

P0 blocks the current workflow. P1 can invalidate results or undermine the primary search objective. P2 improves experiment reliability, compute use, or statistical confidence. P3 is follow-up hardening. Effort is relative implementation size, not a time estimate.

| ID | Priority | Improvement | Evidence | Effort |
| --- | --- | --- | --- | --- |
| F01 | P0 | Restore the predator-free simulator interface | Reproduced test failure | Small–medium |
| F02 | P1 | Restore and validate event diagnostics | Reproduced failure; absent event producers | Medium |
| F03 | P1 | Validate experiment identity before batch cache reuse | Reproduced guide-batch mismatch; entrapment source inspection | Medium |
| F04 | P1 | Search on a horizon that exercises late survival | Default horizon and controller schedules | Medium |
| F05 | P2 | Make the search space specific to the selected policy | Construction probe and controller call paths | Small–medium |
| F06 | P2 | Bring standalone benchmarks up to the tuner's identity standard | Manifest and comparison code | Medium |
| F07 | P2 | Make orchard sweeps recoverable and preserve run versions | Sweep/harness persistence paths | Medium |
| F08 | P2 | Record complete orchard provenance and bundle replay support | Harness hash path and bundle allowlist | Small |
| F09 | P2 | Bind guide results to the exact subprocess attempt and protocol | Runner command and receipt selection | Small–medium |
| F10 | P2 | Schedule seeds according to free worker capacity | Candidate-width scheduling code | Medium |
| F11 | P2 | Remove eager diagnostic serialization from every policy tick | Snapshot call paths | Small–medium |
| F12 | P2 | Profile and amortize world initialization/checkpoint costs | Rendering/RNG and checkpoint code | Medium–large |
| F13 | P2 | Add uncertainty and an untouched final test stage | Three-seed winner selection | Medium |
| F14 | P2 | Gate changes with integration tests and a reproducible environment | Current failures and dependency manifests | Small–medium |
| F15 | P3 | Harden resume transactions, failures, and publication | Coordinator/report persistence code | Medium |

## Findings

### F01 — The main optimizer cannot start a fresh episode

**Evidence:** [tuning_evaluation.py](../tuning_evaluation.py), line 201, calls `SimulationCore(seed=seed, predators_enabled=False)`. [src/core.py](../src/core.py), lines 9–11, accepts `starting_predators` but has no `predators_enabled` parameter. [benchmark_harvest.py](../benchmark_harvest.py), line 56, makes the same call; `benchmark_survival.py` uses the shared evaluator. The existing predator-free and checkpoint tests reproduce this failure.

**Impact:** A fresh BO trial fails before running its episode. A one-line replacement with `starting_predators=0` would also be incorrect: [environment.py](../src/elements/environment.py), lines 759–762, continues spawning predators during the game.

**Recommendation:** Reconcile the evaluator and simulator versions. Implement an explicit research scenario adapter or restore the supported switch, preserving native behavior by default. The switch must cover initial, explicit, later, and post-reset spawns. Run the existing predator-free test and the checkpoint integration test before starting another tuning campaign. Make the resolved simulator path and supported scenario visible during preflight.

### F02 — Event metrics are disconnected from the simulator

**Evidence:** The evaluator installs an `event_sink` at [tuning_evaluation.py](../tuning_evaluation.py), lines 219–250. The checked-in `Environment` neither initializes that attribute nor emits the expected birth/death/fruit callbacks. [run_diagnostics.py](../src/utils/run_diagnostics.py), line 63, reads it immediately and raises `AttributeError`. [benchmark_mapping.py](../benchmark_mapping.py), lines 185–214, expects the same event mechanism for lineage and population counts.

**Impact:** After fixing F01 alone, ordinary evaluation can run with zero birth/fruit/death event counters because assigning a new Python attribute does not make the simulator call it. Mapping lineage can remain founder-only. Score and final population come directly from simulator state and are separate from this missing diagnostic stream; they should not be conflated with the incorrect counters.

**Recommendation:** Define a single event contract with documented emission timing, cause names, and parent identifiers. Attach it through a tested observer interface. Verify at least one known birth, death, eaten fruit, and rotten fruit; check score-component reconciliation and observer-on/off trajectory equality. Assert observer support at startup so unsupported instrumentation cannot silently produce plausible zeroes. Preserve simulator RNG consumption and list-update behavior when observing events.

### F03 — Batch resume can silently reuse results from another experiment

**Evidence:** [guide_batch.py](../scripts/guide_batch.py), lines 42–52, overwrites `manifest.json` and then accepts any existing `case-N/result.json`. A temporary-directory probe requested world seed `561019409` and encounter seed `399920095`, but returned a cached success for seeds `123` and `456`. No simulation was needed to reproduce it. The later guide report detects seed/index disagreement, but cannot detect every same-seed change in policy, horizon, or scenario.

[entrapment_benchmark.py](../scripts/entrapment_benchmark.py), lines 242–251, similarly overwrites the protocol and reuses existing per-seed results without comparing their horizon to the requested horizon. Source verification protects the currently loaded frozen source, not the identity of each cached run. Its merge tool catches some inconsistencies later; the runner itself can still publish a mixed aggregate.

**Recommendation:** Validate an immutable manifest before any write or cache reuse. Key episodes by a digest of simulator and policy sources, resolved configs, world and policy/encounter seeds, horizon, timestep, scenario, runtime versions, and metric schema. Store that digest in each result and checkpoint. Add a coordinator lock and atomic result publication to the guide runner. Handle failure retries explicitly rather than treating every existing error result as final forever.

**Acceptance check:** Changing any scientific setting in a populated output directory must fail clearly or create a distinct study. Repeating the identical command should reuse valid completed work. A source change with unchanged seeds must be tested as well as seed changes.

### F04 — The default search misses the late-game behavior it is meant to improve

**Evidence:** [tune_policy.py](../tune_policy.py), lines 450–453, trains for 600 seconds and validates for 3,000. The HPC launch script uses the same horizons. [harvest.py](../src/utils/controllers/harvest.py), lines 398–423, changes population scheduling at 900 and 1,800 seconds. The conservation schedule reaches its full effect at 1,800 seconds. These late regimes never execute in the default training episodes. Finalists are selected solely by training rank.

**Impact:** A configuration that trades early fruit score for substantially better late survival can be discarded before full-horizon validation. The existing [survival findings](../SURVIVAL_FINDINGS.md) already document large seed-dependent late-survival effects; those are historical experiments, not fresh results from this review.

**Recommendation:** For survival work, use a new study with a 3,000-second training horizon. Keep actual game score as the default objective when score is the goal; report survival probability and survival time alongside it. If species survival is the explicit goal, introduce a separately named, versioned objective rather than silently changing the score definition.

For a tighter budget, investigate promotion by **number of completed full-horizon seeds**: start with a small common seed block, then evaluate promising candidates on larger blocks. Keep comparisons within the same block/fidelity and never feed truncated episodes into ordinary full-horizon score observations. Do not add early-score pruning until it has been shown to retain late winners.

Also measure how often each tuned control is active. With the current middle/late floors equal to their caps, changing `food_budget_per_agent_second` cannot change the scheduled population target in those phases. Late caps, renewal timing/reserves, and conservation timing are more direct candidates for a separate, constrained survival search.

### F05 — The tuner accepts a policy mode that bypasses most of its search space

**Evidence:** [tune_policy.py](../tune_policy.py), lines 38–47 and 480–491, defines a fixed harvest-oriented search space and checks mapping/selection flags, but not `policy_mode`. [expert_policy.py](../src/utils/controllers/expert_policy.py), lines 105–117, selects `SimpleCoordinator(config.simple)` in simple mode. The configuration-construction probe accepted this combination.

**Impact:** At least six dimensions—the three coverage parameters, food budget, ripe energy, and harvest parent reserve—configure the unused `HarvestCoordinator` in simple mode. The emergency threshold still affects shared action logic, so this is not a claim that every dimension becomes inactive. A user selecting simple mode through the supplied expert config can waste most of the search budget.

**Recommendation:** Immediately reject unsupported modes. Subsequently define a policy-specific search-space registry with constrained schedules for simple mode, standard harvest, and any future orchard/entrapment optimizer. Record active dimensions in the manifest and add small parameter-effect checks using relevant observation fixtures. Check sampled configurations against model constraints before launching workers.

### F06 — Standalone benchmarks provide weaker comparison and resume guarantees

**Evidence:** [benchmark_survival.py](../benchmark_survival.py), lines 60–62, records configs, seeds, horizon and `source_hash()`, but omits the Python/system/package identity included by the main tuner. The imported hash covers the tuner, evaluator and `src`, not the benchmark script itself. [benchmark_harvest.py](../benchmark_harvest.py), lines 18–25 and 115–116, pairs existing files by seed and writes filenames without horizon or configuration identity.

**Impact:** The standalone survival benchmark can mix completed results or checkpoints across dependency changes. Running only the centralized harvest variant into an old output directory can pair a new 180-second run with an old baseline of a different horizon or config. The direct evaluator only verifies seed and horizon on completed cache reads; it relies on callers for the rest of this protection. The main tuner's stronger manifest currently supplies that protection.

**Recommendation:** Share one experiment-identity builder among all runners. Require matched horizon, scenario, simulator/runtime identity, and metric definitions for paired comparisons, while allowing the intended policy/config difference. Store the requested horizon separately from actual survival time so legitimate early extinction remains comparable. Require a complete declared seed set before presenting a final benchmark mean; mark partial summaries explicitly.

### F07 — Orchard sweeps lose orchestration progress and overwrite prior versions

**Evidence:** [orchard/sweep.py](../oscar-orchard-research/survival/research/orchard/sweep.py), lines 47–53, submits all jobs and writes `summary.json` only after the entire pool completes. [society/harness.py](../oscar-orchard-research/survival/research/society/harness.py), lines 154–155, names results by policy, label and seed, excluding config digest and horizon, and writes them directly.

**Impact:** An interrupted sweep retains some individual result files, but the sweep does not resume from them. Repeating a label/seed with new kwargs or horizon overwrites that run's result. A worker exception interrupts aggregation. There is no cooperative episode checkpoint or wall-clock budget in this path. Duplicate seeds can target the same output filename.

**Recommendation:** Validate configs, finite horizons, worker counts, and unique seeds before starting. Write an immutable plan first; persist each finished job atomically and update an aggregate incrementally. Add unique run IDs and configuration digests. Resume compatible completed work, record failures explicitly, and define timeout/retry behavior. Reuse the main tuner's coordinator/evaluator patterns through a small common runner layer while keeping research scenarios distinct.

Apply numerical-library thread limits before imports here too. The orchard sweep sets SDL/Pygame variables but does not enforce the one-thread-per-process limits already used by the main tuner and guide batches.

### F08 — Orchard results omit important reproduction information

**Evidence:** The harness hashes `research/society/<module basename>` at [society/harness.py](../oscar-orchard-research/survival/research/society/harness.py), line 150. The default orchard module lives in `research/orchard/orchard.py`, so its per-run `policy_sha256` becomes `None`. Constructor kwargs are not included in the individual result; only the final sweep summary records the config dictionary. The result uses a hardcoded source commit and says “Local Linux” regardless of the executing platform or the no-predator wrapper.

The [CPU bundle allowlist](../oscar-orchard-research/survival/research/orchard/build_cpu_bundle.py), lines 6–11, omits `debugger/recorder.py`, although the bundled sweep exposes `--record` and the harness imports `ReplayRecorder` when enabled.

**Recommendation:** Hash the actual imported module path and its local dependency closure, including the harness and predator-disabling adapter. Put resolved kwargs, scenario modifications, simulator hash, package versions and measured platform into every receipt. Include the recorder and its required dependencies in recording-capable bundles, with an import smoke check. Treat lightweight replays and resumable checkpoints as separate artifacts: one supports inspection, the other execution recovery.

The trap-research reliability runner already contains a dependency-manifest implementation that can inform this work. Preserve archived evidence rather than rewriting old receipts as if they contained missing information.

### F09 — Guide execution does not always enforce the recorded protocol

**Evidence:** [guide_batch.py](../scripts/guide_batch.py), lines 53–65, forwards `--seconds` in ordinary mode but drops it in `--multi` mode. [guide_multi.py](../scripts/guide_multi.py) uses fixed 60-second delivery and 30-second final-hold stages. The batch manifest nevertheless records the supplied `seconds`. Ordinary mode also excludes `guide_lab.py`, `guide_lab_sites.py`, and the batch runner from its source hash list.

The batch runner selects the first matching `map-*/summary.json` or `multi-*/summary.json`. If a prior attempt left a summary but not `case-N/result.json`, a retry can choose that older summary. A nonzero subprocess exit does not automatically invalidate an existing successful summary.

**Recommendation:** Define scenario-specific protocol fields. Either wire a supported multi-mode duration through every stage or reject an inapplicable duration option. Include runner/setup sources in both modes. Assign an attempt ID/output directory before launching and accept only that attempt's receipt after checking the return code, seeds, final status and protocol digest. Keep the original attempt outcome when retrying infrastructure failures.

### F10 — Candidate-based scheduling leaves CPUs idle

**Evidence:** [tune_policy.py](../tune_policy.py), lines 237–249, limits active candidates to `workers // len(train_seeds)` and retains each candidate until all of its seeds finish.

**Impact:** Four workers with three seeds use only three workers initially. Even with 24 workers and three seeds, finished seeds create idle slots until their slowest sibling finishes and releases a candidate slot. Long surviving episodes can be slower than extinct episodes, making this imbalance relevant to the objective. This is a scheduling consequence, not a measured throughput loss percentage.

**Recommendation:** Maintain an explicit bounded queue of pending seed jobs and submit on available capacity, with a separately controlled candidate-in-flight limit. A ceiling-based candidate count is a small initial improvement; adaptive scheduling should still limit how many suggestions are made without feedback. Prefer cached completions before new work and report worker utilization, queue delay, episodes/hour, and completed configurations/hour.

Keep process isolation and the main tuner's single-thread numerical-library setup. More workers should be chosen from measured CPU/RAM limits, not assumed to provide linear speedup.

### F11 — Headless evaluation eagerly builds large diagnostic snapshots

**Evidence:** [expert_policy.py](../src/utils/controllers/expert_policy.py), lines 160 and 174, builds population and harvest snapshots every policy tick. [harvest.py](../src/utils/controllers/harvest.py), lines 812–828, serializes all fruit tracks and nested navigation/coverage state. [coverage.py](../src/utils/controllers/coverage.py), lines 412–428, materializes a dictionary for every survey point. The evaluator's coarse trace directly reads a small subset of state and samples it every 60 simulation seconds.

**Impact:** At 10 Hz, a full game can rebuild visualization-oriented collections about 30,000 times per seed. This consumes allocations and CPU even with `draw_overlay=False`. It is a concrete source of avoidable work, but its share of total runtime still needs measurement.

**Recommendation:** Construct detailed snapshots lazily when a viewer/report requests them or at a diagnostic cadence. Retain only the small decision-relevant state in the hot loop. First verify that no policy path depends on snapshot construction side effects, then check action/score equivalence on fixed observations and full episodes. Profile simulator step time, planner time, snapshot time, diagnostics, checkpoint serialization and I/O separately. Add p50/p95 or maximum decision latency and peak memory; the current mean controller time does not explain tail costs.

A second profiling candidate is the per-observer rebuilding of a fruit `cKDTree` in `HarvestCoordinator._observe`. Any index reuse must preserve visibility of tracks added by earlier observers; blindly sharing a stale tree would change matching behavior.

### F12 — Amortize startup and checkpoint work without changing seeded worlds

**Evidence:** [environment.py](../src/elements/environment.py), lines 38–43 and 71–78, generates/render-colors the entire biome surface during every world initialization. The default world has 1,920,000 pixels, and rendering consumes the **same RNG** used for later world generation. [tuning_evaluation.py](../tuning_evaluation.py), lines 126–150, checkpoints the simulator/policy graph with gzip pickle, retaining accumulated diagnostic trace state.

An initialization-only cProfile sample of native `SimulationCore(seed=1)` took 15.739 seconds, with 14.180 seconds (about 90%) attributed to biome rendering. It made 1,920,000 `Surface.set_at` calls. These numbers include profiler overhead and describe one startup on this machine; they do not establish unprofiled episode throughput or a projected speedup.

**Recommendation:** Measure initialization time, checkpoint bytes, serialization time and resident memory before changing worker counts or checkpoint intervals. Consider a versioned immutable initial-world cache per seed, copied into private episode state before any policy actions, to amortize repeated world construction. Include source/runtime/scenario identity and RNG state, keep cached originals untouched, and verify identical state and observation sequences after loading.

Do not simply replace rendering with a no-op: that changes the RNG position and therefore the map associated with a seed. `traps/experiments/verify_sites.py` currently does this for a geometry experiment; those fixtures should not be described as identical to native worlds at the same seed. `coverage_scan.py` also samples geometry directly, so its preset statistics should remain explicitly separate from native-game success rates.

Consider streaming append-only diagnostics separately from the minimal resumable state if checkpoint growth is material. Preserve exact events and valid replay output where required by the research subtree's instructions. A checkpoint is not a replay.

### F13 — Three selection seeds are insufficient evidence for small gains

**Evidence:** The main defaults use three training seeds and three validation seeds for a search allowing 300 completed trials. [validation_report()](../tune_policy.py), lines 308–347, ranks complete candidates by mean score and reports only the selected winner's mean paired improvement. The code correctly warns that validation seeds are used for selection and are not an unbiased test set.

**Recommendation:** Turn that warning into an explicit final evaluation stage on untouched seeds. Select the winner once, freeze it, and compare it with the baseline on the same final seed set. Add per-seed paired deltas, win/tie/loss counts, uncertainty for the mean paired difference, full-horizon survival counts and intervals, and worst-seed results. With only three seeds, label estimates as exploratory; a bootstrap alone cannot manufacture reliable evidence from such a small sample.

Choose additional sample sizes from observed between-seed variability and a predefined practically useful improvement. Keep the native game-score objective unless a different objective is deliberately specified. Report predator-free and native-predator scenarios separately; good harvesting performance alone does not establish complete-game robustness.

The guide reports already preserve all-map denominators and compute Wilson intervals; the frozen trap reliability code keeps missing cases in its denominator. Reuse these established reporting principles instead of building another incompatible convention.

### F14 — Existing integration tests need to become a release gate

**Evidence:** The current suite already catches F01/F02, so the immediate problem is running/enforcing the checks. `test_tuning.py` skips its optimizer tests when Optuna is absent. Root [pyproject.toml](../../pyproject.toml) specifies Python >=3.13 and a different NumPy/SciPy/Pydantic stack from the pinned Python >=3.11 tuning requirements. No tracked GitHub Actions workflow was found in this checkout. The scoped pytest configuration also leaves the separate `scripts/test_*.py`, trap tests, and research suites outside the default survival test collection.

**Recommendation:** Define one supported tuning installation entry point and test it on the intended Windows and Linux Python environments. Keep the broader project environment separate if necessary, but document the distinction. A tuning CI job should require Optuna rather than succeeding with skipped optimizer coverage. Run the current 524-test suite and explicitly selected script/geometry suites appropriate to a change.

Add a small spawned-process integration job after F01/F02 are repaired: baseline search, complete validation, cooperative stop, resume, and exported-config reload. Existing `ImmediatePool` tests are useful for coordinator logic but cannot establish spawn/import/signal behavior. Include one real event-producing replay-equivalence test, and the incompatible-cache checks from F03/F06. Test the CPU bundle's advertised recording path without launching remote compute.

### F15 — Harden error recovery and publication after the main defects are fixed

**Evidence and recommendations:**

- `open_study()` and interrupted search completion mark a trial `FAIL` before enqueueing its retry. A crash between those journal operations can leave the configuration unretried; completed seed files survive, but automatic recovery only repairs `RUNNING` trials. Record interruption intent durably and reconcile interrupted trials with queued/completed retries on startup.
- Search errors save only `repr(exc)` to one overwritten `worker_error.json`; validation errors do not create the same structured receipt. Persist phase, trial/candidate, seed, full traceback and attempt ID. Distinguish an invalid candidate from a broken process pool, disk error, or simulator contract failure.
- `configured()` runs outside the future-result error handler. Custom baseline mechanics can make some static search bounds invalid and abort search during candidate construction. Validate the space against the chosen schema/config before workers start, and define candidate-rejection semantics explicitly.
- JSON files in the main tuner are atomically replaced, while Markdown reports are written directly. Publish reports atomically as well. Keep an immutable final selection receipt with candidate/config digests and completion status, separate from changing best-so-far exports.
- Resuming intentionally reseeds the TPE sampler using `optimizer_seed + len(study.trials)`, and asynchronous completion order influences suggestions. The documentation already states that exact search replay is not guaranteed. For scientific comparisons, log suggestion/completion ordering and use a deterministic batch option or validated sampler-state persistence only if exact replay is required.
- The orchard harness advances to its horizon using floating-point time comparisons; use a shared integer-tick termination rule like the main evaluator. Validate finite horizons and make overshoot/terminal-action semantics explicit across runners.

## Suggested implementation sequence

1. **Restore trustworthy execution:** F01/F02, then require the current suite to pass. Add one spawned-worker smoke run and a diagnostic conservation check.
2. **Protect experiment evidence:** F03/F06/F07/F08/F09. Introduce a shared immutable experiment identity and atomic episode receipts; preserve existing archived data.
3. **Improve the scientific search:** F04/F05/F13. Freeze a full-horizon protocol, verify parameter activity, and reserve an untouched final seed set before searching.
4. **Improve throughput:** F11, then F10, then F12 based on profiling. Compare action/score equality and completed full-horizon evaluations per wall-clock hour, including checkpoint overhead.
5. **Finish operational hardening:** F14/F15, bundle checks, resume fault injection, and consistent final-result publication.

The strongest immediate acceptance criterion is a complete baseline/search/validation/resume cycle with accurate events, an immutable study identity, and identical resumed versus uninterrupted episode results. After that, judge optimizer changes by reproducible held-out score and survival improvements per compute budget.
