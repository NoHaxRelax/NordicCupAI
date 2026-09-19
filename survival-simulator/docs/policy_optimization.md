# Policy experiments and resumable optimization

Implementation status: **written and reviewed by reading the source only**.
No simulations, tests, imports, or optimizer runs were executed for this change,
as requested. These are hypotheses to test, not measured improvements. The
current control comes from `models/core.py`, integrated from
`survival-simulator/lucas-experimental` through `803bdd57`. The original 9059 source
files remain unchanged for the separate historical benchmark; they are no
longer the tuning baseline. See [the branch integration](policy_branch_integration.md).
The optimizer evaluates **only natural-predator games**, in both training and
validation. It never disables spawning or forces a predator encounter.

For code-changing major/subgeneration research, see the separate
[research supervisor](research_supervisor.md). That prepared outer loop uses
focused mutation within subgenerations and a bounded Gaussian-process BO search
at major boundaries; this `tune` command remains evolutionary. Neither the
supervisor nor its new contract tests have been executed.

## What appears to be missing

| Optional feature | Reason to test it | Tradeoff |
| --- | --- | --- |
| `consistent_escape` | Add one configurable radial escape trigger alongside the current directional bystander avoidance, using the same radius in the explorer escape controller. | Earlier escape can consume energy or abandon food. |
| `escape_memory` | Keep the escape controller active after a sighting disappears, instead of immediately returning to Orchard. | Excessive memory can waste travel. |
| `shared_danger` | Use recent observed predator tracks and the segment between predator and guide as avoidance areas. | Uncertain or stale estimates can exclude useful space. |
| `trap_exclusion` | Keep ordinary agents away from the entrance/goal of an established baited trap. | Food near a trap becomes harder to reach. |
| `safe_steering` | Compare alternative moves against multiple threats and observed walls, including when fleeing. | More decision compute; finite candidate directions can still get stuck. |
| `risk_aware_food` | Release dangerous fruit claims and tree posts, and avoid assigning new unsafe targets. | An overly conservative radius may cause starvation. |
| `role_budget` | Reserve ordinary workers, limit guide recruitment, and release guides that lose contact or take too long. Also limit new bait recruitment when workers are scarce. | Fewer guides/bait can reduce trapping reliability. |
| `refresh_guide_geometry` | Existing guide paths snapshot the map at assignment. Refresh from newly observed geometry and invalidate navigation/following caches when it changes. | Rebuilding paths costs compute and temporarily loses tracking history. |
| `emergency_reproduction` | Rescue a shortage of young agents using an eligible parent with enough energy, including older parents. Bound requests by deficit and a per-tick limit. | Rescue births can consume the last available food budget. |
| `late_conservation` | Gradually reduce idle turning as time passes. Existing Orchard settings also govern late travel and population decline. | Less turning can delay discovery of food or predators. |
| `corner_pockets` | Add the upstream contact-excluding short corner pockets to ordinary gap candidates using observed geometry. | Additional geometry compute; estimated rectangles and crowd access can still be inaccurate. Disabled in the baseline. |

Shared positions are accepted only within the same map group/frame and below
an uncertainty limit. Tracks expire; guides and bait are exempt from ordinary
avoidance. The two map implementations are connected through an agent's local
coordinate frame, never by assuming their global coordinates are identical.
These checks are correctness constraints, not optional things to optimize away.

The updated control already includes local bystander avoidance around observed
predators and the bait area, plus observed-target proximity before guide handoff.
All optional additions off preserves that behavior. The `bystander.*` parameter
block covers its enable switch, contact/hearing/vision/trap margins and steering
resolution. New shared-danger and food filters remain distinct experiments.

The new logic does not certify permanent trapping, infer hidden predator IDs,
or provide perfect future trajectories. Escape steering uses observed positions
and a movement lookahead, not access to future simulator states. Guide release
does not assume an observed delivery proves that a predator is permanently held.

## Search strategy

This uses **block evolutionary search**, with explicit feature comparisons
before numeric tuning. It has no additional package dependency.
For many-core Runpod hardware, use [the overnight profile](runpod_policy_search.md).

1. Evaluate the current integrated baseline and its configurable version with all additions off.
2. Enable each addition separately, then enable all, then remove each from that combination.
3. Probe every tunable scalar once, using the best completed training candidate
   as the parent. Enable its owning optional block when necessary. These probes
   help cover the search space; they are not independent causal estimates.
4. Mutate small groups of related parameters, selecting parents from the best
   five completed candidates. Occasionally toggle an addition or copy a related
   parameter block from another successful candidate.

The laptop plan includes 25 feature/control comparisons, all parameter probes,
and 64 evolutionary candidates. The Runpod profile allows 1,024 evolutionary
candidates after the probes. This can take substantially longer than one
night. `--hours` limits each session and `--trials` can cap the total number of
candidates. An unfinished search is still useful: completed comparisons and
the current best candidate are saved. `coverage.json` shows settings not yet
varied in a successfully evaluated configuration. A setting may have no effect
on a particular game if the relevant situation never occurs.

CPU workers share a queue across multiple candidates and seeds, rather than
waiting on one candidate at a time. Each complete candidate is saved immediately.
Parameter probes cycle through related blocks so every subsystem is reached
early. Completed controls gate the feature comparisons; completed feature
comparisons gate parameter mutations. New batches use completed parent results.
`--workers 0` detects CPU affinity, Linux container quotas and available memory.
Explicit `--workers 1` remains the conservative laptop choice.

The Runpod `--phase overnight` flow reserves roughly 60% of the time for screening,
20% for a shortlist on more training maps, and 20% for held-out validation. Early
completion carries unused time into later phases. Candidate selection only uses
a complete common set of training maps. If shortlist comparison cannot finish,
the screening winner is retained and the report labels that limitation. A known
failed winner falls back to the current baseline. No partial game's score counts.

Within each stage, candidates use the same training seeds and independent policy RNG
seed. Every case runs the integrated trapping policy with natural predator
spawning enabled. Completed identical configurations/maps are cached.
Every scored game runs until extinction or the full requested horizon; short
survival cutoffs are not used to select a policy intended for 3,000 seconds.

The primary ranking is `mean survival / horizon + 0.25 * worst survival / horizon`.
Mean simulator score breaks ties, followed by fewer enabled additions. This
prioritizes living and reduces the attraction of candidates that excel on one
map but die immediately on another. Per-mode scores and survival are also saved.
An error makes the whole candidate ineligible, rather than dropping the bad map.
Failure of either control stops the search so a broken implementation does not
waste the rest of the budget. No candidate is automatically made the live policy.

## Parameter coverage

`parameter-inventory.json` is generated from the current sources when launched:

- Every explicit Orchard constructor parameter used by trapping. Trapping
  retains its own `extra_old=False` and disabled
  age-based heir defaults in the control configuration.
- Every scalar in the validated explorer/planner schemas, including defaults
  omitted from the JSON files: population, exploration, movement, crowding,
  map estimation, biome refit schedules and compute limits.
- Bait navigator settings and the named guide/pathfinding/following heuristics.
- Current bystander avoidance settings, including disabling it for comparison.
- Every new feature switch and safety/population/conservation parameter.

The inventory also lists exclusions with reasons: engine physics, the disabled
old harvesting coordinator, required mapping setup, rendering-only settings,
and Orchard constructor arguments stored but never read. Those values stay
fixed. Simulator rules, world-generation parameters, world seed, absolute world
dimensions and hidden state are never search variables. Unnamed numeric
literals remaining in production source are not automatically rewritten.

Default numeric bounds are conservative multiplicative neighborhoods, with
explicit ranges for safety distances, population limits and disabled-time
sentinels. Schema constraints and coupled bounds are checked before evaluation.
`null` represents disabled infinite-age/time Orchard thresholds in saved JSON.
Search ranges live in `models/experiment_config.py`; changing them requires a
new study directory because existing comparisons used a different search space.

## Commands (not executed during implementation)

From the repository root in PowerShell, inspect the generated inventory without
starting games:

```powershell
.\survival-simulator\run.cmd tune --out logs/policy-search --describe
```

Start an eight-hour session with one game at a time:

```powershell
.\survival-simulator\run.cmd tune --out logs/policy-search --hours 8 --workers 1
```

Repeat **the same command** to resume. Each game also uses a separate policy
process, so `--workers 1` is the conservative laptop setting. This workload uses
CPU; a GPU is not used. There is no fixed claim about how many trials eight
hours will complete. Progress prints the candidate, map, simulated time, living
population, and completed score.

For a small functional check when you are ready, use a separate directory:

```powershell
.\survival-simulator\run.cmd tune --out logs/policy-search-smoke --seconds 20 --train-seeds 0 --trials 2 --hours 1
```

That check exercises the baseline/control wiring, not long-term survival or
the new behaviors; natural predators may not spawn within 20 seconds. The
main default horizon is 3,000 seconds. Run feature
comparisons at that horizon before trusting improvements.

After training, compare its best candidate against the current baseline on
held-out seeds 101, 102, and 103:

```powershell
.\survival-simulator\run.cmd tune --out logs/policy-search --phase validate --hours 8 --workers 1
```

Validation freezes the selected candidate in `validation-selection.json`
before looking at those results. Repeating validation resumes the same
comparison even if training later finds another candidate. Do not select or
tune repeatedly against these held-out results; a few maps provide limited
evidence. Train/validation seeds must be disjoint. If overriding seeds, horizon,
or RNG settings, repeat those settings on resume/validation.

## Stopping and recovery

Ctrl+C requests a cooperative stop and closes policy workers. The wall-time
budget does the same. Completed games are reused, including results written
just before the coordinator crashed. Interrupted games restart from their
seed; this is completed-game resume, not a mid-simulation snapshot. If the
coordinator disappears, evaluator heartbeat checks request shutdown after
90 seconds (plus any in-flight policy decision/timeout). Each attempt has a
separate directory so partial attempts cannot overwrite a completed game.

A study lock prevents two coordinators writing the same study concurrently.
Source hashes, dependency versions, seeds, horizon and parameter space are
recorded and checked on resume. Source or dependency changes require a new
directory. Games have a configurable one-hour wall-time limit by default;
timeouts are errors, never successful survival. Logs retain the cause of errors.
Existing failed controls require fixing the cause and starting a new directory.

`PYTHONHASHSEED=0` and one BLAS thread per worker reduce avoidable variation.
The existing map estimator has wall-clock-adaptive behavior, so source/seed
matching does not promise identical trajectories across machines or loads.
The Runpod wrapper additionally records cumulative active runtime and estimated
compute spending across launches. The general `--hours` option is a per-session
limit; the wrapper's `--max-hours` is a total across its launches.

## Outputs

| Artifact | Contents |
| --- | --- |
| `report.md` | Training ranking and baseline differences |
| `feature-effects.json` | Each addition alone versus control, and its contribution within the all-additions combination |
| `best.json` | Selected training configuration, with baseline flag and refinement selection basis when applicable |
| `study.json` | Proposed/completed candidates, parentage and changed settings |
| `parameter-inventory.json`, `default-config.json` | Full parameter catalog and control settings |
| `coverage.json` | Successfully evaluated changes and settings still unvaried |
| `protocol.json` | Search definition, source hashes and environment |
| `cases/` | Cached results, progress, attempt requests and console logs |
| `validation.json` | Held-out results and paired deltas for each seed/mode |
| `refinement-plan.json`, `refinement.json` | Frozen shortlist, additional training maps and final training selection |
| `validation-report.md` | Held-out comparison and paired bootstrap intervals |
| `resources.json`, `progress.json` | Detected CPU/memory limits and current batch progress |
| `budget.json` | Runpod wrapper's cumulative runtime/cost estimate; not a Runpod invoice |

Policies run in fresh spawned processes and receive sanitized public agent
observations, time and policy configuration over JSON. They do not receive the
world seed, simulator object, actual positions, true fruit ages or evaluator
scores. Trapping retains the baseline's documented model assumptions.
This is an accidental-leak boundary, not an OS
sandbox against malicious policy code.
