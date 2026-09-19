# Runpod predator-only overnight search

This document describes the older standalone evolutionary search launcher.
The current task uses the hierarchical research supervisor instead: follow
[Runpod research launch](runpod_research_launch.md). Local tests and a short
diagnostic pilot have passed; no cloud campaign is running yet.

## Hardware and $25 budget

Recommended starting configuration: **one CPU5 Compute-Optimized (`cpu5c`)
on-demand CPU Pod, 32 vCPUs, 64 GB RAM, no GPU**, with a 20 GB persistent
network volume mounted at `/workspace` and a 10 GB container disk. CPU Pods do
not support persistent Pod mounts (confirmed by the live API); use a network
volume. Use a Linux x86-64
CPU template with Bash, Python 3 and its `venv` support. A GPU/PyTorch image
is unnecessary: this simulator and policy use CPU Python, NumPy/SciPy and
geometry operations, with no CUDA implementation.

Runpod's live `list_cpu_types` and `get_cpu_type` catalog queries on
2026-09-18 returned these Pod quotes. All six families accepted a 32-vCPU
request and reported top-level availability `HIGH`; deployment availability
is not reserved or guaranteed by a catalog response.

| Pod family | vCPUs | RAM | Whole-Pod compute/hour | Nine hours |
| --- | ---: | ---: | ---: | ---: |
| CPU3 Compute-Optimized (`cpu3c`) | 32 | 64 GB | $0.96 | $8.64 |
| **CPU5 Compute-Optimized (`cpu5c`)** | **32** | **64 GB** | **$1.12** | **$10.08** |
| CPU3 General Purpose (`cpu3g`) | 32 | 128 GB | $1.28 | $11.52 |
| CPU5 General Purpose (`cpu5g`) | 32 | 128 GB | $1.472 | $13.248 |
| CPU3 Memory-Optimized (`cpu3m`) | 32 | 256 GB | $1.76 | $15.84 |
| CPU5 Memory-Optimized (`cpu5m`) | 32 | 256 GB | $2.08 | $18.72 |

The CPU5 choice is a proposed starting point, not a benchmark claim that its
throughput per dollar beats CPU3. CPU3 Compute-Optimized is a cheaper fallback.
The catalog does not promise a specific processor model for these families;
`resources.json` records the actual model and allocation when launched. Extra
RAM has no demonstrated benefit yet. One Pod also avoids multi-server database
and coordination overhead. There is no reason to spend the entire $25 before
measuring throughput.

Runpod charges compute and storage separately. Check the console's final
quote before deployment. The current documentation lists a standard network
volume below 1 TB at $0.07/GB/month and container disk at $0.10/GB/month.
The recommended nine-hour compute estimate leaves substantial room for setup,
storage and tax within the approximate budget.
[Runpod pricing](https://docs.runpod.io/pods/pricing).

The launcher defaults to a $25 allowance, a $5 reserve, $1.12/hour compute and
**nine cumulative active hours across resumes**. It saves estimated spending
every 15 seconds. It limits search runtime, not your cloud account's bill:
setup, idle time, other Pods, tax, storage and downtime between resumes are
not measured. A crash can lose the last 15 seconds of accounting. Enter
additional known external spending through `--already-spent` when needed.

**Finishing Python does not stop the Pod. Stop it in Runpod when finished.**
Keep results on the persistent volume and download them first. Retained
storage remains billable after stopping; the launcher never deletes anything.
[Pod lifecycle and storage](https://docs.runpod.io/pods/manage-pods).

## Search design

Block evolutionary search handles the many numeric settings, conditional
features and interactions without fitting a large, poorly informed Bayesian
model at the start. This is an engineering choice, not a measured comparison
against Bayesian optimization.

| Stage | Default work | Approximate nine-hour allocation |
| --- | --- | ---: |
| Screening | 25 controls/feature combinations; one probe per tunable scalar; then up to 1,024 block mutations/crossovers, on seeds 0-3 | 5.4 hours |
| Finalist comparison | Best six distinct configurations plus both controls, on seeds 0-11; reuse seeds 0-3 | 1.8 hours |
| Validation | Freeze the selected candidate; compare with the current integrated baseline on seeds 1001-1016 | Remaining 1.8 hours |

Those are time allocations and candidate limits, **not promises that all work
will finish overnight**. Throughput has not been measured. `coverage.json`
identifies untouched parameters. Resume can finish interrupted evaluations;
after validation has begun, further tuning requires a new study so those
held-out maps cannot silently become training data.

Every scored game runs until extinction or **3,000 simulated seconds with
natural predators**. No early survival cutoff changes the objective. Survival
is ranked first, then simulator score, then fewer optional behaviors. A crash
or timeout makes that candidate ineligible. Both baseline/control comparisons
must work before the search expands.

The optional features are consistent escape thresholds, short escape memory,
shared predator/lure avoidance, trap exclusion, collision-aware escape steering,
risk-aware food assignment, guide/bait role limits, guide geometry refresh,
emergency reproduction, late energy conservation and optional corner pockets. Existing policy settings
and new knobs are inventoried automatically; immutable simulator mechanics
and dormant settings have explicit exclusion reasons. See the complete
[experiment rationale and parameter policy](policy_optimization.md).

The baseline now uses `models/core.py` and the modular integration from
`survival-simulator/lucas-experimental` (through `803bdd57`). Its local bystander
avoidance, updated guide handoff and navigation recovery are included before testing additions.
Avoidance settings are also tunable. The historical 9059 benchmark is a
separate reference, not the control for this search.

The scheduler runs independent **candidates and maps** through one shared
pool. CPU sizing respects process affinity, container CPU quotas and memory
limits. It reserves two CPU slots on larger machines and four GiB RAM, with a
planning allowance of two GiB per simultaneous game. Expect approximately
27-30 game workers on the proposed Pod, depending on available memory.
Each game also has an isolated policy process; these alternate engine and
policy work, rather than each requiring a continuously busy core. The memory
allowance is unmeasured; reduce `SEARCH_WORKERS` if needed.

BLAS/OpenMP threads are capped at one per process; there is no renderer or
frame recording. Progress files update every ten wall seconds and the
coordinator reports every thirty. Identical cases are deduplicated, including
within a parallel batch. Linux cancellation/timeout terminates the evaluator's
process group so its policy process does not keep using CPU.

The policy receives only sanitized public observations, simulation time and
its settings. World seeds and evaluator scores stay outside the policy process.
The optimizer does not rewrite production policy or simulator source. Existing
wall-clock-adaptive map fitting can still produce variation under different
CPU loads; common seeds do not guarantee identical trajectories. Held-out
results include paired differences and descriptive bootstrap intervals.

## Prepare and launch later

First provision the above Pod yourself and place this checkout under
`/workspace/NordicCupAI`. These new files must be included when copying the
checkout; cloning an older remote revision will not include unpushed changes.
Use persistent `/workspace`, not the temporary container filesystem.

From the Pod terminal, setup only:

```bash
cd /workspace/NordicCupAI/survival-simulator
bash scripts/runpod_setup.sh
```

Setup creates a Python 3.12 environment at
`/workspace/predator-search-venv`, installs the pinned repository requirements
using `uv`, and starts no games. If the image lacks `python3-venv`, install
that OS package first. Reuse the environment on resume; installing different
dependencies causes the provenance check to reject the old study.

Inspect the inventory without games, when ready:

```bash
bash scripts/runpod_night.sh --describe
```

Launch the overnight search detached from the SSH connection:

```bash
setsid bash scripts/runpod_night.sh > /workspace/predator-search-console.log 2>&1 < /dev/null &
echo $! > /workspace/predator-search-launcher.pid
```

This is an ordinary Linux launch, not Slurm. The `.sh` cannot choose hardware
on an already-running Pod; the requested shape is specified above and in the
script header. It never provisions paid resources itself.

Monitor and stop gracefully:

```bash
tail -f /workspace/predator-search-console.log
```

```bash
kill -INT "$(cat /workspace/predator-search-launcher.pid)"
```

Run the same launch command to resume after the previous process has exited.
The OS lock prevents two coordinators writing the study. Completed games are
cached; interrupted games restart at their seed. Completed stages are skipped.
The nine-hour cumulative runtime and spending ledger are retained. If the
allowance is exhausted, increasing `SEARCH_MAX_HOURS` explicitly permits
more runtime, still within the estimated spending allowance. Never change
the source, dependency versions or protocol halfway through a study.

For a CPU3 fallback or reduced concurrency, set the actual Pod quote explicitly:

```bash
POD_HOURLY_USD=0.96 SEARCH_WORKERS=24 bash scripts/runpod_night.sh
```

Other shell settings: `TASK_OUT`, `TASK_ENV`, `TASK_WORKSPACE`,
`SEARCH_BUDGET_USD`, `SEARCH_RESERVE_USD` and `SEARCH_MAX_HOURS`.
An explicit worker count overrides auto sizing. The underlying optimizer also
accepts `--profile runpod --phase search|refine|validate|overnight`, seed lists,
`--finalists`, `--batch-candidates` and `--evolution-trials`; its `--hours` is a
per-session limit without the wrapper's cumulative spending ledger.

## Read the results

Look in `/workspace/predator-search/`:

- `validation-report.md`: the final held-out survival/score comparison, if complete.
- `validation.json`: paired per-map results and bootstrap intervals.
- `best.json`: selected configuration, baseline flag and selection basis.
- `report.md`: the four-map screening leaderboard, not final validation.
- `refinement.json`: whether the twelve-map finalist comparison finished.
- `feature-effects.json`: individual and leave-one-out feature comparisons.
- `coverage.json`, `parameter-inventory.json`: evaluated changes and remaining settings.
- `resources.json`, `progress.json`, `budget.json`: hardware, progress and cost estimate.
- `cases/`: saved individual results, boundary audits, progress and error logs.

If the night ends partway through finalist comparison, selection falls back to
the screening winner and says so explicitly. It never selects a candidate
because only its easiest maps finished. If validation cannot finish, its
completed games are saved for resumption and no complete comparison is claimed.
No result automatically replaces the live trapping policy.
