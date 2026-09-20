# Survival simulator

**Final evaluated Survival Simulator submission:** [code explanation and build instructions](fast1m-endpoint/README.md). The submitted code is self-contained in `survival-simulator/fast1m-endpoint/`; the material below describes earlier work.

For a new coding agent, start with the [current handover](docs/agent_handover.md):
working branch, policy integration, untested search tooling, execution constraints,
and the proposed Codex improvement loop.

Two supported policies share the same native simulator:

| Mode | What it runs | Predators | Display |
| --- | --- | --- | --- |
| `trapping` | Shared-map exploration, Orchard gathering/reproduction, predator guides and replacement bait | Natural spawns | Browser replay, updated while the run records |
| `orchard` | Orchard survival through the audited observation-only worker | Disabled | Headless score or native window with speed buttons |

The current integration comes from `survival-simulator/lucas-experimental`
at `803bdd57`, including ordinary-agent avoidance, updated guide handoff and
recovery from blocked navigation starts. Corner-pocket sites are optional.
The branch integration and optimizer have **not been run or tested locally**.
See [the branch review](docs/policy_branch_integration.md) for sources and changes.

## Run on Windows

From the repository root in PowerShell, use `survival-simulator/run.cmd`.
It selects the existing simulator virtual environment and changes into the
right folder automatically. The Python equivalent, from this directory, is
`python run.py <command> ...`. Use `<command> --help` for options.

### Trapping with rendering

Start one simulation:

```powershell
.\survival-simulator\run.cmd trapping --seed 0 --seconds 3000 --out logs/trapping-live
```

After the first 50-second progress update, open a second terminal at the
repository root:

```powershell
.\survival-simulator\run.cmd view logs/trapping-live --port 9059
```

Open http://localhost:9059 and press **Play**. The viewer shows actual recorded
frames, including food, trees, obstacles, agents, and predators. Cyan rings
mark guides, yellow marks bait, and orange marks replacement bait. It supports
playback speed, stepping, agent inspection, and jumping to role events.

The browser can only display frames already recorded. Closing the browser
leaves the simulation running; Ctrl+C in its terminal stops the simulation.
Use a new output directory for another run. Final results are in `summary.json`.

### No-predator survival

Headless, with progress and a final score:

```powershell
.\survival-simulator\run.cmd orchard --seed 1 --seconds 3000 --output logs/orchard-run
```

Or show the native simulator with 1x, 2x, 5x, 10x and 20x buttons:

```powershell
.\survival-simulator\run.cmd orchard --seed 1 --seconds 3000 --output logs/orchard-view --render --speed 5
```

Keys 1-5 select the five speeds. The timestep stays at 0.1 seconds; high speed
is limited by available compute. Closing the window or pressing Ctrl+C saves
a partial result with its stopping reason. Completed output files are never
overwritten. `seed-1.json` contains the score, duration, population and
observation-boundary audit. These runs do not resume from progress files.

Both runners accept an independent `--policy-seed` (default 0). Orchard also
accepts optional `--policy-kwargs` JSON. Both modes use `models/survival/oscar_orchard.py`;
the trapping coordinator changes some reproduction settings for trap roles.
The no-predator worker adds inferred bounds and input isolation; see
[the observation audit](docs/orchard_observation_audit.md).

## What matters in this folder

| Path | Responsibility |
| --- | --- |
| `run.py`, `run.cmd` | Supported command launcher |
| `models/core.py` | Current coordinator connecting exploration, survival and trapping |
| `models/survival/` | Shared Orchard gathering/population policy |
| `models/exploration/` | Shared map, exploration and navigation |
| `models/entrapment/` | Trap detection, guide tracking/steering and ordinary-agent predator avoidance |
| `models/observation_only.py`, `observed_bounds.py` | Audited no-predator Orchard adapter |
| `src/` | Native simulator; its source remains pinned |
| `scripts/trapping_game.py` | Current trapping recording runner |
| `scripts/entrapment_viewer*`, `entrapment_benchmark*` | Replay and frozen-reference benchmark tools |
| `scripts/run_orchard.py`, `playback.py` | No-predator runner and native rendering controls |
| `models/experiment_*.py`, `experimental_policy.py` | Optional tuning configuration, observation worker and experimental behaviors |
| `scripts/optimize_policy.py`, `experiment_case.py` | Optional predator-only search and isolated evaluation |
| `scripts/research_loop.py`, `research_study.py`, `research_bo.py` | Prepared major/subgeneration supervisor, focused search and broader BO |
| `scripts/runpod_night.*`, `runpod_setup.sh`, `search_resources.py` | Budgeted CPU Pod launch, environment setup and worker sizing |
| `tests/` | Current integration, observation boundary and frozen-reference contracts |
| `docs/` | Current mechanics, input contract, source manifest and reference evidence |
| `logs/` | Local recordings and results; ignored by Git |

The old flat policy modules and `models/nikolaj/` are immutable copies used
only by the historical 9059 benchmark. Current runners and tuning use the
three module directories above. [The module guide](models/README.md) identifies
the reference files. `src/utils/controllers/dummy_agent_policy.py` is the
unchanged engine reference; supported commands do not select it.

## Benchmark and API

Optional policy experiments and resumable tuning are available through
`run.cmd tune`. The new experimental code has **not been executed or tested**;
read [the optimization guide](docs/policy_optimization.md) before running it.
It compares optional safety/population additions and exposed hyperparameters
using natural-predator games only,
without changing the default trapping or Orchard policy.
For the prepared $25 overnight CPU Pod configuration and Linux launch commands,
see [the Runpod search guide](docs/runpod_policy_search.md). Nothing has been
launched or provisioned; the scripts are untested.

The [research supervisor](docs/research_supervisor.md) adds up to four
subgenerations per major generation, LLM policy reviews, Lucas-branch snapshots,
paired comparisons across immutable code versions, focused tuning, broader BO,
budget accounting and a final unopened holdout. It is implemented but **has not
been executed or tested**. `run.cmd research prepare --out <new-directory>`
prepares a campaign; `research run` is a separate, explicit execution step.
The template requires an actual hourly compute rate before execution. No
supervisor, experiment, Git fetch or cloud job was launched during preparation.

For the **historical 9059 policy** benchmark, with fixed controller seed,
source checking and completed-case resume:

```powershell
.\survival-simulator\run.cmd benchmark --out logs/trapping-benchmark --count 1 --seed-start 0 --workers 1 --seconds 3000
```

Use `benchmark-view` for downloaded fleet recordings and `benchmark-merge`
for shard validation. Their output layout and limits are documented in
[the protocol](docs/entrapment_9059_benchmark_protocol.md) and
[the benchmark viewer guide](docs/entrapment_benchmark_viewer.md).
The ordinary `view` command expects the full recording from `trapping`.
Use `trapping` or `tune` to evaluate the current policy; `benchmark` deliberately
keeps the original source and score reference.

`run.cmd serve` starts the observation-driven trapping `/predict` endpoint
on port 9052. `simulation_server.py` is the optional local HTTP test client;
it does not contact competition services.

## Setup and tests

Use Python 3.11 or newer. From `survival-simulator`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The same commands work on Linux using `.venv/bin/python`.

## Evidence and limits

The original trapping baseline's recorded seed-0 run held up to 11 predators
near bait for 30 seconds and went extinct at 1,073.6 seconds. Proximity is not
proof of permanent capture. See [the baseline](docs/entrapment_9059_baseline.md).
Historical Orchard scores include 2,834 and 2,727, but survival to 3,000 seconds
is not reliable; the three saved reference measurements are in
[orchard_reference_results.json](docs/orchard_reference_results.json).
These are historical measurements, not new scores from the cleanup.

The original 39 frozen baseline files remain separate from the current modules. See
[cleanup notes](docs/cleanup.md) for what was removed and where older work can
be recovered. Other challenge folders are independent of this simulator.
