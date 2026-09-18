# Overnight policy tuning on a laptop

From the **NordicCupAI repository root**, run this in PowerShell:

```powershell
.\survival-simulator\.venv\Scripts\python.exe -u .\survival-simulator\tune_policy.py --laptop --hours 8
```

The existing laptop environment already has the tuning dependencies. On a new
machine, install `survival-simulator/requirements-tuning.txt` into a Python 3.11+
virtual environment first. No GPU or display is needed.

## Defaults

- Uses half the detected logical CPUs, capped at **6 workers** (6 on this laptop).
  Each worker uses one numerical-library thread. Override with `--workers 2`
  if you want more capacity available for other programs.
- Searches for up to **6 hours**, then uses the remaining budget to validate
  the best **two distinct candidates plus the baseline**. Search can finish
  early if it reaches 300 completed trials. `--hours 10` gives 7.5 hours to
  search and 2.5 hours to validation, unless `--search-hours` is also supplied.
- Uses 16 startup trials, three training seeds at 600 simulation seconds, and
  three separate validation seeds at 3,000 seconds. The objective remains mean
  game score with predators disabled. Founder-relative breeding selection and
  the existing shared-map policy remain active.
- Saves episode checkpoints every **60 wall-clock seconds**, and on Ctrl+C or
  the time limit. Change the interval with `--checkpoint-seconds 30`.
- On Windows, temporarily prevents **idle sleep** while running, and lets the
  display turn off. Plug the laptop in and keep its lid open: this does not
  override a deliberate sleep action or closing the lid. Use `--no-keep-awake`
  to disable this behavior. Normal idle-sleep behavior returns when the job
  exits. See [Microsoft's API documentation](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate).

For **survival tuning**, use `--train-seconds 3000` and a fresh output folder,
for example `--output "$env:LOCALAPPDATA\NordicCupAI\bo-survival"`. A 600-second
episode does not measure survival through late food scarcity. Full episodes
cost more compute, so expect fewer trials. Changed policy code also requires
a new study; existing BO results and checkpoints remain available separately.

## Stop and resume

Press **Ctrl+C once**, then wait for the "Saved results" message. Run the
**same command** to resume. Each invocation gets a fresh time budget.

Completed trials and seed results are reused. In-progress episodes restore the
simulation, random-generator state, agents, shared map, policy memory, actions,
and accumulated metrics. Partial episodes never enter the optimizer as scores.
If a process is forcibly closed, its latest completed checkpoint is used;
normally at most roughly one checkpoint interval of work per active episode is
lost (plus any ongoing simulation step or checkpoint write).

If stopped during validation, the next invocation finishes that same set of
finalists and exits. A subsequent invocation continues the search. The 300-trial
limit includes previous completed trials; add `--trials 600` to extend a study
that has already reached the limit. Use `--search-hours 0` to go directly from
the completed search results to validation.

Keep the same code, dependencies, policy configs, seeds, episode lengths and
startup/optimizer seed settings for a study. The manifest rejects incompatible
resumes rather than mixing results. Worker count, checkpoint interval, wall-clock
budget and trial limit can change. Only one coordinator may use an output folder
at a time. Checkpoints use Python pickle and should only be loaded from your own
study folders.

## Results are useful before the run finishes

The Windows default is **`%LOCALAPPDATA%\NordicCupAI\bo-laptop`**, outside
OneDrive so frequent checkpoint writes do not need syncing. The tuner prints
the full path at startup. On other systems the default is `runs/bo-laptop`
inside `survival-simulator`. `--output PATH` overrides it and selects which
study to start or resume.

Open the results folder:

```powershell
explorer "$env:LOCALAPPDATA\NordicCupAI\bo-laptop"
```

- **`search_report.md`**: ranked completed training trials, updated after each
  trial. The first finished trial is already usable.
- **`best_search_expert_policy.json`**, **`best_search_global_planner.json`**:
  the best complete training result so far. These remain provisional until
  validated over the longer horizon.
- **`report.md`**, **`validation_results.json`**: validation progress, baseline
  comparison and winner. Only candidates with every validation seed completed
  can win, and the baseline must also be complete.
- **`best_expert_policy.json`**, **`best_global_planner.json`**: the best settings
  with a complete validation comparison so far. The baseline can win. A partial
  comparison is marked provisional in the report.
- **`candidates/`**: completed seed results and in-progress episode checkpoints.
  Each `seed-N.progress.json` lists the last saved simulation time, population
  and active runtime. `optuna.journal`, `manifest.json` and `run_state.json`
  retain the optimizer history, study identity and current phase.

Do not edit the running study's result files. To start an independent search,
use a different `--output` folder. Nothing overwrites the project's default
policy configurations.

To view a validated winner, run from `survival-simulator`:

```powershell
$env:EXPERT_POLICY_CONFIG = "$env:LOCALAPPDATA\NordicCupAI\bo-laptop\best_expert_policy.json"
$env:GLOBAL_PLANNER_CONFIG = "$env:LOCALAPPDATA\NordicCupAI\bo-laptop\best_global_planner.json"
.\.venv\Scripts\python.exe local_playground.py --no-predators --central-harvest
```

Before validation has produced those files, substitute the two `best_search_*`
files to inspect the provisional training winner. Clear these variables with
`Remove-Item Env:EXPERT_POLICY_CONFIG, Env:GLOBAL_PLANNER_CONFIG` to return the
playground to its default configs.

See the [DTU guide](hpc/README.md) for the parameter ranges and evaluation
methodology. The laptop and cluster use the same tuner; keep separate studies
when changing operating systems or Python versions.
