# Overnight Bayesian tuning on central DTU HPC

The supplied job targets **DTU's central LSF 10 cluster**, using the `hpc` CPU
queue: **24 CPU slots on one node, 4 GB per slot (96 GB reserved), 12 hours,
no GPU**. The simulation uses Python/NumPy/SciPy CPU code. An A100 would sit idle.
DTU lists many 24- and 32-core Intel nodes and some 48-core nodes; the submission
does not restrict the CPU model, allowing more scheduling options.

Official DTU references, checked 2026-09-17:

- [Central cluster hardware](https://www.hpc.dtu.dk/?page_id=2520)
- [LSF submission directives and per-slot memory](https://www.hpc.dtu.dk/?page_id=1416)
- [Python modules and virtual environments](https://www.hpc.dtu.dk/?page_id=3678)
- [Python job example using the 3.11.7 software stack](https://www.hpc.dtu.dk/?page_id=5198)

Departmental DTU clusters may use different schedulers. This script requires
`bsub`; it is not a Slurm submission. Check `classstat` and `nodestat -F hpc` on
the cluster for your available queues and current resources.

## Prepare and submit

Copy the **whole current `survival-simulator` directory**, including `src`,
`config`, the two tuning Python files and `hpc`. Many current policy files may
not yet be committed, so a clone of an older revision is insufficient. Do not
copy the Windows `.venv`, `.uv-cache`, or local `runs` directory.

From a DTU login shell:

```bash
cd NordicCupAI/survival-simulator
bash hpc/setup_dtu.sh
bsub < hpc/submit_bo.sh
bstat
```

Setup installs the minimal pinned tuning dependencies in `.venv-hpc`; it runs
no simulations. It defaults to `python3/3.11.7`. If that module is unavailable,
use `module avail python3` and select an available version **>=3.11**:

```bash
DTU_PYTHON_MODULE=python3/<available-version> bash hpc/setup_dtu.sh
```

The setup records the module name. The batch job reloads that same module and
uses the Linux environment. No package downloads are needed during the job.
Avoid changing its Python/module/dependencies midway through a study.

Logs are `bo_JOBID.out` and `bo_JOBID.err` in the submission directory. Follow
progress with `tail -f bo_JOBID.out`. `bkill JOBID` stops a job; completed episodes
remain saved. There is no display window, renderer, or GPU allocation.

To check a short real submission first, copy `submit_bo.sh` to a separate script
and use 4 slots, `-W 00:15`, `--workers 4 --trials 3 --startup-trials 1`,
`--train-seconds 5 --validation-seconds 5 --finalists 2`,
`--search-hours 0.1 --total-hours 0.15 --output runs/bo-smoke`.
Leave the seed lists disjoint. Run simulations on compute nodes, not login nodes.

## What the job does

1. Enables the centralized, territory-based harvesting policy with predators
   disabled. The baseline is the current policy under these same conditions.
2. Uses Optuna's multivariate TPE, beginning with 24 completed startup trials
   (the baseline is included). Eight candidates are normally in flight, with
   their three seeds evaluated across 24 independent processes.
3. Searches for up to 8 wall-clock hours or 300 **completed** trials. Each trial
   is the mean game score over seeds `1, 7, 42`, with a 600-simulation-second
   horizon. There is no early-score pruning: slow starts can recover.
4. Evaluates up to seven distinct top configurations **plus baseline** on seeds
   `1001, 1007, 1042`, over the full 3,000-second horizon. Search ends early if
   all trials finish, giving validation the extra time.
5. Stops cooperatively at 11.5 wall-clock hours, leaving 30 minutes before LSF's
   12-hour limit. Only complete seed sets can win. Extinction ends an episode
   normally with its earned score; a wall-clock interruption earns no score.

The number of completed trials depends on cluster throughput and population
growth. The 300-trial setting is a ceiling, not a promised overnight count.
The finalist phase may need another submission if episodes are slow.

The objective is actual mean game score, with no invented population/trait
bonus. Reports and individual episodes include population, extinction, fruit
energy, rotting, births, trait scores, and first shared-map phase time. Mean
living traits are diagnostic and can rise simply because weaker agents died.

| Tuned setting | Current baseline | Search range |
| --- | ---: | --- |
| Orchard revisit interval | 20 s | 8–30 s |
| Territory cell size | 100 | 75, 100, 125 |
| Food imbalance tolerance | 0.35 | 0.15–0.50 |
| Food budget per agent per second | 6 | 4–10 |
| Desired ripe fruit energy | 56 | 48–60 |
| Emergency energy threshold | 90 | 70–140 |
| Parent energy reserve | 100 | 80–160 |
| Normal breeding cooldown after alignment | 8 s | 6–16 s |

Elite selection, the founder-normalized breeding floor, gene backups, mapping
behavior, and elite cooldown are preserved. Fruit mechanics are not tuned.
The exact space lives in `SPACE` in `tune_policy.py`; changing it requires a
new output directory.

## Results, resuming, and using the winner

Results are in `runs/bo-dtu/`:

- `report.md` and `validation_results.json`: full-horizon comparison, completion
  status, and selected configuration. An incomplete comparison is provisional.
- `best_expert_policy.json` and `best_global_planner.json`: selected validated
  settings, exported only once a comparable baseline has also finished. The
  baseline itself wins when it scores highest.
- `best_search_expert_policy.json` and `best_search_global_planner.json`: the
  shorter-horizon search winner; this is **not** the validated result.
- `search_results.json`: all completed trials ranked by training score.
- `manifest.json`, `optuna.journal`, and `candidates/`: configs, versions, source
  fingerprint, optimizer history and individual seed results for resuming.
- `.venv-hpc/installed-packages.txt` (outside the output directory): installed
  dependency versions from setup.

Re-submit `bsub < hpc/submit_bo.sh` to continue the same study. It retains
completed trials and seed results; an interrupted episode restarts from its
beginning. Each submission receives a fresh time budget. The trial limit counts
previous completed trials. Increase `--trials` to extend the search.

To finish validation without further searching, change `--search-hours 8` to
`--search-hours 0` in the submission script. Use the same output directory,
seeds, episode lengths, configs and software. Do not run concurrent jobs against
the same output: the coordinator lock rejects that. Separate output directories
create independent studies.

The manifest rejects incompatible resumes after changes to policy code,
settings, Python, relevant package versions, seed batches, or episode lengths.
Worker count and wall-clock budgets can change. Keep results on persistent
storage; do not use disposable node-local scratch as the only copy.

To try the selected settings in the playground, load **both** files:

```bash
export EXPERT_POLICY_CONFIG="$PWD/runs/bo-dtu/best_expert_policy.json"
export GLOBAL_PLANNER_CONFIG="$PWD/runs/bo-dtu/best_global_planner.json"
python local_playground.py --no-predators --central-harvest
```

Use a normal playground environment with the full `requirements.txt` for the
interactive viewer; the minimal batch environment does not include its plotting
dependencies. The tuner never overwrites the project's default configs.

## Reproducibility and interpretation

The exported planner disables `biome_inference.adapt_to_wall_clock` while keeping
confidence-based schedule changes. This prevents CPU contention from changing
when biome fits happen. Normal project defaults retain wall-clock adaptation.
Load the exported planner with the expert configuration to preserve the policy
that was evaluated.

All workers use one BLAS/OpenMP thread. Identical seeds start with identical
worlds in the same software environment; differing actions and births consume
different later RNG draws. Asynchronous TPE suggestions can vary with completion
order. Seeded evaluation does not make parallel optimization bit-for-bit
identical across submissions or hardware.

Validation seeds select a winner, so they are not an unbiased final test set.
Before claiming a general improvement, compare the exported winner and baseline
on additional untouched seeds at the same 3,000-second horizon. The search is
specifically for predator-free play.
