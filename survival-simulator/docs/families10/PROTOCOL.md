# Ten-family experiment

Status: complete. SSH access was restored and all ten families finished 30 trials
and their 100-map evaluations. See RESULTS.md and the per-family raw artifacts.

Build and public-ABI boundary checks passed. Four full local smoke games (seed
5999, separate from both experiment panels) completed with baseline, gaze,
cone and wall modes in 2.5–5.2 seconds each. These are execution checks, not
performance estimates. Source reviewed includes Nikolaj's native Orchard/evasion,
the previous 16-family experiment and its held-out results, Oscar's orchard
population/reproduction/navigation core, and the newly pushed wall-clear wrapper.

Base: Nikolaj's `f4c2a53`, with the public-observation-only native Orchard policy.
His newest push documents inert parameters in the earlier search. This experiment
uses only parameters read by the native family. The earlier 100-map mean of 1421.4
is context, not a paired control for these new seeds.

| Family | What changes |
|---|---|
| sidestep | Reaction distance, sideways angle, sprint threshold |
| radial | Direct retreat with no sideways dodge |
| gaze | Face the observed predator during escape |
| cone | Avoid reacting outside its sight cone, beyond hearing range |
| wall | Search headings that break sight behind observed wall edges |
| population | Population limits as food declines |
| breeding | Early, late and emergency reproductive energy reserves |
| heirs | Timing, energy and trait selection for replacement generations |
| harvest | Fruit ripeness, range and travel cost |
| exploration | Exploration energy, distance and patience at orchards |

Each family receives 30 trials: one initial configuration, five random trials,
then 24 Gaussian-process expected-improvement trials. Every trial runs the same
32 full games (world seeds 6001–6032). Maximize mean score, not survival time.
Freeze the best training configuration and evaluate it on world seeds 7001–7100.
The sidestep pod also runs the unchanged baseline on those 100 maps.
Total: 9600 training games + 1000 test games + 100 control games.

Game horizon 3000 seconds, natural predators, normal energy and aging. Policy seed
is always 0, separate from world generation. Native policy can access only the
public ABI; wall escapes use observed edge memory, never engine coordinates.
The corner heuristic rewards occlusion only when current and proposed positions
are outside hearing range. It is not a guarantee of successful evasion.

Use one existing 32-vCPU Oscar pod per family, with 32 worker processes and
single-threaded BLAS. Each process runs the entire simulator and policy in C++.
The per-pod job deadline is 1200 seconds (20 minutes): ten jobs at $0.96/hour
cost at most $3.20 of attributed active compute. The user subsequently removed
the historical $10 research cap; this run retained its existing deadline.
These already-running pods
continue billing outside the job interval and are left running as requested.
Run a 32-map pilot before launching the ten jobs; reduce training-panel size only
with a documented protocol revision before any optimization sees results.

Commands (from survival-simulator):

```sh
python fastsim/build.py
python fastsim/build_policy.py
python fastsim/check_boundary.py
python scripts/tune_families10.py --family sidestep --out /workspace/pilot --pilot
python scripts/tune_families10.py --family FAMILY --out /workspace/results/FAMILY
python scripts/report_families10.py /path/to/collected/results
```

Keep raw per-game JSONL, every trial/configuration, source and binary hashes,
Python/NumPy versions, and completion markers. The report refuses incomplete
campaigns. Plot both each trial's mean and best-so-far mean for all 30 iterations.
The 100-map test set is never fed back into the optimizer.
