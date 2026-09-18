# Simple policy experiment

The new controller is selectable with `--policy simple`. The existing policy
remains the default and is selectable with `--policy standard`.

## Behavior

- Population targets: **12 agents under 40** initially, **6** from 900 seconds,
  **3** from 1,800 seconds. Older agents do not count against these targets.
  Birth spacing is approximately 2.7 / 5.3 / 8 seconds respectively, allowing
  replenishment within the 40-second window. Births preserve 75 energy in the
  parent and eligible parents are ranked by founder-normalized inherited traits.
- Stable homes spread across productive observed/estimated biome cells, refreshed
  every 30 seconds or on membership changes. Agents patrol locally and rest;
  patrol and scanning intervals increase over simulation time.
- Shared fruit memory and one collector per fruit. Eat immediately, without
  predicting ripeness. Food assignments refresh every 0.5 seconds. Unreachable
  destinations have a 30-second retry delay.
- Initial shared-coordinate discovery is retained. After alignment, map sensing
  slows from approximately 0.5-second to 2-second intervals. Position prediction
  still integrates every action. Biome fits start at 10-second intervals and can
  back off to 60 seconds. Shared trap estimates remain available.
- Confirmed senescence converts an agent into an aging scout: no food assignments,
  breeding, or productive home; instead it explores a distinct unvisited frontier.
  Detection uses two plausible excess-drain measurements after deducting action
  costs, not age alone. Fruit gains do not clear confirmed aging. Scouts avoid
  contact with observed fruit, but the engine can still force incidental pickups.

Parameters are in the `simple` section of `config/expert_policy.json`.

## Aging scout check

A predator-free 180-second integration check on seed 42 finished with 17 living
agents (7 under 40), score 197.97. It reached 12 young agents at 60 seconds; total
population later peaked at 30 at the sampled 120-second checkpoint. It is a target,
not a guaranteed population: parents still need enough energy to reproduce.

Across the run, 26 agents were detected as aging scouts and they issued 1,280
moving actions. Runtime assertions verified that confirmed scouts held no food
reservations or homes and issued no birth actions. At 120 seconds the map had
588 sampled cells, versus 422 in the earlier simple-policy snapshot. The higher
young population and scout change were applied together, so this does not isolate
their individual effects or establish full-map coverage. Local artifacts are in
`runs/simple-aging-scout-check/`.

The full 523-test suite passed. The subsequent regression check for avoiding all
visible fruits, including those beyond the five nearest, passed with the other
17 simple-policy tests. Long-horizon survival of this revised policy is unverified.

## Previous version measurements (8 / 4 / 2 total-population targets)

The following measurements predate the higher young-population targets and aging
scouts. They document the earlier implementation, not performance of the revision.

Both short comparison runs used seed 42, no predators, no rendering or diagnostics,
and 120 simulated seconds. The standard run enabled centralized harvest.

| Measurement | Standard | Simple |
|---|---:|---:|
| Wall time, including startup | 40.34 s | 29.77 s |
| Score | 130.80 | 130.17 |
| Final population | 9 | 9 |
| Controller time | 29.41 s | 18.89 s |
| Shared map update time | 8.26 s | 2.04 s |
| Navigation time | 3.09 s | 8.26 s |

The simple run took about **26% less wall time**, with a score about **0.5% lower**.
These are local measurements from one seed, not a controlled statistical benchmark.
Component times overlap: navigation is part of the coordinator and controller.
Navigation remains an important cost, particularly when the coarse home assignments
require detours. Simplifying the policy does not eliminate simulation/sensing costs.

A separate **600-second** predator-free run on seed 42 completed with:

- Score **654.61**, **7 living agents**, 36 births and 34 deaths from energy depletion.
- 1,488 fruits eaten, averaging 36.70 energy each.
- 118.06 seconds of wall time; average controller time 11.35 ms per tick.
- Mean living trait score 1.292 relative to the founder population.

This demonstrates multiple generations, not permanent survival. There is no matched
600-second standard-policy run in this check. The scheduled population reductions
at 900 and 1,800 seconds are covered by unit tests, not this short episode.
**Survival to 3,000 seconds remains unverified for the simple policy.**

Validation: 518 unit tests passed. After adjusting the simple-map legend, all six
renderer tests passed again, and a real 120-second internal-map snapshot rendered
successfully. Tests cover birth limits and replacement overlap, trait preference,
food reservations and stale observations, stable productive homes, blocked-goal
retry limits, scan/rest cycles, and odometry between map sensing passes.

Local artifacts:

- `runs/simple-policy-comparison/results.json`
- `runs/simple-policy-comparison/simple-map.png`
- `runs/simple-policy-600-seed42/seed-42.json`

## Run it

From `survival-simulator`:

```powershell
.\.venv\Scripts\python.exe local_playground.py --policy simple --no-predators --seed 42 --speed 5
```

For headless progress and a final score:

```powershell
.\.venv\Scripts\python.exe -u local_playground.py --policy simple --no-predators --seed 42 --headless --no-diagnostics
```

For a checkpointed full-horizon comparison across seeds:

```powershell
.\.venv\Scripts\python.exe -u benchmark_survival.py --policy simple --seeds 42 1001 1007 --workers 1 --seconds 3000 --output runs/simple-survival
```

Ctrl+C saves progress. Repeat the same command to resume; choose a new output
folder after any source or configuration changes.
