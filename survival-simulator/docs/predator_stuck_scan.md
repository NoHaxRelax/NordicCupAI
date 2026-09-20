# Predator confinement scan

The scanner now has both `--backend python` (default) and `--backend cpp`.
The C++ backend runs map generation, predator movement, all game ticks, and the
rolling confinement detector in compiled code. Python starts worker processes
and saves finished-game evidence. NumPy supplies native C math functions to
preserve numerical behavior; there are no Python calls in the per-tick loop.

Build and run the C++ version from the repository root:

```powershell
& "..\NordicCupAI\survival-simulator\.venv\Scripts\python.exe" survival-simulator/scripts/predator_stuck_cpp/build.py
& "..\NordicCupAI\survival-simulator\.venv\Scripts\python.exe" survival-simulator/scripts/predator_stuck_scan.py --backend cpp --workers 4 --no-images --output survival-simulator/runs/predator-stuck-cpp
```

The build needs a C++17 compiler, Python development headers, and NumPy.
Use a separate output directory for each backend. Native output manifests
also record the binary hash and Python/NumPy versions.

Run from the repository root in PowerShell using the simulator environment:

```powershell
& "..\NordicCupAI\survival-simulator\.venv\Scripts\python.exe" "survival-simulator/scripts/predator_stuck_scan.py" --workers 4
```

Defaults match this experiment:

| Setting | Default |
| --- | --- |
| Games | 10,000, seeds 0 through 9,999 |
| Agents | None, throughout the game |
| Starting predators | Exactly 100 successful native spawns |
| Game duration | 600 simulated seconds |
| Confinement radius | 15 world units, measured from the predator's center |
| Confinement interval | 60 simulated seconds |
| Simulation timestep | Native 0.1 second |
| Maps | Native 1600-by-1200 maps, biomes, rivers and obstacles |
| Output | `survival-simulator/runs/predator-stuck-scan/` |

The script is headless and uses separate worker processes. Change `--workers`
to suit the machine; the default is at most four. Each worker owns a full native
map, so memory usage also increases with the number of workers. `--games`,
`--predators`, `--seconds`, `--stuck-seconds`, `--radius`, `--start-seed` and
`--output` are configurable. `--no-images` omits the PNG crops.

For a short two-game check with a separate output directory:

```powershell
& "..\NordicCupAI\survival-simulator\.venv\Scripts\python.exe" "survival-simulator/scripts/predator_stuck_scan.py" --games 2 --seconds 65 --workers 2 --output survival-simulator/runs/stuck-quick-check
```

## What counts as a finding

At every native tick, the detector examines the full preceding 60-second
interval, including both endpoints: 601 sampled positions at the default
timestep. The first position in that interval is the circle's fixed center.
Every position must stay within 15 units of that center. Distance exactly equal
to the radius is allowed. A predator that leaves and returns does not qualify
for an interval containing the excursion.

This is a rolling test, so a predator can travel normally and then become
confined later. It catches stationary predators and small oscillations or
loops. It does not solve for the smallest circle with an arbitrary center:
the circle must be centered at the interval's first position. Positions are
checked at native simulation ticks, not interpolated between ticks.

The first qualifying interval per predator is saved. The game continues to its
full duration, even when there are no agents or findings already exist. This
detects a measured period of confinement, not proof of permanent entrapment.

## Preserved engine behavior

The Python backend calls native `SimulationCore` setup and `Environment.non_agent_step`
without replacing movement, sensing, collision, energy, rest, map rendering,
or random-number generation. Trees, fruits, and ambient predator spawns remain
enabled. Predators born later are tracked too, but cannot be flagged until they
have a full interval of history. IDs in the reports are stable indices within
each game, starting at zero.

Native setup makes a fixed number of spawn attempts, some of which fail. The
runner retries native spawning after setup until the starting population is
exactly 100. It records how many extra attempts were needed. Thus the requested
population is guaranteed, but these worlds need not match a stock invocation
that merely requests 100 spawn attempts.

The native placement check can accept a predator whose radius overlaps an
obstacle. Those placements are retained and recorded as
`spawned_overlapping_obstacle`, with separate counts in the summary. This helps
distinguish a spawn problem from a predator becoming trapped after valid
placement. Normal rest cycles are retained, and each finding reports how many
samples the predator was awake.

## Outputs and resumption

- `summary.json`: counts of complete and failed games, games with findings,
  flagged predators, and initial obstacle overlaps.
- `findings.csv`: one row per flagged predator, with game status, interval,
  coordinates, radius, maximum observed distance, and evidence paths.
- `manifest.json`: experiment configuration and hashes of the runner and
  simulator source files.
- `games/seed-XXXXXXXX/result.json`: durable per-game result and spawn catalog.
- `games/seed-XXXXXXXX/progress.json`: current simulation progress for a worker.
- `games/seed-XXXXXXXX/map.json`: obstacle geometry for games with findings.
- `games/seed-XXXXXXXX/events/predator-NNNN.json.gz`: the complete interval's
  positions, headings, energy and rest states, including column names.
- Matching `.png`: a close-up of the map, trajectory, and detection circle.

A `FOUND` message appears in the Python backend when the first predator in a
game qualifies. Its evidence is saved immediately. The C++ backend writes each
game's evidence after the complete game returns, and reports completed games
instead of per-tick progress. Aggregate CSV and counts update every 10 seconds
and at completion. C++ optional PNGs use flat biome colors with the exact
obstacle geometry and trajectory.
The summary separates findings from complete games and unfinished/error games.

Press **Ctrl+C** to stop. Python workers finish their current native tick or setup
and save partial results. C++ workers finish their current game and save it;
queued games check the stop request before starting. Rerun the same command to skip completed games; interrupted
or failed games restart from their seed. More games can be added and the worker
count can change. Changes to experiment parameters or source files require a
new output directory so incompatible results are not mixed.
Changes to unrelated agent policies do not invalidate this agent-free scan.

The output directory is locked against concurrent runners. After a hard crash,
if no scan process remains, delete its `.batch.lock` before resuming. Normal
completion and Ctrl+C release the lock automatically.

## Validation

The tests compare the optimized detector with a brute-force check over random
trajectories, including rests and excursions. They also check the exact interval
boundary, sliding windows, circular versus square distance, native confinement
in an enclosed cell, trace/image output, and resume/configuration checks:

```powershell
& "..\NordicCupAI\survival-simulator\.venv\Scripts\python.exe" -m unittest discover -s survival-simulator/scripts -p test_predator_stuck_scan.py -v
```

A real two-game run (seeds 0 and 1, 100 starting predators each, 65 seconds)
completed without errors and correctly resumed without rerunning either game.
Seed 1's predator 32 was motionless throughout seconds 0–60; it had spawned
overlapping a rock. The trace contained all 601 samples. Across the two games,
18 predators initially overlapped obstacles; only one was flagged by 65 seconds.

A full 600-second run of seed 1 also completed without errors. It started with
100 predators and ended with 101 because native ambient spawning remained
active. Twenty predators met the confinement rule, including 15 that had not
spawned overlapping an obstacle. Every saved finding was independently checked
against all 601 coordinates. Results and images from this validation are in
`survival-simulator/runs/predator-stuck-validation600/`.

The two-game check took about 31 wall-clock seconds with two workers. The full
600-second game took 288.7 seconds with one worker on this machine. At that
per-game rate, 10,000 games would take roughly 33 days on one worker or 8.4 days
with ideal four-worker scaling. Actual throughput depends on the machine,
other workloads, and each map. Use measured per-game times from `result.json`
to update the estimate. The 10,000-game experiment is configured by default,
but has not been run as part of creating the script.

## C++ validation and Runpod

The Windows C++ pilot ran seeds 0 and 1 for 600 seconds with 100 starting
predators each in 7.2 seconds total with two workers. They produced 16 and 20
findings. Seed 1 reproduced the earlier Python run's 20 finding records and all
12,020 saved trajectory samples exactly (ignoring output paths and PNG fields).
The separate parity test matched positions, headings, energy, rest state, and
RNG state at all 651 ticks in each of two 65-second games. It also compared the
bulk C++ detector against Python sliding windows. Boundary, excursion, diagonal,
and randomized path detector checks passed.

```powershell
& "..\NordicCupAI\survival-simulator\.venv\Scripts\python.exe" survival-simulator/scripts/predator_stuck_cpp/verify.py --seconds 65 --seeds 0 1
```

This is platform-specific validation. Floating-point trajectories are not
assumed identical between Windows and Linux. `runpod_run.sh` builds on the target
and runs the same Python/C++ parity check there before launching its assigned shard.
Its pinned validation dependencies are in `predator_stuck_cpp/requirements.txt`.

Prepare the source-only transfer archive with:

```powershell
& "..\NordicCupAI\survival-simulator\.venv\Scripts\python.exe" survival-simulator/scripts/predator_stuck_cpp/package.py survival-simulator/runs/predator-stuck-source.tar.gz
```

The archive contains the scanner and its Python reference dependencies. It
excludes agent policies, runs, credentials, virtual environments, and binaries.
After extracting on a CPU pod with Python 3.12, development headers, a C++17
compiler, and `uv`, run `bash scripts/predator_stuck_cpp/runpod_run.sh`.
It uses 32 workers by default (`SCAN_WORKERS` overrides this), omits PNGs, and
writes complete diagnostic traces under `runs/predator-stuck-diagnostics`.
`SCAN_GAMES` defaults to 1,250 and `SCAN_START_SEED` defaults to zero; the fleet
launcher assigns disjoint ranges weighted by each pod's CPU count.
Configure a pod shutdown deadline separately and retrieve the output before
terminating the pod.

The user approved the source upload and expanded the experiment to eight pods
with passive movement diagnostics. The distributed run and retained evidence
are documented in [predator_stuck_diagnostics.md](predator_stuck_diagnostics.md).
Use `--diagnostics --backend cpp` to collect the detailed traces locally.
