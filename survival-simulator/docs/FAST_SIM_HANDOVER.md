# Fast simulation handover

Research branch: **`codex/late250-checkpoints`**. The frozen late-game training
results are commit **`76433d1`**. This branch contains the engine, native policy,
experiment runners, configurations and raw results. The 1,000-map final evaluation completed on 2026-09-19; results are in
`docs/late250/final/RESULTS.md`. Do not mistake training gains for full-game
validation scores.

## Why it is fast

1. **Both the game engine and the agent policy run in C++.** Porting only the
   engine leaves the Python decision loop as the main bottleneck.
2. **One Python call runs the whole game.** `run_policy` keeps observations,
   decisions and stepping inside C++, releasing the GIL. No per-tick JSON or
   process-pipe exchange. Python only schedules games and collects results.
3. **Independent games run in parallel processes.** Currently ten CPU pods,
   32 workers each: up to 320 simultaneous games. GPUs are not used.
4. **Late-game tuning reuses exact checkpoints.** Linux `fork()` copies a saved
   process state on write, including engine RNG and policy memory. Trials skip
   the shared opening. Final evaluation still runs complete games from the start.

The older measured comparison in `fastsim/README.md` is 99.6 s with Python policy
and native engine versus 21.4 s with the fully native loop on one machine/seed
(4.65x). This is not a universal speedup claim. Fleet throughput and checkpoint
savings are additional effects, not improvements in single-game execution speed.
The current fleet completed 8,245 full games in roughly six minutes after launch;
that is aggregate throughput across 320 workers, not six minutes per game.

## Files to read

| File | Purpose |
|---|---|
| `fastsim/_engine.cpp` | Native game rules and state |
| `fastsim/_policy.cpp` | Python binding and native observation/decision/step loop |
| `fastsim/_orchard_policy.cpp` | Native policy implementation and memory |
| `fastsim/_orchard.hpp` | Orchard survival and population behavior |
| `fastsim/_evasion.hpp` | Predator avoidance behavior |
| `fastsim/policy_abi.hpp`, `policy_iface.hpp` | Public observation/action interface |
| `fastsim/fastpolicy.py` | Small Python wrapper: `PolicySimulationCore` |
| `fastsim/build.py`, `build_policy.py` | Compile the extensions and record build hashes |
| `scripts/evaluate_frozen1000.py` | Parallel full-game runner with per-tick timing |
| `scripts/tune_late250.py` | 20-family BO search using exact fork checkpoints |
| `scripts/evaluate_late250.py` | Frozen winners on fresh full games |
| `docs/late250/PROTOCOL.md` | Exact current experiment design |

Paths in this document are relative to `survival-simulator/`.

## Build and run one full game

Use a separate clone/worktree so you do not disturb someone else's working tree.
Linux needs Python development headers and a C++17 compiler; the pods use Python
3.12 and NumPy 2.3.5. Build locally on each machine; do not copy extension binaries
between Python versions or architectures. Git LFS may be required by repo hooks.

```bash
git clone --branch codex/late250-checkpoints https://github.com/NoHaxRelax/NordicCupAI.git NordicCupAI-fast
cd NordicCupAI-fast/survival-simulator
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.3.5
python fastsim/build.py
python fastsim/build_policy.py
python fastsim/check_boundary.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python - <<'PY'
import json, time
from pathlib import Path
from fastsim.fastpolicy import PolicySimulationCore

cfg = json.loads(Path('docs/families20/scheduled_breeding/winner.json').read_text())['config']
sim = PolicySimulationCore(seed=12001, predators=True)
sim.policy_init(0, cfg)
start = time.perf_counter()
steps, peak, interface, policy, engine = sim._engine.run_policy(3000., 3000., True)
elapsed = time.perf_counter() - start
print(dict(score=sim.env.score, simulated_seconds=sim.env.time,
           ticks=steps, peak_agents=peak, wall_seconds=elapsed,
           policy_us_per_tick=policy / max(steps, 1) / 1000,
           wall_us_per_tick=elapsed * 1e6 / max(steps, 1)))
PY
```

The horizon is 3,000 **simulated seconds**, not a wall-time delay. Runs stop on
extinction. One tick advances the entire population, not one agent. Profile values
are nanoseconds; wall timings include scheduling effects under CPU contention.
The evaluation runner also records process CPU time, which is useful alongside
wall time. Initialization is outside its timed game loop.

For batches, follow `evaluate_frozen1000.py`: a `multiprocessing.Pool`, one fresh
`PolicySimulationCore` per job, `imap_unordered`, and one JSONL result per game.
Set worker count to available CPU cores and keep BLAS thread counts at one to avoid
oversubscription. Do not run expensive verification or visualization on every tick
in a throughput benchmark. Record frames separately for selected runs.

## Correctness and policy information

The native policy is a separate translation unit with public observations and
agent state as its interface. It cannot include the engine or Python API under the
checked build boundary. `check_boundary.py` checks includes, identifiers, ABI
fields and compiled symbols. This is a code boundary, not an OS security sandbox.
The host evaluator can inspect true world state for scoring; the policy cannot.

`fastsim/verify.py` and `verify_evasion.py` provide engine/policy lockstep checks;
see `fastsim/README.md` before changing numerical behavior. Do not enable
`-ffast-math`: the build deliberately preserves floating-point semantics. NumPy
math dispatch can differ across CPU architectures, so do not assume cross-machine
bit identity without checking it. Keep source/build/config hashes with results.

For the current checkpoint search, all 200 unchanged baseline continuations were
verified to reproduce their original final score and extinction time exactly on
each of ten pods. Checkpoint metadata was identical across the pods. Each trial
changes only parameters, retaining RNG and policy memory in a disposable fork.
Snapshots live in RAM; the saved metadata and frozen source allow regeneration,
but are not portable serialized snapshots. Baseline future extinction time selects
training checkpoints only and is never given to the policy. Final evaluation uses
fixed winner parameters from game start, with no future-death trigger.

## Reproduce or inspect this experiment

Training: `python scripts/tune_late250.py --pod-index 0 --out results/pod-0`
(`0..9`, two families per index, default 32 workers). It is Linux-specific because
it uses fork. Install the full `requirements.txt` if using the research/tuning
tools rather than just the minimal native example. Never reuse an output directory.

After collecting all 20 winner files under `docs/late250/pod-*`, run
`python scripts/evaluate_late250.py --shard 0 --out final/shard-0` for each shard
`0..9`. Each shard evaluates 22 configurations on 100 maps: 2,200 games.
The runner refuses to overwrite an existing manifest. Training maps are
11001–11200; fresh final maps are 12001–13000.

Existing results: `docs/late250/pod-*` includes every trial, per-map score gain,
timing, winner, checkpoint metadata and provenance. Final results are saved
under `docs/late250/final` with means, confidence intervals and paired comparisons.
The watcher is `python scripts/watch_late250.py`; it requires separately authorized
SSH access to the research pods. No SSH private keys are stored in the repository.
