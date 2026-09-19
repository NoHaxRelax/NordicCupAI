# 1,000-game native benchmark protocol

## Frozen strategy

Baseline commit: `9b0c3e8` (`Freeze exact 9059 native entrapment baseline`). The runner verifies all 39 source hashes from the original replay before starting any games. No later reproduction or mapping fixes are included.

## Cases and inputs

- World seeds **0 through 999**, once each, without site filtering or rerolling.
- Run each native game until extinction or **3,000 simulated seconds**, timestep 0.1.
- Native defaults: five initial agents, zero initial predators, 50 initial trees, normal initial fruit. Predators spawn naturally.
- Controller inputs: normal agent DTOs and simulation time only. Its tie-breaking RNG uses the same fixed policy seed, 0, for every map; world seeds are not passed to the controller.
- Native energy, aging, collisions, birth costs, mutations, fruit/tree dynamics and predator decisions remain unchanged.
- Independent fresh worker process per game; one worker per vCPU and one BLAS/OpenMP thread per worker. CPU provisioning changes throughput, not the game timestep.

The simulator and shared-map policy contain ordering/time-budget effects, so a world seed alone is not a promise of identical trajectories across machines. Source, environment and recorded trajectories are retained. No policy edits or case exclusions may follow inspection of benchmark results.

## Results

The primary whole-game result is survival to the 3,000-second horizon. Also report score and survival-time distributions, geometric trap availability, trap discovery, physical bait establishment, bait gaps, replacement events, guide assignments, delivery arrivals, predator proximity retention and rear-entrance occupation. Keep errors separate and count all 1,000 requested seeds in denominators.

**Delivery arrivals / guide assignments is not a predator-capture success rate.** A predator may need multiple guides, and public observations do not expose its identity. “Held30” is a spectator proxy: remaining within 40 units of physically verified bait for 30 seconds. It does not prove permanent capture.

Full geometry is available only to the independent evaluator. It verifies the selected site's coordinates and counts bait when an assigned bait-role agent is physically within 3 units of the actual depth-5 goal. Rear occupation means a predator is within 40 units of the opposite mouth and has axial depth at least `overlap - 5`. None of those evaluator values enter policy decisions.

## Saved artifacts

Per game: result, role events, one-second metric history, static obstacle geometry, and **every-tick, lossless agent/predator trajectories** with energy and roles. These compact trajectories are not a full fruit/tree/observation replay. The original 9059 reference replay retains its complete recorded frames.

Batch metadata includes exact source hashes, dependency versions, shard assignments, infrastructure cost and cleanup evidence. Completed case files are resumable; duplicate seeds are rejected when merging shards. Report any errors or incomplete cases explicitly.

## Running locally

Use `run.py benchmark` to run a local batch. Start with one worker on a laptop;
increase workers only when CPU and memory allow it. A completed case is reused
when the same output directory and matching protocol are supplied again.
An interrupted case starts over. This runner does not provision cloud hardware.

Example local command:

```bash
python survival-simulator/run.py benchmark \
  --out /tmp/entrapment-9059-benchmark --count 1000 --workers 1
```
