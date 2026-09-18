# Observation-only predator wall funnel

Use **`observed_run.py`**, not the older privileged physics probes in `run.py`.
The tested 33-predator setup uses a 30×100 wall, two stationary baits and one
sacrificial guide. Read the [full report](../../../docs/survival-wall-funneling.md)
for geometry, evidence and the unresolved repeated-arrival limitation.

From `survival-simulator/oscar-trap-research`, with simulator dependencies installed:

```sh
python survival/research/wall_funneling/observed_run.py --width 30 --length 100 --baits 2 --predators 33 --seed 62 --initial-jitter 8 --awake --seconds 180 --native
python survival/research/wall_funneling/observed_run.py --width 30 --length 100 --baits 2 --predators 33 --seed 42 --seconds 3000
python survival/research/wall_funneling/checks.py
python survival/debugger/catalog.py
python survival/debugger/serve.py
```

The existing replay viewer is at `http://127.0.0.1:9053`. Every completed case,
including failures, has a unique replay under `survival/results/wall_funneling/replays`.
Replays are intentionally ignored by Git, as in the original handoff.

`ObservedFunnel.act(observations, sim_time)` accepts native observation DTOs and
public time only. `adapter.py:make_policy` exposes the existing debugger interface.
Food refill is an explicit fixture assumption, not part of that adapter.

Experimental `--renew-guides`, `--gather`, `--waves` and `--staged-guides` cases retain unsuccessful
approaches for inspection. They are not recommended deployment options. Archival
`observed_policy_v*.py` and `run_v*.py` files preserve recorded source hashes.
