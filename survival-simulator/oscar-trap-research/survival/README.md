# Survival Simulator: simple strategies and local mechanics research

This directory contains a pinned public simulator copy, controlled mechanics probes, and observation-only baseline policies. See [the findings](../docs/survival-simple-strategies.md) for recommendations and measured results.

- [Replay debugger](debugger/README.md): play, scrub, change speed, inspect agents, and record future policies.
- [Source provenance](SOURCE.md)
- [Game edge cases](research/edge_cases.md)
- [Predator control](research/predator_control.md)
- [Water bait tuning and food constraints](research/water_tuning.md)
- [Wall acquisition, generated sites and replacement](research/wall_acquisition.md)
- [Feeding a wall crew across generations](research/wall_orchard.md)
- [Dedicated sprinters, relays and banishment](research/sprinter_decoys.md)
- [Speed selection and fruit timing](research/speed_and_fruit.md)
- [Observation-based fruit waiting experiments](research/fruit_waiting.md)
- [Policies](research/simple_policies.py)
- [Full-map benchmark harness](research/benchmark.py)

Setup: Python3.10+, create `.venv` here and install `vendor/survival-simulator/requirements.txt`. Exact upstream dependencies are pinned. From the project root:

```sh
python3 -m venv survival/.venv
survival/.venv/bin/python -m pip install -r survival/vendor/survival-simulator/requirements.txt
survival/.venv/bin/python survival/research/edge_cases.py
survival/.venv/bin/python survival/research/speed_and_fruit.py
survival/.venv/bin/python survival/research/predator_control.py
survival/.venv/bin/python survival/research/benchmark.py --mode speed --seeds 1 2 3 --horizon 3000
```

Benchmark modes: `dummy`, `greedy`, `nursery`, `speed`. `SimplePolicy` receives only agent observation DTO fields and simulation time, returns one finite action per living observed agent, and uses no map coordinates, engine handles, fruit ages, predator energy or rest state. The harness reads engine state only to record diagnostic metrics. The vendored engine is unmodified.

Results are local macOS runs; the organizer says official evaluation is Linux and recommends same-OS testing for reproducibility. Do not assume seed-identical or hosted-server-identical results. Files under `results/initial/` are superseded exploratory runs made before a population-budget/exploration correction and retained only as provenance. Current benchmark JSON includes the policy file SHA256.

No endpoint was deployed, no online attempt was started, and the single final evaluation remains unused by this work.
