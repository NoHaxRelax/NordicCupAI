# Harvest reliability handover

The active entry point is `exploit_lab/harvest_policy.py`. The current controller
is C++ in `models/harvest_controller.cpp`, built as the
`exploit_lab.reliable_harvest` extension. Python loads that extension ahead of the
historical same-named `.py` file. Assert `IMPLEMENTATION == 'cpp'` when running the
current version. The observation tracker remains `predator_memory.py`; this is
not yet a fully native policy/simulation loop. The previous v3 implementation is
retained as `harvest_v3.py`. The starting source came from the adjacent
`NordicCupAI-bug-validation` worktree's `exploit_lab/HANDOVER.md`; native engine
and orchard-policy sources came from `NordicCupAI/survival-simulator/fastsim`.

The requested order is reliability first, minimum actions sufficient for the
remaining game, then score tuning. Headline validation requires 1,000 seeds.

## Current behavior

- Select a farm only when motion compensated for the observer is verified and
  the predator is approaching and facing the farm. The C++ controller recognizes
  sprint strides of 15, 12, 7.5 and 4.5 units across the shipped terrain types;
  walking strides are excluded. The earlier Python baseline's 11.1-unit threshold
  excluded valid slow-terrain encounters and contributed to zero-harvest games.
- Intercept using the predator's reported heading. Recent displacement can be
  deflected by obstacles without changing its heading.
- The sacrifice is the actual immediate living predecessor. Simulate the full
  earlier mortality/skip chain using energy after native actions, including
  spawning. An earlier starvation can otherwise skip the sacrifice itself.
- Reserve both lost participants and foreseeable natural deaths; never count
  newborns toward the colony floor. Recheck the floor on every retry.
- Build a complete batch before changing policy state. Insufficient action
  capacity cancels the attempt; batches are never truncated after sacrificing.
- Drain once. A retry holds the farm still and renews its skip with the next true
  predecessor. Its unchanged age proves its observations are stale. The default
  limit is two delivery ticks, bounding intentional losses to one farm and two
  predecessors.
- Confirm disappearance using the public score increase, allowing for bounded
  simultaneous positive meals elsewhere. The earlier baseline missed some real
  harvests because these meals reduced the net score gain. Suspend new commitments
  for 0.5 seconds after any pending farm disappears, even if confirmation fails,
  while pre-harvest observations age out. Block certified targets immediately.
- Ignore a certified sleeper only after stationary confirmation. Share matching
  sightings between observers and retain exact static-terrain fingerprints for
  later observers. Unknown stationary predators remain threats: a stationary
  observation alone does not prove safe sleep. Renewed movement revokes a track's
  certificate.
- The C++ base policy establishes an opening target of eight agents, only spending
  birth energy when the parent retains more than 25 energy after its planned move.
  It collects fruit immediately during the first 60 seconds, then uses a 15-second
  ripening wait with the existing hunger override. One eligible agent can pause
  for a perception observation, with a one-second cooldown and conservative range.

## Minimum drain and the later score knob

Every stationary half-turn costs 0.5 energy. Opposite half-turn pairs preserve
heading and avoid large accumulated angles. Let `E` be farm energy after travel,
`R` remaining simulation seconds, and `M` a conservative bound on positive meals
the predator could also eat in the same tick. The minimum pair count is

```text
debt = max(200.000001, 100 + 30*(R + 0.1)) + M
actions = 2 * ceil(E + debt)
```

The bounds are predator energy at most 200, wake threshold strictly above 100,
and recovery 30 energy/second. The immediate-negative condition matters near the
end of a game. The extra 0.1 seconds covers one recovery tick. Possible positive
meals include other agents and same-tick births; never assume the predator's
nearby-agent iteration order.

`extra_drain_actions=0` is the reliability default. Later increase this **even**
integer to add drain without changing targeting or retry logic. With the default
capacity, the response budget grows by the same amount. If an explicit
`max_actions_per_tick` is supplied, it remains a hard commitment budget and may
prevent attempts when insufficient. Additional 200 drain actions add one score
unit per successful harvest, algebraically; large response transport has not
been validated by these in-process tests.

```python
from exploit_lab.harvest_policy import HarvestGame
result = HarvestGame(seed=1, extra_drain_actions=0).run()
```

## Reproduction

Build both native extensions with the active interpreter:

```text
python fastsim/build.py
python fastsim/build_policy.py
python -m unittest exploit_lab.tests.test_reliable_harvest exploit_lab.tests.test_harvest_engine -v
python -m exploit_lab.verify_safety
python fastsim/check_boundary.py
python -m scripts.harvest_benchmark --seeds 1000 --start 40001 --workers 16 --out runs/harvest-reliability/native.jsonl
python -m scripts.harvest_benchmark --engine python --seeds 1000 --start 40001 --workers 32 --out runs/harvest-reliability/python.jsonl
```

The policy RNG is fixed at zero, independent of the hidden world seed. The native
policy's new `policy_observe(states, time)` entry point consumes only public
dictionaries; it supports both simulation engines and observation filtering.
Engine truth is used only in the separate benchmark's diagnostics. Regression
tests compare public-input decisions to the existing native interface.

Per-run manifests record arguments and source hashes. The merge tool rejects
incomplete runs, duplicate seeds, mixed engines/settings, or mixed policy source.
The Python engine uses object-identity set ordering; a seed alone does not promise
bit-identical replay across processes. Native results are reported separately.

Use the C++ engine and native base policy for future large sweeps. The user also
requires further behavior changes and performance optimizations to be implemented
in C++. The controller has now been ported to C++; the original Python-engine
1,000-seed run applies to the earlier baseline. Future Python-engine checks should
be targeted checks, not the default optimization workflow.

Build the harvest extension with pybind11 3.0.1 headers, C++17 and the active
Python interpreter's development headers. On the Linux validation image:

```sh
python3 -m pip install --target build-deps pybind11==3.0.1
g++ -O2 -std=c++17 -shared -fPIC -I build-deps/pybind11/include \
    -I /usr/include/python3.12 models/harvest_controller.cpp \
    -o exploit_lab/reliable_harvest.cpython-312-x86_64-linux-gnu.so
python3 -c "import exploit_lab.reliable_harvest as h; assert h.IMPLEMENTATION == 'cpp'; print(h.__file__)"
```

Use the matching extension suffix/include paths on other platforms. The Linux
build was shared byte-for-byte across the four fresh-seed workers. Each worker's
`build-info-policy.json` also records the harvest C++ source and binary hashes.

## Validation status

The current C++ controller completed 1,000 fresh seeds (50001–51000), with extra
drain actions fixed at zero. See [the complete report](harvest_reliability/cpp_fresh_1000.json)
and its adjacent per-seed records/manifests.

| Measurement | C++ candidate, 1,000 fresh seeds |
|---|---:|
| Mean score | 8,046.30 |
| Mean of worst 100 scores | 5,192.56 |
| Minimum score | 1,097.00 |
| Successful harvests / attempts | 8,148 / 8,174 (99.682%) |
| Games with a failed delivery | 26 |
| Zero-harvest games | 0 |
| Sleeping targets / premature wakes / false ignores | 0 / 0 / 0 |
| Games surviving the entire 3,000 seconds | 0 |

This is harvest reliability, not perfect colony survival. A same-seed candidate
comparison against the 40001–41000 baseline is still finishing. The earlier native
baseline scored 6,920.09 on average and 3,348.31 over its worst 100, with a minimum
of 150.34 and five zero-harvest games. These are different cohorts from the fresh
candidate; use the paired comparison before attributing an exact improvement.

The original Python engine also passed a regression that applies the real
minimum-drain action list and keeps live bait beside the inert predator for the
full 3,000 seconds. The C++ controller passes the same 22 regressions locally and
on all four fresh-seed workers. Eight standalone C++ readiness regressions cover
early population, perception-probe limits and early fruit collection.

All 22 policy/engine regression tests pass. Six additional cohort-validation tests
verify that incomplete, overlapping, and mixed-source runs cannot be published as
a 1,000-seed result. The native public-observation boundary check also passes.

Delivery and sleep correctness are separate measurements. A predator entering
slow terrain can invalidate the two-move intercept estimate; the retry limit and
colony floor bound the resulting losses. Extra drain actions do not fix these
geometry misses or guarantee that the colony survives the entire horizon.

The lab's existing offline socket interlock remains engaged. The benchmark opens
no policy-serving listener. No competition endpoint or submission is changed.

The proposed lower-tail BO search space, parameter dependencies and required
configuration forwarding are in [the BO handover](harvest_bo_search_space_2026-09-20.md).
