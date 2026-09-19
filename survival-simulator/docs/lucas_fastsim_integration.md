# Lucas policy and native engine integration

Integrated 2026-09-18 at the user's request. No unrelated branch merge, commit
or push was made. The original Python simulator and 39-file historical benchmark
manifest remain unchanged. The previous working source, including research and
telemetry, is preserved in `runs/integration-lucas-fastsim-20260918/before/`.

## Pinned sources

| Source | Revision | Use |
| --- | --- | --- |
| `survival-simulator/lucas-experimental` | `53f1a4c70862ded2e6cb570642e79cffef6a9c8b` | Integrated changes since `803bdd57`; raw policy/evidence exports retained locally |
| `survival-simulator/oscar-fastsim` | `696bbd86272c27d9edba56238d3ec3b9469faaf9` | C++ engine, Python adapter, build, verifier, benchmark and frozen verifier policy |

The remote heads were confirmed with `git ls-remote`. The initial commit objects
were available locally; later Lucas updates were fetched. Integration used
pinned Git blobs. The newer
Oscar no-predator Orchard tuning is a separate research candidate, not a proven
predator-enabled improvement; it was not silently substituted for the colony's
current survival module. The `fastsim/policy` copy is only a verifier fixture.

Lucas advanced from the first reviewed `1d7df707` during integration. A final
fetch reviewed through `53f1a4c7`: the only additional production change exposes
actual guide movement/sprint selection, energy and the native walking cap in
debug output. That change is included. The remaining commits are experiment
evidence, including a rejected visible-contact variant and preliminary renewal
and guide-energy diagnoses; their raw reports are preserved in `lucas-latest/`.

## Policy changes

- Bystanders also avoid fresh teammate predator sightings transformed through
  their estimated shared map. Stale, different-frame and already-local sightings
  are excluded. No actual world positions or native predator IDs enter policy.
- Guides are released after a tracked predator has remained near bait for ten
  seconds and is freshly observed. Release clears guide memory and completion
  state, and records a release event/counter.
- Certified corner pockets become a fallback when no regular gap qualifies.
  An occupied corner remains eligible when exploration later finds a crevice.
  Revalidation uses tighter position tolerance for corners. The existing optional
  `corner_pockets` feature considers corners alongside ordinary gaps as well.
- Guide routing supports a bait exclusion radius for the optional sight-based
  handoff experiment. The measured upstream default returns to the 55-unit stop
  after the existing following check. Its optional distant vision handoff stays
  disabled. Our motion-association adapter, tunable avoidance parameters,
  reproduction filters and outward decision traces are preserved.
- Guide debug now distinguishes requested intent from selected movement pace,
  including whether low energy caps movement at walking speed.
- The tuner now preserves both route-planner defaults (clearance and exclusion
  radius); a direct import otherwise broke default argument binding.

Lucas's pinned evidence reports the selected guide at 72/100 development cases
and 74/100 fresh map encounters in a crowded 30-plus-one-predator fixture. It also
records unsuccessful handoff alternatives. These are **upstream fixture results**,
not measured whole-game improvements of our combined policy. The newer defaults
replace the earlier stricter bait-hearing gate; its old unit expectation was
updated to test the intended measured-default contract.

The newest branch evidence also reports 319/380 successful encounters at
offline-selected sites and 19/20 maps exceeding half success. This measures site
potential after offline selection, not deployed site selection or whole-game
survival, and does not execute bait replacement. Its energy audit found 36/113
guide assignments already below the native sprint-energy threshold. Prioritize
guide viability and the coordinator's age-55 reproduction veto as next research
hypotheses; the cited renewal experiments are too small/confounded to auto-promote.

## Native engine integration

`fastsim` preserves the upstream engine and adds:

- Windows/MinGW support with statically linked compiler runtime; Linux/macOS
  builds remain supported. Build on the execution host with
  `python fastsim/build.py`. Generated `.so`/`.pyd` files are not committed.
- `build-info.json` with C++ source/binary hashes, compiler and Python/NumPy
  versions. Changed or missing builds fail explicitly rather than falling back
  to another backend. The verifier now exercises its generated movement angles,
  fixing an upstream test bug that always passed zero.
- Optional native accounting events for movement, turning, reproduction,
  maintenance, births, gross/absorbed food energy, cap waste, fruit spawns/rot
  and death causes. These are read-only observations of operations and never
  affect state transitions or RNG. Stable predator keys are evaluator-only.
- An adapter to the existing diagnostic collector, terminal/event clips and PNG
  renderer. Policy decisions still run in a separately spawned JSON-only worker;
  the worker's import check now rejects `fastsim` as well as Python engine modules.

`research_config.json` selects **C++ for focused and BO screening** and **Python
for initial controls, Lucas comparisons, promotion and the final holdout**.
Configuration rejects native promotion/holdout. Cases include backend/build
identity, so a native cache entry cannot satisfy a Python evaluation. Frozen
candidates include native source, build metadata and the current local binary;
portable source bundles omit binaries/metadata and rebuild on the destination.
The supervisor protocol is version 3 and telemetry version 2; prepare a new
campaign rather than resuming an earlier protocol. A pilot also freezes its
source before evaluation.

The native engine replaces address-based Python set ordering with creation-counter
ordering. Lockstep verification patches only the reference process's object hashes
to match that order. Exact agreement under this condition is valuable evidence,
but it does not prove identical trajectories to arbitrary stock Python allocation
orders or across CPUs/NumPy builds. All final assessment therefore stays on Python.

## Validation and performance

- The full local suite passed **105 tests**, including immutable native source,
  separate backend cache identity, Python-only promotion/holdout and the native
  accounting verifier. Historical source hashes remain unchanged.
- Three random-action development seeds (0, 1, 2), each starting with predators,
  passed exact observation, world-state and RNG comparison on every tick until
  extinction (23.0, 30.1 and 33.6 simulated seconds). Engine steps were about
  **44–49 times faster** in this small Windows test. This is engine-only timing.
- Eight real 60-second colony cases completed: baseline/control/variant with
  recorded and unrecorded controls across Python and C++. Every case reached
  score **63.501** with **15 agents**. Recorded cases agreed on mean absorbed
  fruit energy (**41.1882353**) and wrote terminal clips and screenshots.
- Representative recorded baseline/control times were **33.6–35.3 seconds in
  C++ versus 49.3–50.1 seconds in Python** (about 1.4–1.5x overall). These single
  smoke pairs shared the laptop and are indicative, not a stable throughput or
  overhead benchmark. The conservation variant is inactive at this early time.
- A **1,200-tick / 120-second** Orchard-driven predator-enabled lockstep passed
  every world/RNG check and cumulative/per-agent accounting checks. It exercised
  **45 births, 278 meals, 64 fruit rots and 29 predation deaths**, including
  **640.37 energy of cap waste**. This is engine/telemetry verification with a
  fixture policy, not a colony performance claim.
- Four further five-second cases passed from a frozen source/binary snapshot.
  A native diagnostic PNG was visually inspected.

Evidence is in `runs/integration-lucas-fastsim-20260918/`: upstream exports,
before-source manifest, `verify-random.jsonl`, `verify-accounting.jsonl` and
`pilot-fast`/`pilot-python` reports. Those initial pilot cases ran before the new
pilot snapshot wrapper; no policy or engine behavior was edited during them.
Subsequent pilots preserve exact source snapshots.

Full-horizon remote throughput, a completed generation and whole-game policy
improvement remain unmeasured. No final holdout has been opened. Runpod account
funding and remote Codex authentication still need resolution before launch;
see [the current launch guide](runpod_research_launch.md).
