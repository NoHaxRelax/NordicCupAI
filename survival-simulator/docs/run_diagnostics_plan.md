# Run diagnostics design

Status (2026-09-18): **core instrumentation implemented and locally validated**.
The specification below also includes future analytical extensions. The selection
objective remains full-horizon survival with the existing promotion rules.

The later C++ integration supports the same collector through measured native
events, with telemetry schema version 2. Native per-step and per-agent accounting
was checked against Python for 1,200 predator-enabled ticks. See
[Lucas/fastsim integration](lucas_fastsim_integration.md) for the current 105-test
validation, backend-specific provenance, and longer food/death checks. Python
remains the promotion and final-holdout engine.

## Implemented evidence and limitations

`scripts/research_diagnostics.py` runs only in the evaluator. It wraps native
engine calls without editing engine source. Native call sites distinguish fruit
consumption from rot and predation from energy depletion; changed engine call
structure fails collection explicitly. Maintenance combines passive and aging
costs rather than claiming an unmeasured split. Decision debug exports leave the
policy worker; spectator truth never enters it.

Each recorded attempt writes:

| Artifact | Contents |
| --- | --- |
| `manifest.json`, `static.json` | Collection settings, version, engine hash, seed references, immutable geometry and frame contract |
| `events.jsonl` | Births, role changes, native deaths, individual fruit meals and rot |
| `series.jsonl` | One-simulation-second population, energy, food, role and policy summaries |
| `agents.json`, `summary.json` | Lifetime ledgers, role time, meal gaps, mean gross/absorbed fruit energy and cap waste |
| `recent.jsonl.gz`, `checkpoint.json` | Atomic recent-history checkpoints every ten simulated seconds |
| `terminal.jsonl.gz` | Up to the last 30 seconds of every tick: pre-action world, public inputs, chosen actions, policy debug and ensuing events |
| `event-*.jsonl.gz` | Up to three clips around bait loss or multiple deaths, including ten seconds after the trigger |
| `frame-*.png` | Offline spectator diagrams from recorded terminal states near T-30, T-10 and T-0.1 |

Defaults bound replay memory to 64 MiB and each attempt's diagnostics to 128 MiB.
Truncation counts and captured spans are explicit. If essential collection fails,
the attempt is ineligible rather than silently promoted without evidence. Abrupt
process termination can lose data since the latest checkpoint. Collection timing
excludes engine-hook and worker-export overhead; it is not a total overhead
measurement. Screenshots are schematic world diagrams, not rendered native-game
frames; biome shading, target/path overlays, close-ups, a sparse-clip interactive
viewer and automated causal classifications are not implemented.

Five diagnostic tests passed for consumption/rot/cap waste, native death causes,
fixed-action engine and RNG equivalence, durable bounded tails, and cache identity.
The complete local suite passed 97 tests. A real four-case, 60-second seed-0 pilot
also completed, and a saved PNG was visually inspected. Each case ended with 15
agents and score 63.501. Recorded baseline/control produced identical outcomes.
This early smoke test does not measure policy improvement, predator survival or
stable instrumentation overhead; the tested conservation variant is inactive
at this early stage. Each recorded case saved about 5.5 MB. Full-horizon remote
pilots are still required for capacity planning. See `runs/research-local-pilot-1/`.

## What a recording must explain

For each run, distinguish the immediate failure from its contributing conditions:
which agents died, what depleted their energy or exposed them to predators, what
the policy perceived and chose, and which proposed feature actually changed an
executed action. Record successful runs too, including near failures. A plausible
timeline is a hypothesis about causation until a controlled comparison supports it.

### 1. Compact history for every run

Sample on **simulation time**, initially once per simulated second, independently
of the existing wall-clock progress heartbeat. Save timestamps and tick numbers.
Keep whole-run aggregates and per-agent lifetime summaries, including:

| Area | Evidence |
| --- | --- |
| Population | Alive count by role, births, parent/child IDs, deaths and measured causes, age distribution, lifespan, time without a replacement bait or guide. |
| Energy | Energy/max-energy quantiles and minimum by role; time below the native low-energy movement threshold; food income and movement, turning, passive, aging and reproduction expenditure where directly measurable. |
| Food | Fruits eaten, gross fruit energy, energy actually absorbed, cap waste, time since last meal, food income per living-agent-second, fruit spawns/rot, local food availability and fruit lost while being targeted. |
| Movement | Requested movement versus actual displacement, energy spent without progress, target switches, path failures, blocked duration, escape duration and uncertainty in estimated position. |
| Predators and traps | Predator pressure, attacks, bait continuity, arrivals/departures near traps, guide deliveries and losses, escapes after apparent containment. Proximity alone is not proof of trapping. |
| Policy | Time in each phase/role, feature activation counts, actions changed by each feature, override/conflict counts and reasons, fallback usage, unavailable/stale targets. |
| Compute | Policy CPU and elapsed time, RPC time and latency quantiles, agent decisions, evaluator time, diagnostic collection/serialization time, bytes written and memory limits. Normalize policy cost by agent decisions and population where appropriate. |

Keep food averages separate:

- Gross energy per fruit = sum of fruit energy at consumption / fruits eaten.
- Absorbed energy per fruit = sum of actual energy increases at consumption /
  fruits eaten. A full agent may consume a fruit with little benefit.
- Cap waste = gross minus absorbed energy.
- Absorbed energy per living-agent-second = absorbed energy / integral of alive
  population over simulation time. Also report by role and early/middle/late game.
- Report counts and distributions alongside means. With no meals, the per-fruit
  mean is null, not zero. An average across survivors must not hide agents that died.

Use actual measured engine quantities. Current fruit `age` advances at twice
simulation time; store birth/consumption timestamps in seconds separately. Aging
increases energy drain; it is a contributing mechanism, not a separate direct
death event in the current engine. Track declining food supply over the run so
late resource scarcity is distinguishable from poor food selection.

### 2. Exact event records

Record compact events at their actual tick: birth, death, fruit consumption/rot,
role change, bait loss/replacement, guide delivery, trap departure, navigation
failure and safety override. Include involved stable IDs, relevant before/after
values, policy role/intent, and references to detailed frames when retained.
Scope IDs to a run; never assume equal IDs correspond to equal agents across
different policies, even on the same seed.

World state before/after a whole step is insufficient to establish every event:
energy costs, reproduction, multiple meals and predation can occur in the same
step. A vanished fruit may have rotted. Do not label ambiguous deltas as measured
consumption or invent a predator cause from a disappearance alone.

Use trusted evaluator-side tracing around existing engine operations, delegating
to their unchanged implementations. Preserve random calls, execution order and
physics. Where exact tracing is unavailable, retain an explicit unknown or
inferred label. Do not edit the pinned engine to make a policy look better.

### 3. Bounded detailed clips

Start with a rolling **30 simulated seconds** of per-tick data (300 ticks at the
native timestep), retaining at least the final 10 seconds. Keep a cheaper history
for the entire run: starvation and resource collapse can begin minutes earlier.

Retain the terminal clip for extinction, horizon completion, interruption or
error, plus up to three non-overlapping event clips per run. Suggested triggers
are rapid population loss, loss of an established bait, a trap departure followed
by attacks, sustained failure to move or feed, and an exception. Where the run
continues, include 10 seconds after the trigger. Tune trigger thresholds only
using development runs and record their configuration.

Each detailed frame should contain:

- The exact official observations supplied to the policy and returned actions,
  including action validation errors if present.
- Bounded outbound policy debug data: phase, role, estimated position and
  uncertainty, selected target, intended controller, reason codes, proposals
  overridden, and final action owner. Capture reason codes at decision points;
  do not synthesize them after failure.
- Evaluator-only world state: true positions, agent traits/energy/age, predator
  state, fruit identity/energy and trees. Save static terrain/obstacles once.
- Tick/time, events and links to the immutable candidate, configuration and run.

True world state and policy-visible state must remain separate. Hidden predator
IDs, absolute positions, fruit ages and world seeds must never enter the policy
worker. Debug exports travel outward only. Avoid copying complete policy memory
or repeatedly serializing large static geometry.

Checkpoint bounded recent chunks periodically and flush on handled failures.
A RAM-only ring buffer is lost on a killed worker. Record the last durable tick
and any lost interval; never claim a complete tail after an abrupt kill. Complete
chunks and metadata need atomic writes; incomplete files must be identifiable.

### 4. Screenshots and replay

Render images **after the run from recorded states**, initially for selected
regressions, champions and novel failure types. Save an overview and a close-up
around the decisive event, with energy, roles, targets and paths. Useful frames
include roughly 30 seconds before, 10 seconds before and immediately before the
event; a picture after extinction alone is rarely useful.

Clearly label spectator truth versus the agent's observations/estimated map.
Screenshots supplement the numerical evidence and action trace. They need not
be generated for every optimizer trial. Do not rerun a game and present that
trajectory as the original: time-dependent policy behavior can affect replay.

`scripts/trapping_game.py` already demonstrates world serialization and compressed
replay chunks. Reuse suitable serialization logic inside the isolated evaluator;
do not replace it with the direct-policy recorder. The current
`scripts/entrapment_viewer.py` assumes contiguous, absolute tick indexing, so
sparse event clips require an explicit clip index/viewer adaptation.

## How the supervisor and LLM use the evidence

Give the reviewer a compact development report with links to raw evidence:

1. Paired candidate/incumbent outcome changes, uncertainty and relevant seeds.
2. The common failure modes and their frequency, with explicit classification
   rules and unknown cases. Separate immediate death cause from contributing
   food, energy, navigation, coordination or predator conditions.
3. A few representative timelines, including successes and counterexamples.
4. Changes in absorbed food income, expenditure, role continuity and executed
   feature activations. Compare the shared simulation interval as well as full
   lifetimes; longer survival itself changes lifetime totals and compute.
5. Evidence-backed hypotheses and the smallest useful ablation or combination
   test. Distinguish observed association from demonstrated feature effects.

Example hypothesis, **not an observed result**: a safety override repeatedly
rejects nearby food; energy falls below the movement threshold; slower movement
precedes predation. Check observations, override events and energy history, then
compare an isolated change to that override on paired development cases.

Add aggregate findings and evidence references to the cumulative research journal
after each subgeneration. Keep raw case artifacts immutable. Diagnostics explain
research direction; survival and the established promotion rules remain primary.
Holdout recordings are not available to development or review jobs. Inspect them
only after final selection, with no further tuning against that holdout.

## Budgets, provenance and implementation order

Store telemetry schema/version, tracing configuration, source/config hashes,
simulation and policy seeds, runtime/dependency versions, hardware/worker settings,
termination reason, completeness and artifact hashes. Include telemetry mode in
case identity/cache compatibility; a cached summary cannot supply a missing clip.
A diagnostic rerun needs its own identity and must be labelled as a new execution.

Set per-run memory/disk limits and a campaign disk limit before collection.
Always reserve capacity for outcomes, summary/events and terminal evidence;
reduce optional clips/screenshots first. Record any truncation and stop launching
new work if essential recording cannot be preserved. Diagnostic work counts
against campaign compute/time budgets, including final evaluation reserves.
Measure overhead before promising a recording rate or storage size.

The implementation order was:

1. Fixed-simulation-time summaries, exact food/death accounting and event schema.
2. Bounded terminal/event clips and outward policy decision evidence.
3. Offline screenshots/replay and paired diagnostic reports for the supervisor.

Before expanding cloud optimization, measure full-horizon overhead and throughput.
Instrumentation can change timing-sensitive policy behavior even when fixed-action
physics are unchanged. The implemented tests and smoke pilot establish the
contracts above; they do not substitute for that longer development pilot.
