# Handover: survival simulator optimization and Codex research loop

Updated 2026-09-19 (Paris). This document describes the local working tree, including
uncommitted work. Start here before changing or running the simulator.

**Current operational status supersedes the original launch details below:**
the first campaign stopped; its recovery resumed at 06:54 UTC on Pod
`a0aruuldun08t9`. Read [the recovery handover](research_recovery_20260919.md)
and [the latest upstream refresh](upstream_refresh_20260919.md) first.

## Immediate instructions for the next agent

The user wants a repeatable overnight workflow: run useful predator-enabled
experiments for at most a few hours, produce a report, have a coding agent
diagnose failures and improve the policy, evaluate the changes, and repeat.
The user increased the overnight Runpod campaign budget to **$40**, with a $5
reserve and a nine-hour campaign limit.

**Execution is now authorized.** The later user instruction "start working on
all of that now" superseded the earlier preparation-only restriction. It covers
diagnostic implementation, tests, Runpod setup, a development pilot, and launching
the bounded research campaign once prerequisites pass. Do not ask again for the
same authorization. The final holdout remains unopened during pilots.

**The remote supervisor is running**, last observed at generation 1.1 preflight
with eight Python development cases active. Do not launch a duplicate campaign.
Pod `emkmq3v2dd25rn` (`cpu3c`, 32 vCPUs/64 GB, $0.96/hour) uses persistent network
volume `u9kre0adae` in `EU-RO-1`. Campaign: `/workspace/research-night-1`.
SSH: `root@213.173.105.101:30450`, dedicated key in ignored
`runs/runpod-launch-20260918/id_ed25519_user` (normal Windows user owns it).
Read `remote-state.json` in that directory for launch identifiers and artifacts.

115 tests passed locally and remotely. Both four-case full-horizon seed-0 pilots
completed with natural predators; no final holdout was used. Native baseline,
recorded baseline and configurable control all survived 514.7 seconds. Python
baseline repetitions differed (794.7 versus 596.6 seconds): object-address
iteration makes independent Python runs nonidentical. Protocol version is now 4:
full Python controls remain, exact baseline/control equivalence is a separate
native-only check, and only Python comparisons can promote or evaluate holdout.

Codex 0.155.1 is authenticated with ChatGPT. This Pod denies the user namespaces
needed by its writable sandbox. The tested alternative is **read-only proposal
mode with legacy Landlock**; Codex reads evidence and returns exact text edits,
which the supervisor validates and applies to a fresh draft. A real structured
proposal smoke test passed; workspace sandboxing was not disabled.

The separate shutdown watchdog is armed (PID 2642), stops on terminal campaign
status or launcher exit, and has a hard deadline **2026-09-19 07:31 UTC / 09:31
Paris**. Its private API key is outside the campaign/source directories. It uses
the provider stop API, but depends on the Pod and API connectivity; verify the
provider's `EXITED` status afterwards. Network volume/storage remains billable.
The controller wrapper is PID 4117. Budget accounting uses $0.965/hour including
storage, $0.55 setup allowance, 16 workers, and a two-hour final reserve. The
launched archive is `research-source-launch-v5.tar.gz`, SHA256
`aff206289870e9f9b3109d2aae148f699e90b7810c2893831ba08ac4357f6241`.
See [the current launch guide](runpod_research_launch.md) for operations.

For a live read-only update from the local repo root, use
`survival-simulator\scripts\watch_research.cmd --once` (omit `--once` for a
15-second refreshing view). The helper runs at `/workspace/research-monitor.py`
outside frozen source. Latest verified status at 22:10 UTC was running preflight,
6/8 initial Python controls completed, two active, $0.68 estimated compute/setup,
watchdog running and holdout unopened. Later agents must fetch live status.

The working branch is **`sim-optimization`**, created from
the existing `challenge1_nikolaj` checkout with all working-tree changes kept.
No commit or push was made for this handover or the latest integration.

The research supervisor and bounded diagnostics are implemented and locally
tested. Start with [the supervisor guide](research_supervisor.md): four
subgenerations per major generation, adaptive stopping, Lucas fetch/comparison
at major boundaries, focused tuning, broader GP Bayesian optimization, frozen
candidates, paired promotion, a cumulative journal, budgets and final holdout.
`scripts/research_loop.py` is the outer supervisor; `optimize_policy.py` remains
the separate inner evolutionary optimizer. The subsequent user request explicitly
authorized integrating new Lucas changes and the faster C++ environment. Lucas
is now integrated through `53f1a4c70862ded2e6cb570642e79cffef6a9c8b`, with local
observation association, tuning and diagnostics preserved. The native engine is
from `survival-simulator/oscar-fastsim` at `696bbd86`, with local build/telemetry
adapters. It is used for research screening, while Python remains mandatory for
promotion and final holdout. Read [the latest integration record](lucas_fastsim_integration.md)
for measured checks, qualifications, backup and packaging details.

## Repository and working-tree state

| Item | Value |
| --- | --- |
| Repository | `C:\Users\nikol\OneDrive\Dokumenter\GitHub\NordicCupAI` |
| Task directory | `survival-simulator/` |
| Current branch | `sim-optimization` |
| Previous branch | `challenge1_nikolaj` |
| Base HEAD | `282446421263e7bb11081d692f0ac443b99f9e20` |
| Base commit subject | `Merge native predator-trapping MVP into challenge1_nikolaj` |
| Remote | `https://github.com/NoHaxRelax/NordicCupAI.git` |
| Local shell | PowerShell on Windows |
| Existing Python | `survival-simulator\.venv\Scripts\python.exe` |
| Planned remote environment | Linux, Python 3.12, persistent `/workspace` |

There is a large intentional cleanup in progress: before writing this document,
Git reported 4,387 tracked status entries, plus many untracked new files.
Most retired files are cleanup deletions. Current policy modules, search
scripts, tests, and documentation include untracked files. **A clone of HEAD
does not contain the current implementation.** Preserve both tracked edits
and untracked additions when packaging, committing, or transferring it.

Do not use `git reset --hard`, `git clean`, restore the old tree wholesale, or
apply an old stash indiscriminately. Do not delete files simply because they
appear to be duplicate policies: some are frozen benchmark dependencies.
Keep unrelated medical fine-tuning and other challenge folders outside scope.

Read [AGENTS.md](../AGENTS.md): behavior belongs in `models/`, tools in
`scripts/`, and documentation in `docs/`. Prefer scoped `rg`, Git diffs, and
status summaries because whole-repository output is very large. Git metadata
writes may require sandbox escalation; creating this branch was authorized.

Local recovery material, ignored by Git:

- `runs/cleanup-backup-20260918-185556/`: retired code and local results,
  preserving repository-relative paths.
- `runs/branch-integration-20260918/before/`: pre-integration versions.
- `runs/branch-integration-20260918/lucas-experimental.zip` and `incoming/`:
  source snapshot of the first imported Lucas revision, `6bebca41`.
- `runs/branch-integration-20260918/lucas-803bdd57.zip` and
  `incoming-803bdd57/`: later navigation/corner updates.

Do not assume ignored backups are present in a future clone or on Runpod.

## User intent and project history

The project began with distributed mapping: agents connect sightings into
clusters, align shared coordinates using observed boundaries, track stones,
traps, fruit and biomes, and estimate Voronoi biome borders. Shared absolute
coordinates and survival/population became more important than complete
coverage or perfectly timed fruit harvesting.

There were many Simple/Expert policy, population, conservation, fruit-ripening,
and Bayesian optimization experiments. The user later explicitly chose the
better-performing `oscar-orchard-research` policy and asked to discard the
earlier policy direction. That research folder was consolidated during cleanup.
**Do not revive the retired Simple policy, old BO/HPC scripts, or old population
targets as the current specification.** The maintained Orchard module is the
successor to that research code.

The user then requested the predator-trapping MVP be integrated, giving that
integration precedence where code overlapped. They wanted the predator-free
Orchard runner preserved, ordinary agents to avoid lured/trapped predators,
and a comprehensive optimizer for additional behaviors and existing settings.
The optimizer must test **only natural-predator games**. The no-predator
runner remains a supported capability, not a mode for this search.

Persistent priorities:

- Survival is the main optimization priority; aim for the full 3,000 seconds.
- Keep policy changes understandable and avoid unnecessary complexity.
- Use only information available through agent observations at runtime.
- Preserve the integrated trapping, shared map, Orchard survival, and rendering.
- Make interrupted searches useful and resumable.
- Produce reports with measured findings, clearly separated from hypotheses.

## Current policy and entry points

| Path | Responsibility |
| --- | --- |
| `run.py`, `run.cmd` | Supported command router; Windows wrapper selects the existing venv and correct directory |
| `models/core.py` | Current `EntrapmentPolicy` coordinator |
| `models/exploration/` | Public-observation localization, shared map, exploration, biome estimation, navigation |
| `models/survival/oscar_orchard.py` | Orchard food collection and reproduction |
| `models/entrapment/` | Observed trap geometry, guide tracking/pathfinding/steering, predator following, bystander avoidance |
| `models/observation_only.py`, `models/observed_bounds.py` | Spawned Orchard adapter, public-input sanitization, inferred arena bounds |
| `models/experiment_config.py` | Defaults, feature switches, parameter inventory, bounds and validation |
| `models/experimental_policy.py` | Optional extensions around the current coordinator |
| `models/experiment_actor.py` | Spawned JSON policy worker for current baseline and experiments |
| `scripts/trapping_game.py` | Current trapping recording runner |
| `scripts/run_orchard.py`, `scripts/playback.py` | Orchard headless/native-window runner and speed controls |
| `scripts/entrapment_viewer*` | Browser playback of recorded trapping games |
| `agent_server.py` | `/predict` endpoint using the current coordinator |
| `src/` | Native simulator, kept unchanged |

`run.py trapping`, `serve`, and the tuning baseline use `models.core.EntrapmentPolicy`.
The no-predator adapter imports the relocated Orchard module. The trapping
coordinator overrides Orchard defaults with `extra_old=False`, `heir_age=1e9`;
legacy exploration harvesting and planner post-alignment population control
are disabled to avoid competing controllers.

**Historical reference only:** `models/entrapment_policy.py`,
`models/oscar_orchard.py`, `models/nikolaj/`, the old flat guide/trap modules,
and `scripts/entrapment_game.py`. These remain for `run.py benchmark` and its
39-file frozen manifest, `entrapment_native_seed0/manifest.json`.
Do not edit them to improve the current policy. See [models/README.md](../models/README.md).

## Latest branch integration

Remote survival branches were fetched and compared on 2026-09-18. The imported
production lineage is `origin/survival-simulator/lucas-experimental` at
**`803bdd57b860416d6ee189495c2b93e59f0ad6e3`**. It advanced during review from
`6bebca41`, and the later changes were included. This was selective source
integration into the working tree, not a whole-branch merge or new commit.
Do not describe this dated snapshot as necessarily the latest remote forever.

Other reviewed branches:

- `survival-simulator/entrapment-9059-benchmark` (`dfef3c5a`): preserve frozen
  benchmark; import viewer completion/incompletion labels.
- `survival-simulator/entrapment` (`91437ef0`): earlier guide work already in
  the integrated lineage.
- `survival-simulator/oscar-orchard-population` (`f6ab0357`): Orchard source
  is the same Git blob as the retained implementation,
  `7c2b4b45d2d14d264f77b3bab92769754c7ebb26`.
- `survival-simulator/oscar-trapper` (`2c4b1195`): older alternative Society/
  TrapManager integration with observed and oracle research paths; not imported.
- `survival-simulator/entrapment-AI-attempt` and `lucas-trap-slopsesh1`
  (`df3ef96d`), and `codex/survival-trap-handoff-2026-09-17` (`6a21ffec`):
  older experiments/handoffs, not newer production coordinators.

Important behavior and local integration fixes:

1. Ordinary agents steer away from observed predator contact/hearing/vision
   zones and the estimated bait area, check observed walls, suppress births
   while avoiding, and limit sprinting with low energy. Guides/bait use their
   dedicated controllers. Parameters are exposed under `bystander.*`.
2. Guides associate observations over time so a held predator does not simply
   replace the newcomer being delivered. The coordinator now passes
   `associate_target=True`; previously its explicit target bypassed this path.
   Initial target hints are filtered against the held group. Observation
   ambiguity remains; there are no hidden predator identities.
3. Guide delivery/sacrifice requires the selected predator to be sensed near
   the bait, as well as guide arrival. Reaching the destination alone is not
   sufficient evidence of delivery.
4. A displaced guide can use agent-width navigation to rejoin a route with
   predator clearance, instead of getting stuck outside the wider lane.
5. Normal trap gaps now accept widths 10.1-19.9. Optional corner-pocket sites
   check contact exclusion and rear access. They default off and supplement
   normal gaps through `include_corner_pockets` / `--corner-pockets`.
6. Production runner, API, observation adapters, and optimizer imports now
   select the modular policy. The recorder has an independent `--policy-seed`.

All 39 historical manifest files were statically hash-checked and unchanged
during integration. Current regression tests subsequently passed in the 97-test suite.
See [policy_branch_integration.md](policy_branch_integration.md) for details.

## Observation boundary and evaluation integrity

The policy may receive sanitized official per-agent observation DTO fields,
simulation time, its configuration, and an independent policy RNG seed.
DTO schema imports are allowed; importing the simulator state implementation
into the isolated worker is not.

The search runs the simulator in an evaluator and the policy in a fresh spawned
process connected by JSON. The worker clears inherited command-line arguments
(which can contain the world seed), removes exploration config environment
overrides, checks the observation whitelist against the official DTO schema,
and checks for prohibited simulator modules before/after decisions.

Do not pass the world seed, true absolute positions, true fruit ages, hidden
predator IDs, simulator objects, or evaluator scores into the policy. Relative
and shared positions must come from observations. Cross-agent danger estimates
must respect map group, frame revision, uncertainty, and observation age. The
two map implementations are bridged through a shared agent's local frame;
their global coordinates cannot simply be assumed identical.

Evaluator-only telemetry may use true outcomes for offline diagnosis, but it
must never become runtime policy input. The JSON boundary prevents accidental
leaks; it is not an OS sandbox against deliberately malicious policy code.
The direct recording runner and replay have spectator data, so preserve the
separation between recording/evaluation and policy decision inputs.

## Existing optimizer: implemented but unexecuted

The current algorithm is **block evolutionary search**, not Bayesian
optimization. It requires no third-party optimizer. See
[policy_optimization.md](policy_optimization.md).

- `scripts/optimize_policy.py`: candidate generation, shared game queue,
  caching, provenance, search/refinement/validation, reports, progress, resume.
- `scripts/experiment_case.py`: one full game in an evaluator subprocess,
  separate policy worker, cancellation and result writing.
- `scripts/search_resources.py`: CPU affinity/cgroup quotas and available RAM
  for conservative worker sizing.
- `scripts/runpod_night.py` / `.sh`: existing single-study budget wrapper.
- `scripts/runpod_setup.sh`: setup only; creates a persistent Python environment.

Eleven optional feature keys, all off by default:

| Feature | Purpose |
| --- | --- |
| `consistent_escape` | Configurable radial escape trigger alongside native bystander avoidance |
| `escape_memory` | Continue escape briefly after losing a sighting |
| `shared_danger` | Recent observed predator tracks and estimated lure paths |
| `trap_exclusion` | Avoid the entrance/goal area of an established baited trap |
| `safe_steering` | Compare alternative moves against threats and observed walls |
| `risk_aware_food` | Release/filter dangerous fruit and tree assignments |
| `role_budget` | Reserve workers and limit/release guide and bait assignments |
| `refresh_guide_geometry` | Refresh paths/caches after new observed geometry |
| `emergency_reproduction` | Bounded rescue births when young population is low |
| `late_conservation` | Gradually reduce idle turning |
| `corner_pockets` | Add observed short corner traps to normal gap candidates |

All additions off still includes the current baseline's native bystander
avoidance. Disabling that avoidance is a separate `bystander.enabled` choice.

The initial plan has 25 configurations: baseline, configurable control,
11 individual additions, all additions, and 11 leave-one-out combinations.
Then it probes every tunable non-feature scalar and uses small block mutations,
occasional crossover, and feature toggles from the best completed parents.
Inventory covers active Orchard, exploration/planner, navigator, guide,
bystander and new behavior settings. Simulator physics and hidden state,
mapping invariants, dormant harvesting, and unread parameters are excluded.

Ranking is lexicographic:

1. `mean_survival / horizon + 0.25 * worst_survival / horizon`.
2. Mean simulator score.
3. Fewer enabled optional features.

Only natural-predator `trapping` games are evaluated. Every scored game reaches
extinction or the requested horizon (normally 3,000 simulated seconds, native
0.1-second timestep). Partial games are not scored as complete. Errors/timeouts
make the candidate ineligible; a failed control stops search expansion. No
winning configuration is automatically installed as the live policy.

| Default | Laptop | Runpod |
| --- | --- | --- |
| Phase | `search` | `overnight` |
| Wall hours per optimizer session | 8 | 9 |
| Game workers | 1 | 0: auto-size |
| Screening seeds | 0, 1 | 0-3 |
| Refinement seeds | 0-11 | 0-11 |
| Validation seeds | 101, 102, 103 | 1001-1016 |
| Evolutionary candidates after initial comparisons/probes | 64 | 1,024 |
| Finalists | 6 plus controls | 6 plus controls |

Independent policy seed defaults to 0; search seed defaults to 123. A game's
wall-time timeout defaults to 3,600 seconds. Candidate limits are ceilings,
not throughput promises. `--trials` means total candidates, not additional ones.

## Resume, provenance, and artifacts

Completed games are cached. Interrupted games restart from their seed; there
are no mid-game state snapshots. Saved errors/timeouts are also reused rather
than silently retried. For a failed control, fix the cause and use a new study
directory.
OS locks prevent concurrent coordinators writing one study. Unique attempt
directories preserve partial logs and results after a coordinator crash.

Resume compares protocol version 3, source hashes, all installed dependency
versions, Python/OS/architecture, seeds, horizon, objective, and parameter
inventory. The source manifest includes `models/`, `src/`, and `scripts/`
Python files, scripts' shell files, models' JSON, `run.py`, and requirements.
Even an orchestration-script edit changes provenance. **Use a new output
directory after code, environment, or protocol changes.** Copying a Windows
study to Linux does not make it resumable under the strict environment check.

General `--hours` is per session; wrapper `--max-hours` is cumulative wrapper
runtime. In `overnight`, approximately 60% goes to screening, half the remainder
to refinement, and the remainder to validation. Early completion carries time
forward. A phase deadline may advance with unfinished planned work: refinement
can select the screening winner and mark the comparison incomplete; if that
winner has a known refinement error, selection falls back to baseline. Ctrl+C
instead leaves the current stage. **Resuming overnight resumes the saved
stage; it does not necessarily finish an earlier, timed-out refinement.**
Once `validation-selection.json` exists, search/refinement is blocked for that
study to prevent tuning against opened holdout results.

Progress is written about every 10 wall seconds per game and printed about
every 30 seconds by the scheduler. Evaluators check cancellation/heartbeat;
an abandoned coordinator is detected after roughly 90 seconds plus any pending
policy call. Linux cancellation also handles evaluator process groups.

Usual output roots: `survival-simulator/logs/policy-search/` locally and
`/workspace/predator-search/` on Runpod.

| Files | Meaning |
| --- | --- |
| `study.json`, `report.md`, `best.json` | Candidates, training ranking, selected configuration |
| `feature-effects.json`, `coverage.json` | Feature comparisons and successfully varied settings |
| `protocol.json`, `default-config.json`, `parameter-inventory.json` | Provenance, baseline settings and search space |
| `resources.json`, `progress.json` | Hardware allocation and live batch status |
| `refinement-plan.json`, `refinement.json` | Frozen shortlist and expanded training comparison |
| `validation-selection.json`, `validation-progress.json`, `validation.json`, `validation-report.md` | Frozen choice, held-out results, paired deltas and descriptive bootstrap intervals |
| `cases/<case_id>/` | Cached result and individual attempt request/result/progress/console logs |
| `control/<uuid>/heartbeat`, `STOP` | Coordinator liveness and cancellation |
| `study.lock`, `budget.lock`, `budget.json` | Study/wrapper exclusion and estimated active compute spending |

Equal source and seeds do not guarantee identical outcomes: map fitting already
has wall-clock-adaptive behavior. Control CPU load and document variability.

## Runpod preparation and limits

Recorded proposed hardware: one CPU5 Compute-Optimized (`cpu5c`) Pod,
32 vCPUs, 64 GB RAM, no GPU, 20 GB persistent `/workspace`, 10 GB container disk.
This workload is CPU Python/NumPy/SciPy/geometry code; there is no CUDA path.
The 2026-09-18 catalog quote was $1.12/hour for that shape; CPU3 Compute was
$0.96/hour. These are dated quotes, not a guaranteed current price or evidence
that CPU5 has better throughput per dollar. Recheck availability/price when
deployment is authorized. See [runpod_policy_search.md](runpod_policy_search.md).

The worker-sizing estimate is 27-30 simultaneous games, with two CPU slots
and four GiB reserved and a two-GiB-per-game planning allowance. Neither actual
throughput nor memory use has been measured. Each game also has its own policy
process. Numerical-library threads are capped at one per process.

Setup targets `/workspace/predator-search-venv` with Python 3.12 and pinned
requirements. If `uv` is unavailable, setup bootstraps `uv==0.8.22` using the
image's Python/venv support. No installer or Pod was launched during preparation.

Current wrapper defaults: $25 allowance, $5 reserve, $1.12/hour, nine cumulative
active hours; estimated spending checkpoints every 15 seconds. It does not
measure setup, idle time, storage, tax, other Pods, or downtime between sessions.
**Finishing or stopping Python does not stop Pod billing.** The wrapper never
creates, stops, or deletes a Pod. A future supervisor needs whole-night
accounting and an explicit end-of-night Pod lifecycle plan; retained storage
also needs separate consideration. Do not provision anything yet.

Important CLI distinction: `runpod_night.sh` forwards arguments to
`runpod_night.py`, whose parser accepts only `--out`, `--budget`, `--reserve`,
`--hourly-rate`, `--max-hours`, `--already-spent`, `--workers`, `--describe`.
It does **not** forward arbitrary optimizer flags such as `--phase`,
`--seconds`, `--trials`, or seed lists. Those require direct optimizer invocation,
which bypasses this wrapper's cumulative spending ledger.

## Codex loop: design background and implemented preparation

The original design below is retained as context. The implemented contract,
commands, limitations and current validation status are now maintained in
[research_supervisor.md](research_supervisor.md). In particular the new schedule
has four subgenerations per major generation and fixed source snapshots; the
old single-study wrapper is still separate. Whole-machine hourly rate must be
set explicitly before `research run`; the supplied template leaves it null.

The agreed design is an external Python supervisor running on the persistent
Runpod workspace. It invokes Codex CLI after each experiment/report, so the
user's laptop can be disconnected. A starting allocation is up to 1-2 hours
of experiments per cycle, about 20 minutes of coding, and explicit comparison
time, aiming for approximately three cycles within a nine-hour overall limit.
These are proposed caps; calibrate after authorized throughput measurements.
Reserve final evaluation time before starting another cycle.

Suggested state machine:

```text
baseline/checks -> bounded search -> diagnostic report -> Codex candidate
     -> correctness checks -> paired comparison -> promote or retain
     -> next bounded cycle -> final holdout evaluation -> final report/stop
```

Required implementation pieces:

1. **Persistent supervisor state.** Save round, stage, deadlines, cumulative
   budget, best version, candidate version, attempts and report locations
   atomically. Handle interrupts and failed subprocesses. Do not depend on an
   LLM remembering state or waiting for hours. Preserve completed work on resume.
2. **Immutable code plus configuration.** A best version consists of a saved
   code revision/snapshot, config, dependencies and manifest. Give Codex a
   separate candidate checkout. Freeze code while simulations use it. Each
   changed revision needs a fresh study directory; do not reset the outer
   budget by creating a new inner study.
3. **Cross-version comparison.** Existing validation compares against
   `models.core.EntrapmentPolicy` in the same checkout. That is insufficient
   when Codex edits the baseline. Evaluate the previous frozen best and new
   candidate with the same trusted engine/protocol/maps. Promote only with
   adequate evidence; retain the best version on errors or inconclusive results.
4. **Useful short rounds.** Add an explicit starting configuration from the
   previous best and a configurable round plan. Repeating all 25 initial
   comparisons before every coding round could consume the night. Focus
   subsequent tuning on affected settings while keeping regression controls.
5. **Better diagnostics.** Existing results contain scores, coarse population
   samples, final feature/trapping counters and errors. Add simulation-time
   traces for energy/food/population, reproduction, role allocation and trapping
   outcomes, plus death causes where the evaluator can establish them. Include
   representative failure logs and links per seed. Distinguish measured causes
   from hypotheses. Current samples occur every 10 wall seconds, not equal
   simulated intervals; `evaluator_cpu_seconds` excludes the policy child.
   Add decision latency/child CPU evidence before blaming a performance bottleneck.
6. **Experiment discipline.** Development and regression maps can inform the
   coding agent. Keep a separate final holdout until coding ends. Once feedback
   from a holdout informs changes, it is development data, including across
   subsequent nights. Use complete paired cases and report uncertainty. A short
   wiring check is not evidence of 3,000-second survival.
7. **Constrained coding scope.** Ask Codex for one or two evidence-based changes,
   preserve a cumulative research journal, and save its patch/explanation.
   Enforce protected engine, scoring, observation boundary and evaluation rules
   outside the editable candidate; a prompt alone is not enforcement. Trusted
   tooling changes can be prepared separately before experiments are frozen.
8. **Whole-night budget and termination.** Include coding/check/report/idle time
   in Pod accounting, plus a separate Codex usage allowance if using API billing.
   Bound coding attempts and retries. If too few complete games finish, report
   inconclusive evidence instead of manufacturing a winner. Save artifacts
   before the planned infrastructure shutdown.

Implemented files now include `scripts/research_loop.py`, `research_support.py`,
`research_study.py`, `research_bo.py`, `research_process.py`, `research_config.json`,
and `docs/research_agent_prompt.md` / `research_agent_result.schema.json`.
The evaluator now records policy child CPU/wall time and normalized RPC timing.
`tests/test_research_supervisor.py` contains passing contract tests. Existing
optimizer helpers gained optional frozen-root/shared-cache hooks without changing
ordinary tuning defaults. `runpod_night.py` remains the single-study wrapper.

Official Codex documentation was checked during planning:

- [Non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode):
  `codex exec`, explicit `--sandbox workspace-write`, JSONL events with `--json`,
  final output with `-o`, optional `--output-schema`, and stdin prompts. The
  current docs prefer explicit sandbox settings over deprecated `--full-auto`.
- [Authentication](https://learn.chatgpt.com/docs/auth): API-key automation is
  usage-billed separately from Runpod; ChatGPT sign-in has account entitlements
  and limits. Headless device authentication is available where enabled.

Do not assume this chat's login is installed on the Pod. Configure remote
authentication deliberately without committing secrets or exposing them to
simulation processes. Model choice, authentication route, exact coding budget,
remote runtime verification still need completion. Promotion and stopping
criteria are implemented and configurable before the campaign freezes.
No particular model was selected by the user. Recheck CLI capabilities against
the installed version when setup/execution is authorized.

## Verification status and first steps

**Latest local validation: 97 tests passed and a four-case 60-second seed-0
pilot completed.** Baseline, instrumented baseline, configurable control and
an early inactive conservation variant all reached the pilot horizon with 15
agents and score 63.501. About 5.5 MB of diagnostic artifacts were saved for
each recorded case; a PNG was visually inspected. There is no full-horizon
improvement, stable overhead estimate, remote throughput result or completed
cloud campaign yet. Runpod funding and remote Codex authentication remain blockers.

Earlier cleanup checks are documented in [cleanup.md](cleanup.md), including
49 passing tests and short runner checks. Those predate the modular integration
and optimizer. Some cleanup text also describes the old module layout; current
layout is defined by the README/module guide and integration document.
Do not reuse those old checks as validation of today's code.

Historical results only:

- Original Orchard runs: score 2,834.033 / survival 2,659.2 seconds on seed 1;
  score 2,726.8146 / survival 2,547.3 on seed 3; score 2,037.9156 / survival
  1,902.3 on seed 5, all with predators disabled. See
  [orchard_reference_results.json](orchard_reference_results.json).
- Original trapping seed-0 reference went extinct at 1,073.6 seconds and had
  up to 11 predators near bait for a 30-second interval. Proximity does not
  prove permanent trapping. See [entrapment_9059_baseline.md](entrapment_9059_baseline.md).

Current passing tests include `test_current_policy.py`,
`test_current_guide_delivery.py`, `test_current_guide_steering.py`, and
`test_current_integration.py`. The latter covers association wiring, held-group
confusion, sensed delivery, blocked-start recovery, optional corners, bystander
behavior and configurable-control defaults. Orchard observation/bounds tests
were adjusted to import the current module; historical tests still serve the
frozen reference.

Next agent should first read this document and the linked source guides,
inspect the scoped working tree, and continue the authorized setup. Once funding
is available, validate remote tests and full-horizon baseline/control wiring
before spending hours on experiments. Measure actual
throughput, then calibrate workers and round lengths. Keep natural predators
enabled; a short test may simply not reach their natural spawning time.

## Existing commands for later authorized use

These older commands are reference instructions. The current authorized launch
path is [runpod_research_launch.md](runpod_research_launch.md).
The new `run.cmd research` command is documented in the supervisor guide. Its
`prepare` and `run` commands have deliberately not been invoked.

From the repository root in PowerShell, record current trapping, then start
its browser viewer in a separate terminal after frames have been written:

```powershell
.\survival-simulator\run.cmd trapping --seed 0 --policy-seed 0 --seconds 3000 --out logs/trapping-live
.\survival-simulator\run.cmd view logs/trapping-live --port 9059
```

Open `http://localhost:9059`. Cyan marks guides, yellow bait, orange replacement
bait. The viewer can only play frames already recorded. Closing it does not
stop the simulation. Use a fresh output directory for another recording.
`--corner-pockets` is an optional flag on `trapping`, disabled by default.

Current headless laptop search (predators enabled; repeat unchanged to resume):

```powershell
.\survival-simulator\run.cmd tune --out logs/policy-search --hours 8 --workers 1
```

The retained no-predator graphical runner, outside the optimizer's scope:

```powershell
.\survival-simulator\run.cmd orchard --seed 1 --seconds 3000 --output logs/orchard-view --render --speed 5
```

It has 1x/2x/5x/10x/20x controls; omit `--render --speed 5` for a headless score.
Use the wrapper from the repository root to avoid the earlier nested-folder
and incorrect `..venv` path mistakes. The retired research directory is no
longer the launch location. `run.cmd benchmark` runs the historical policy,
not the current integrated one.

On a user-provisioned Linux Pod with the complete working tree copied over,
existing setup and single-study overnight launch are:

```bash
cd /workspace/NordicCupAI/survival-simulator
bash scripts/runpod_setup.sh
setsid bash scripts/runpod_night.sh > /workspace/predator-search-console.log 2>&1 < /dev/null &
echo $! > /workspace/predator-search-launcher.pid
```

For monitoring and graceful cancellation:

```bash
tail -f /workspace/predator-search-console.log
kill -INT "$(cat /workspace/predator-search-launcher.pid)"
```

This is the existing parameter search, not the proposed Codex loop. See the
Runpod guide for environment overrides and budgeting limits. Ending it does
not stop the Pod. New directories, source changes and environment changes have
the resume implications described above.

## Suggested prompt for the next agent

> Read `survival-simulator/docs/agent_handover.md`, its linked current guides,
> and `survival-simulator/AGENTS.md`. We are on `sim-optimization` with extensive
> intentional uncommitted cleanup and new source files; preserve them. Continue
> implementing and launching the predator-only Runpod experiment/report/Codex-improvement loop
> around the current modular policy. Keep the native simulator and public
> observation boundary intact, save code plus configuration for every candidate,
> compare against the previous best, and reserve unseen maps for final evaluation.
> The approximate budget is $25 for one night; execution was authorized by
> "start working on all of that now". Local tests and the short pilot passed.
> Resolve the recorded funding/authentication prerequisites, measure remote
> full-horizon throughput and launch within the budget. Keep the final holdout
> untouched until selection freezes. Distinguish measured outcomes, hypotheses
> and unfinished checks in every status report.
