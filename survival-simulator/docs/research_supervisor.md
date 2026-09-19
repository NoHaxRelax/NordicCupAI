# Major/subgeneration research supervisor

Status (2026-09-19): **implemented and running on Runpod**. The active campaign
now permits up to eight major generations, with four subgenerations per major
generation and adaptive early stopping. Its $40 budget, existing deadlines,
sixteen-review limit and untouched final holdout remain in force. See
[runtime controls](research_runtime_controls.md) for the audited override and
the correct launcher for resuming this campaign.

Failure-analysis collection is implemented; its supported evidence and remaining
limitations are in [Run diagnostics](run_diagnostics_plan.md). For current Runpod
setup, authentication, pilot and launch steps, use [the launch guide](runpod_research_launch.md).

## Workflow

`run.cmd research` routes to `scripts/research_loop.py`. The original `tune`
command remains available; `research` is a separate outer code-improvement loop.

1. Freeze the complete current simulator code, including untracked policy files,
   trusted tests, engine, evaluator, observation boundary and dependency pins.
2. Run trusted unit tests and full-horizon baseline/configurable-control games.
   Both must complete. Stop if any paired seed differs by more than 0.1 survival
   seconds or 1 score point; investigate timing-sensitive control differences.
3. Fetch a fixed Lucas branch snapshot and compare a compatible modular policy.
4. Run up to four subgenerations (`g1.1` through `g1.4`). Each gets a bounded
   agent review/edit, protected-source inspection, trusted checks, focused scalar
   search across explicit configuration alternatives, and paired comparison.
5. At the major boundary, fetch Lucas again, compare compatible changes, ask the
   agent to review direction/complexity and propose a bounded broader search,
   then freeze the code for Gaussian-process Bayesian optimization and compare
   the finalist on the expanded development set.
6. Start the next major generation from the retained best. Never promote merely
   because a new generation finished. By default, two unproductive subgenerations
   (after at least two) advance to the major review. Two major generations with
   no promotion anywhere in the generation end development. Agent early-stop
   recommendations also apply after the minimum subgeneration count.
7. Freeze `final-selection.json` before running any holdout game. Evaluate the
   original integrated baseline and selected policy on the untouched holdout.
   Produce a final report; do not invoke another coding job or change selection
   in response to holdout results.

The scheduling defaults are ceilings, not a throughput promise. Measured game
duration may permit fewer generations. Reserve final evaluation time before
starting each development step. Three auxiliary Pods were also allocated for
independent focused studies within the existing campaign budget.

## Preparing and running later

Execution has been authorized for the current task. These are reference commands.
From the repository root on Windows:

```powershell
.\survival-simulator\run.cmd research prepare --out logs/research-night-1
```

`prepare` copies source and templates only. It does not import policy code or
start tests, simulations, Git operations or Codex. Use a new output directory.
Edit the generated `config.json` before the first `run`:

- Set `budget.hourly_rate_usd` to the actual whole-machine quote. The template
  deliberately leaves it `null`; `run` refuses to guess an old cloud price.
- Keep or change `budget.total_usd` (default $25), external spending, reserve and
  total hours. Doubling the dollar cap alone does not change hardware/workers.
- Set a conservative `agent_call_reserve_usd` appropriate to the selected model
  and billing route; this is an accounting estimate, not a provider spending cap.
- Configure the `codex` executable/model and authenticate on the execution host.
  No model is forced by default. Authentication is not copied to policy workers.
- Check `evaluation.workers` (0 means conservative auto-sizing), deadlines,
  candidate counts and seed sets after authorized throughput measurements.

Once execution has been authorized:

```powershell
.\survival-simulator\run.cmd research run --out logs/research-night-1
.\survival-simulator\run.cmd research status --out logs/research-night-1
.\survival-simulator\run.cmd research stop --out logs/research-night-1
```

Repeat `run` to resume. An explicit `run` clears the campaign STOP request.
The Windows wrapper changes into `survival-simulator`, so `logs/...` is relative
to that task directory.
Ctrl+C also requests a cooperative stop. A paused child attempt may be consumed
as incomplete on resume; completed games remain usable. This is not a simulation
state checkpoint. Failed code, environment or control checks need investigation,
and changing frozen configuration/source requires a new campaign.

On a separately provisioned Linux machine, copy the **whole working tree**, not
just an old Git commit. From `survival-simulator`, the equivalent launcher is
`python run.py research ...`. For an authorized unattended run:

```bash
nohup /workspace/predator-search-venv/bin/python -u run.py research run \
  --out /workspace/research-night-1 > /workspace/research-night-1.console.log 2>&1 < /dev/null &
```

Install/authenticate Codex separately on that host. The supervisor invokes
`codex exec --sandbox workspace-write --json --output-schema ...` with an explicit
draft working directory and a saved prompt. The desktop app may connect to that
host over SSH to inspect artifacts and steer subsequent work. The detached
supervisor owns the state and can continue when the app disconnects.
See [non-interactive Codex](https://learn.chatgpt.com/docs/non-interactive-mode)
and [SSH projects](https://learn.chatgpt.com/docs/remote-connections#connect-to-an-ssh-host).

## Search and promotion

The default `evaluation.search_engine` is now `fastsim`, the C++ backend from
Oscar's branch. Build it with `python fastsim/build.py` before preparing the
campaign. Focused studies and major BO use it; all controls, paired promotion
comparisons, Lucas comparisons and the final holdout use the reference Python
engine. Configuration rejects a fast promotion/holdout engine. If a fast
screening winner does not improve on Python it is not promoted. Selecting
`search_engine: "python"` explicitly keeps an entirely Python campaign.

Both backends retain event/energy diagnostics and the isolated observation-only
worker. Native source, binary and build provenance are frozen in each candidate;
case identities include backend/build/ordering so native results cannot satisfy
a Python comparison through cache reuse. Native equivalence is conditional on
normalized object ordering and numerical runtime, as explained in
[Lucas/fastsim integration](lucas_fastsim_integration.md).

Every scored game uses natural predators, a 0.1-second native timestep and the
full 3,000-second horizon or extinction. A small test means fewer configurations
or maps, never a truncated survival objective.

Each subgeneration starts from the previous best settings. The agent can expose
new opt-in additions in one code revision and propose up to six labelled
configuration alternatives (for example A, B and A+B). The inherited configuration
is always included. Focused search mutates only the selected scalar settings;
it does not silently enable dormant feature blocks. Newly added configuration
fields use the new revision's defaults. Removed/carried fields are recorded in
`migration.json`; invalid migration rejects the candidate.

Major-boundary BO uses a fixed RBF Gaussian-process surrogate with standardized
survival objective, a noise floor and expected improvement. A diverse batch of
suggestions is selected from a sampled pool; the initial few suggestions provide
startup observations. Scalar selection defaults to at most 16 dimensions (hard
maximum 24). Variant differences are additional model covariates. This uses the
existing NumPy/SciPy dependencies, not Optuna or a new package. Unit checks pass;
there is no measured evidence yet that BO beats evolutionary search.
Surrogates are never reused across changed code. BO optimizes the current
survival rank; score and enabled-feature count break observed ranking ties.

Completed identical cases share a cache across tuning, comparisons and rounds.
Identity includes full frozen source, configuration, independent policy seed,
map seed, environment and evaluation protocol. Errors are cached as errors;
interrupted cases restart in unique attempt directories. A changed policy
revision cannot inherit another revision's measured outcomes.

Screening defaults to four maps, subgeneration comparison to eight, and major
comparison to twelve. Final holdout uses sixteen disjoint maps. The previous
best and challenger use identical maps and the same policy RNG seed. Promotion
requires a complete pair set, positive primary-rank gain, at least 10 seconds
mean survival gain, a positive lower paired-bootstrap bound, and no more than
30 seconds loss in worst survival. At full survival on every map, a positive
paired score bound can justify promotion. Thresholds are configurable before
the campaign. Bootstrap intervals are descriptive: small samples and adaptive
selection limit confidence. Equivalent complexity reductions are reported but
are not automatically promoted by this conservative initial rule.

The agent receives the cumulative journal, complete development comparisons,
search reports and the parameter inventory. Reports contain game/candidate IDs
for following links into cached results. Evaluator results now distinguish
policy child decision CPU/wall time from evaluator CPU, record RPC wall time and
normalize it per 1,000 agent decisions. Enabled diagnostics add native death and
food events, lifetime ledgers, observed inputs, decision stages and bounded
spectator clips. These are linked from individual result summaries. Missing or
truncated evidence must be acknowledged by the reviewer.

## Lucas branch handling

The supervisor creates its own bare mirror and fetches only
`survival-simulator/lucas-experimental`. It never pulls, merges, resets, commits
or pushes the user's checkout. Record both the previous reviewed and newly
fetched commit. Fetch/network failure is journaled; local work can continue with
that limitation, without pretending the branch is up to date.

Extract only bounded Python/JSON/Markdown policy/evidence files from the pinned
commit. Reject symlinks and unsafe archive paths. Engine/evaluator/dependency
changes from upstream are never installed. A compatible modular layout is
overlaid on a trusted snapshot, checked and evaluated as **Lucas modules on our
trusted harness**, not as an unmodified full-branch benchmark. Missing modules
may remain from our adapter; that distinction is recorded. If the upstream
entry point/layout or checks are incompatible, direct evaluation is skipped with
an explicit reason, and the agent can selectively adapt additions from the raw
source and diff. Every combined candidate still needs our paired evaluation.

Upstream reviews store dispositions and evidence references in the agent's
structured result and cumulative journal. Re-reading the same commit does not
imply all its changes were accepted; rejected/deferred imports remain visible.

## Integrity, resume and budget limits

- Candidates are independent content-addressed copies with verified manifests,
  not mutable references to the user's branch or hardlinks. Verify sources before
  and after execution. Preserve original and selected code plus configuration.
- Only current policy directories, `core.py`, `experimental_policy.py` and
  `experiment_config.py` are editable. Trusted tests, engine, evaluator, worker
  boundary, historical files and dependency pins are rejected if changed.
- This enforces source separation and accidental-leak checks; it is **not an OS
  security boundary against a deliberately malicious coding agent/policy**.
  Do not claim the prompt or manifests prevent arbitrary filesystem reads.
- Holdout seeds/results are omitted from agent review context. The supervisor
  schedules no holdout games until selection is frozen and schedules no tuning
  after that point. Reusing those maps in a later campaign requires an explicit
  decision about contamination; the software cannot infer cross-night history.
- State, decisions, subprocess results, source manifests and the journal persist.
  A campaign lock excludes a second supervisor. Saved step deadlines never reset
  on resume. Bounded subprocess wrappers detect lost parent heartbeats. After a
  comparison coordinator crash, stop the old attempt and allow its observation
  call timeout to expire before launching replacements.
- Total elapsed calendar time since the first run includes search, coding,
  checks, reporting and downtime. Provider spending before that point belongs in
  `external_spend_usd` or the reserve. Every agent invocation reserves its configured
  allowance, including failed invocations. Hard limits cover wall time, call count
  and step deadlines; dollar accounting remains an estimate. Subscription quotas
  and actual API charges must also be monitored at the provider.
- Final evaluation has a reserved minimum window; cancellation/cleanup may take
  additional time. An exhausted or incomplete campaign reports that limitation.
  There is no automatic retry loop, no unbounded agent retries and no new study
  that resets the outer budget.
- The supervisor never provisions, stops or deletes cloud resources. **Finishing
  Python does not end Pod billing.** Arrange infrastructure shutdown separately.
- Storage checks run at most ten seconds apart. The default campaign allowance
  is 14 GiB, with 5 GiB reserved for final assessment, 2 GiB minimum free disk,
  and additional headroom for all active workers to finish saving their bounded
  diagnostics. Reaching a limit ends development or produces an inconclusive
  final assessment. This is a cooperative guard, not a filesystem quota; external
  processes and provider storage billing require separate monitoring.

## Artifacts

| Path | Meaning |
| --- | --- |
| `config.json`, `protocol.json`, `state.json` | Frozen configuration/provenance and durable stage |
| `snapshots/<hash>/code`, `manifest.json` | Immutable candidate source and integrity evidence |
| `best.json` | Retained code snapshot plus settings; does not modify live policy |
| `reviews/<round>/` | Context, prompt, editable draft, raw proposal and code patch |
| `rounds/<round>.json` | Paired decision and previous/new best |
| `steps/<name>/` | Deadlines, requests, console, process results and comparison/search output |
| `evaluation-cache/cases/` | Shared full-game results and unique attempts |
| `upstream/<boundary>/` | Pinned source, diff, review status and comparison decision |
| `journal.md` | Cumulative development record, also stored structurally in state |
| `budget.json` | Calendar-time compute estimate and reserved agent allowance |
| `storage.json` | Campaign disk use, free space and reserved capacity |
| `evaluation-cache/cases/*/attempt-*/diagnostics/` | Events, series, agent ledgers, replay clips and screenshots |
| `final-selection.json`, `final-report.*` | Frozen selection and held-out assessment |

The 97 passing local tests include policy checks, snapshot protection, budgets
across downtime, paired promotion, cache identity, generation scheduling, holdout
closure, storage reserves and exact diagnostic accounting. A local 60-second
pilot passed all four cases. Full-horizon throughput and remote Codex execution
remain to be checked before the cloud campaign expands.
