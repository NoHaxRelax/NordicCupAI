# Runpod research launch

Updated 2026-09-18. The user authorized implementation, tests, a development
pilot, infrastructure setup and the bounded optimization launch. Current state:
115 tests passed locally and remotely after the launch fixes. The supervisor is
running at `/workspace/research-night-1`; last verified in generation 1.1
preflight with eight development cases active. Do not start another copy.
The user increased the
campaign budget to **$40** on 2026-09-19 (Paris), retaining the $5 reserve and
nine-hour duration cap. No final-holdout game has run. The account is funded and
Pod `emkmq3v2dd25rn` is provisioned in `EU-RO-1`, with network volume `u9kre0adae`.
The allocated `cpu3c` has 32 vCPUs/64 GB RAM at $0.96/hour compute; `cpu5c`
allocation failed despite its catalog stock indication. The full-horizon native
and Python development pilots completed. Remote Codex is authenticated and the
shutdown guard is armed. Read-only Codex tool execution passed; the writable
sandbox cannot run on this Pod. The campaign therefore uses validated JSON
proposals applied by the trusted supervisor, as described below.

## Which application to use

You can do this from the **Codex VS Code extension** with the connected Runpod
MCP. The ChatGPT desktop app is optional. MCP manages the Pod and storage; native
SSH/SCP transfer files and run commands. The long-running research supervisor
executes on the Pod and invokes the Codex CLI there. Keeping the chat open is
not what schedules generations.

The local MCP OAuth connection does not authenticate the remote Codex CLI.
Authorize Codex on the Pod with device login; do not paste credentials into chat
or copy local authentication directories into the bundle.
[Codex MCP](https://learn.chatgpt.com/docs/extend/mcp),
[Codex authentication](https://learn.chatgpt.com/docs/auth).

## 1. Provision the execution host

Use one Secure Cloud `cpu5c` CPU Pod with **32 vCPUs and 64 GB RAM**. The live
catalog quote read for this task was **$1.12/hour compute**. Recheck the actual
quote and regional stock immediately before creation. A GPU is unnecessary.
The current launch instead uses `cpu3c` at $0.96/hour after `cpu5c` allocation
failed. Its source and evidence are in ignored `runs/runpod-launch-20260918/`.

Use a **20 GB standard network volume**, mounted at `/workspace`, and a **10 GB
container disk**. The CPU Pod API rejected a persistent Pod mount: this requires
a network volume. The standard volume quote is $1.40/month for 20 GB; storage
persists and remains billable after compute stops. The selected compatible
region was `EU-RO-1`, subject to a fresh availability check.
[Storage pricing](https://docs.runpod.io/pods/pricing).

The verified official CPU image is `runpod/base:1.0.7-ubuntu2404`, pinned as:

```text
runpod/base@sha256:7e45ffb69c0625b22c3fe90886b3192e95e0b9ea44d0861333a874ce37a07187
```

Enable SSH, expose `22/tcp`, disable Jupyter and retain the image's normal
entrypoint. A dedicated public key was registered during preparation. Its
private key is local in ignored `runs/runpod-launch-20260918/id_ed25519`; it is
not in the source bundle. Use the Pod's actual direct TCP SSH host and forwarded
port. VS Code Remote SSH needs direct TCP SSH, not only the basic Runpod proxy.
[Runpod IDE connections](https://docs.runpod.io/pods/configuration/connect-to-ide).

## 2. Transfer the actual working tree

The current work includes many untracked files and intentional deletions.
Cloning the old remote revision would miss this implementation. A verified bundle of the Lucas/C++ integration is already prepared at
`runs/runpod-launch-20260918/research-source-launch-v5.tar.gz`. Use that
file for this launch. To rebuild after further edits, use a new output filename
with the helper below from the local `survival-simulator` directory:

```powershell
.\.venv\Scripts\python.exe -B scripts/research_bundle.py --out runs/runpod-launch-20260918/research-source-next.tar.gz
```

The bundle includes Python/JSON/shell source, tests, dependency pins and docs.
It excludes `runs`, virtual environments, Git metadata and private keys. The
helper verifies every archived source hash and writes an adjacent bundle report.
Use a new bundle filename after further edits.

Transfer with SCP, replacing `HOST` and `PORT` using the Pod's actual connection:

```powershell
scp -i runs/runpod-launch-20260918/id_ed25519_user -P PORT -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=runs/runpod-launch-20260918/known_hosts runs/runpod-launch-20260918/research-source-launch-v5.tar.gz root@HOST:/workspace/research-source.tar.gz
ssh -i runs/runpod-launch-20260918/id_ed25519_user -p PORT -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=runs/runpod-launch-20260918/known_hosts root@HOST
```

On the Pod, extract into a fresh destination and add a remote for Lucas reads:

```bash
set -e
test ! -e /workspace/NordicCupAI
tar -xzf /workspace/research-source.tar.gz -C /workspace
cd /workspace/NordicCupAI
git init
git remote add origin https://github.com/NoHaxRelax/NordicCupAI.git
cd survival-simulator
bash scripts/runpod_setup.sh
```

Setup installs pinned dependencies into `/workspace/predator-search-venv` using
Python 3.12 and compiles the native engine. The image must have a C++17 compiler
and Python headers; if missing, install `build-essential` before setup. It
launches no experiments. Do not overlay an older checkout or
replace simulator files with Lucas engine code.

## 3. Install and authenticate remote Codex

On the Pod:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

Follow the installer's PATH instructions, then:

```bash
codex --version
codex login --device-auth
codex login status
```

Open the displayed verification URL on your own device and enter the temporary
code. Device login may need enabling in account settings. ChatGPT subscription
quotas and API billing are separate; select the intended account route before
starting the campaign. A successful remote `codex exec` must also be checked:
the installed CLI and its Linux sandbox must actually work in the container.
If sandboxing fails, diagnose it rather than disabling safeguards blindly.
[CLI install](https://learn.chatgpt.com/docs/cli),
[device authentication](https://learn.chatgpt.com/docs/auth).

## 4. Run the development pilot and calibrate

From `/workspace/NordicCupAI/survival-simulator`:

```bash
/workspace/predator-search-venv/bin/python -B -m unittest discover -s tests -p 'test_*.py'
/workspace/predator-search-venv/bin/python -B fastsim/verify.py --seeds 0 1 2 --horizon 300 --starting-predators 2
/workspace/predator-search-venv/bin/python -B scripts/verify_native_diagnostics.py --seed 0 --steps 300
/workspace/predator-search-venv/bin/python -B -u scripts/research_pilot.py --out /workspace/research-pilot-smoke --seconds 60 --workers 1
/workspace/predator-search-venv/bin/python -B -u scripts/research_pilot.py --engine fastsim --out /workspace/research-pilot-fast --seconds 60 --workers 1
/workspace/predator-search-venv/bin/python -B -u scripts/research_pilot.py --engine fastsim --out /workspace/research-pilot-full --seconds 3000 --workers 4 --case-timeout 3600
```

The pilot uses only development seed 0 and separate output/cache identities.
It compares unrecorded baseline, recorded baseline, configurable control and
one configuration variant. Short pilots test plumbing, never promotion. The
60-second local results matched at score 63.501 with 15 agents; this does not
establish a policy improvement. The conservation variant is inactive that early.

Inspect `report.json`, per-case console logs and diagnostics. All full-pilot
cases must finish normally; investigate baseline/control divergence, recording
failures and timeouts. Use full-game wall times, peak memory and worker throughput
to set campaign deadlines. A single timing pair does not establish overhead or
predict linear speedup when doubling workers. The default deadlines are ceilings
to calibrate, not guarantees of four completed subgenerations.

## 5. Prepare and launch the supervisor

```bash
/workspace/predator-search-venv/bin/python -B run.py research prepare --out /workspace/research-night-1
```

Before the first run, edit `/workspace/research-night-1/config.json`:

- Enter the actual whole-Pod hourly quote, including a conservative storage
  allowance, in `budget.hourly_rate_usd`. It is deliberately unset by default.
- Keep the $40 total, $5 reserve and maximum nine-hour campaign unless the user
  changes the agreed budget. Enter setup/pilot spending in `external_spend_usd`.
- Set workers and preflight/search/comparison/final deadlines from the remote
  pilot. Preserve all development/holdout seed separation and the full horizon.
- Keep four subgenerations per major generation, focused trial cap 12, broader
  BO cap 48, and adaptive stopping. Caps can produce fewer completed trials.
- Configure the remote `codex` path if needed. The default $1 reservation per
  agent call is conservative bookkeeping; actual account spending/quotas must
  also be monitored. Three major generations are an upper limit, not a promise.
- Keep bounded diagnostics and campaign storage reserves. A 20 GB volume also
  holds the environment and pilot; download pilot artifacts if more room is needed.
- Keep `search_engine: "fastsim"` for focused/BO screening and
  `comparison_engine: "python"`, `holdout_engine: "python"` for authoritative
  comparisons. Native and Python results have separate cache identities.

Source, configuration, environment, prompt and schema freeze on the first run.
Prepare a new campaign after changes; do not mutate a running study.

For this Pod set `agent.proposal_only: true` and `agent.legacy_landlock: true`.
Codex CLI 0.155.1's writable bubblewrap sandbox needs user namespaces denied by
this container. Its Landlock fallback supports read-only execution here. Codex
reads evidence and returns literal policy edits in the response schema; the
supervisor validates paths, exact matches, edit count and size before applying
them to a new draft. Previously frozen candidates stay untouched. It does not
disable sandboxing or run generated shell commands outside the sandbox. Plugins,
apps and delegated agents are disabled for these bounded research calls.

The calibrated launch settings use 16 workers, a conservative $0.965/hour
including storage, 30-minute focused-search windows, 40-minute comparisons,
60-minute BO windows, one-hour individual case limits and a two-hour final
evaluation reserve within the nine-hour campaign. Trial counts and spending
limits remain upper bounds. These are conservative pilot-based starting values,
not a measured claim of optimal 16-worker scaling.

Before detaching, establish who will monitor billing and stop the Pod. The
supervisor stops its own work on limits but **does not stop provider billing**.
In this chat the connected MCP can stop the Pod we create after the process
finishes, but it is not an independent scheduled shutdown service.

```bash
nohup /workspace/predator-search-venv/bin/python -B -u run.py research run --out /workspace/research-night-1 > /workspace/research-night-1.console.log 2>&1 < /dev/null &
```

The supervisor first checks tests and full-horizon controls. It then fetches
Lucas, runs up to four subgenerations, reviews/imports compatible Lucas changes
at major boundaries, performs broader BO, and retains only supported improvements.
The final holdout opens once selection is frozen; it never determines promotion.

### Control equivalence on the remote host

Python's simulator hashes game objects by memory address. Identical seeds in
separate processes therefore need not give identical trajectories; telemetry
can also change allocation order. Full Python controls are still required and
recorded, but their scores cannot establish exact code equivalence. With native
screening enabled, the separate `initial-control-equivalence` step compares
baseline and configurable control under repeatable native ordering. It cannot
select a winner or evaluate final holdout. Independent Python comparisons remain
the only promotion evidence. Python-only screening retains the strict control
check and may fail this gate; use the validated native setup for this campaign.

### Independent Pod shutdown guard

`scripts/runpod_shutdown_guard.py` calls the actual Runpod v2 Pod stop API. It
checks both Pod ID and creation timestamp before mutating anything, stops on
terminal campaign status or a fixed deadline, and logs no credentials. Use a
Runpod API key with Pod read/write permission in a private file outside the
source/campaign directories, for example `/root/.config/nordiccup/runpod-api-key`.
The Pod-injected credential returned HTTP 403 on this launch and is insufficient.
The local MCP OAuth login does not grant unattended scripts its credentials.

Create `/workspace/research-shutdown.json` with `pod_id`, `created_at`, absolute
Unix `deadline_epoch`, and the campaign directory in `campaign`. The deadline
must be within 12 hours of Pod creation. Do not include a key in that JSON.

```bash
python3 /workspace/NordicCupAI/survival-simulator/scripts/runpod_shutdown_guard.py \
  --config /workspace/research-shutdown.json \
  --key-file /root/.config/nordiccup/runpod-api-key \
  --journal /workspace/research-shutdown.jsonl --check
# After the identity/credential check passes, arm before starting research:
nohup python3 /workspace/NordicCupAI/survival-simulator/scripts/runpod_shutdown_guard.py \
  --config /workspace/research-shutdown.json \
  --key-file /root/.config/nordiccup/runpod-api-key \
  --journal /workspace/research-shutdown.jsonl \
  > /workspace/research-shutdown.console.log 2>&1 < /dev/null &
```

Confirm the `armed` journal event and live guard PID before detaching. This guard
survives SSH loss and supervisor failure, but requires its container and access
to Runpod's API. It is not a provider-hosted schedule. Verify `EXITED` through
MCP after it fires. Stopping releases compute; storage remains billable and
results remain available. An unarmed guard is not a completed shutdown setup.

## 6. Monitor, stop and retrieve

The remote Codex reviewer uses ChatGPT subscription authentication. The verified
g1.1 session used `gpt-6-astra`; reasoning effort was not explicitly configured.
On 2026-09-19, the user requested Astra Extra High. Remote
`/root/.codex/config.toml` now pins `model = "gpt-6-astra"`,
`model_reasoning_effort = "xhigh"`, and `forced_login_method = "chatgpt"` for new
reviews. No OpenAI API key is installed in its authentication file or supervisor
environment. The last observed review usage record had `has_credits=false` and
credit balance `0`. These are observations at verification time, not a live
account billing guarantee. The $1-per-review budget reserve is a conservative
planning allowance, not a measured API charge. Runpod compute/storage are billed
separately. The three parallel simulation Pods make no LLM calls.

The settings change and its verification receipt are recorded at
`operator-guidance/astra-xhigh.receipt.json` in the campaign. Immutable experiment
configuration, snapshots and scoring were preserved.

For the auto-updating HTML report, open **http://127.0.0.1:8765** while the local
dashboard is running. Restart it with
`.\survival-simulator\scripts\research_dashboard.cmd` from the repository root.
It includes live progress, budget estimates, candidate comparisons, LLM reviews
and development-run diagnostics/screenshots. See [dashboard instructions](research_dashboard.md).

From a VS Code terminal at the local repository root, run:

```powershell
.\survival-simulator\scripts\watch_research.cmd
```

The view refreshes every 15 seconds and shows the current generation/stage,
completed development cases, active simulation progress, the selected best's
Python development statistics when available, estimated spending, supervisor
checkpoint age and the shutdown watchdog. Ctrl+C closes the monitor without
stopping research. Add `--once` for a single status report. The last successful
view is also saved in `runs/runpod-launch-20260918/monitor-latest.txt`.

The read-only remote helper is `/workspace/research-monitor.py`, outside the
frozen campaign source. It never reads final holdout results. If SSH becomes
unavailable after shutdown, check the Pod status in Runpod and use the saved
local view; the monitor does not restart compute. Research reasoning is in the
campaign's `journal.md` and `reviews/*/proposal.json` on the volume.

```bash
/workspace/predator-search-venv/bin/python -B run.py research status --out /workspace/research-night-1
tail -f /workspace/research-night-1.console.log
```

Read `journal.md`, `budget.json`, `storage.json`, `best.json` and individual
`steps`/`reviews` directories. To pause gracefully:

```bash
/workspace/predator-search-venv/bin/python -B run.py research stop --out /workspace/research-night-1
```

Wait for the process to exit, download the campaign directory and console log,
then stop its Runpod compute through MCP or the console. Do not delete storage
before retrieving results. Resuming uses the same `run` command and keeps old
deadlines/budget accounting. See [supervisor contracts](research_supervisor.md)
and [diagnostic artifacts](run_diagnostics_plan.md).
