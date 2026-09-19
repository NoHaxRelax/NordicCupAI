# Live research report

Open **http://127.0.0.1:8765** while the local dashboard server is running.
The page refreshes automatically; a background collector reads the existing
Runpod campaign over pinned SSH every 15 seconds. No public Pod port, MCP login
or Runpod API key is needed by the dashboard. The existing SSH key is used.

To start or restart from a VS Code terminal at the repository root:

```powershell
.\survival-simulator\scripts\research_dashboard.cmd
```

Leave that terminal open. Ctrl+C closes the website server without stopping
research. Optional: `--port 8766 --interval 30`. The browser and SSH polling
run on this computer; closing the browser does not close the server. If the
computer sleeps, polling resumes when it wakes. The campaign continues on
Runpod independently.

The dashboard shows:

- Generation/subgeneration, current steps and live simulation progress.
- Compute/setup estimates, reserved allowances, runtime and shutdown watchdog.
- Authoritative Python comparisons, separately identified C++ screening/control evidence.
- Cumulative LLM reviews, promotion decisions and research events.
- The latest 64 completed development runs: energy absorbed per fruit, energy
  costs, deaths, policy compute time and recorded screenshots loaded on demand.

The **Parallel capture studies** section shows the three additional Pods' study
status, candidate counts, active simulations, estimates and shutdown deadlines.
Their screening and Python comparisons also appear in the comparison selector.
The top cost estimate includes these Pods; the completed-run counter and detailed
diagnostic selector refer to the main campaign. See [parallel study details](research_parallel_studies.md).

The monitor never starts, stops or changes experiments. It exports only
development results and deliberately excludes final holdout reports and images.
The **Accepted mean survival** card uses the current accepted policy's latest
completed Python development comparison. It is not the highest screening mean:
a challenger with a higher mean can still fail the promotion checks. The card
shows the comparison timestamp and latest retain/promote decision. Its number
can stay unchanged while experiments and the feed continue updating. It changes
when a completed comparison updates that policy's estimate or a new best is
accepted. Incomplete comparisons and native screening cannot replace it.
There is no overall completion percentage because adaptive stopping and
variable run durations make one misleading. Step counts refer to that step.

If the Pod stops or SSH fails, an offline banner appears and the last successful
report remains available. Cached data can also be viewed after restarting the
local server while the Pod is offline. Only screenshots already viewed are
cached. The shutdown watchdog indicator and budget figures in an offline report
describe the last observation, not current cloud state. Check the Runpod console
for current Pod status and actual billing.

Local data is stored in the Git-ignored launch folder:

- `runs/runpod-launch-20260918/dashboard-latest.json`: last successful snapshot.
- `runs/runpod-launch-20260918/dashboard-images/`: downloaded development images.
- `dashboard-server.stdout.log` / `dashboard-server.stderr.log`: background
  server logs if started with those redirects.

**Export current report** downloads a JSON snapshot. Raw diagnostic event
streams, terminal clips and full campaign artifacts remain on the Runpod volume;
this dashboard is not a backup of the entire campaign.

The read-only collector is `/workspace/research-dashboard.py`, deployed outside
the frozen research source. A future Pod needs that helper copied from
`scripts/research_dashboard.py`, plus updated SSH host/port, campaign path and
pinned host key in the ignored launch configuration. The local launcher reads
`remote-state.json`; it does not create or resume cloud compute.
