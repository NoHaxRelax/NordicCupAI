# Lucas branch access on the running campaign

On 19 September 2026, a live check found that the opening Lucas fetch had failed:
the Pod lacked GitHub credentials. A step marked `finished` only meant its process
had exited; `upstream/g1.opening/review.json` correctly recorded `unavailable`.

Local authenticated access confirmed that the latest branch revision was still
`53f1a4c70862ded2e6cb570642e79cffef6a9c8b`. This revision was already integrated into
the campaign's starting policy; see [integration details](lucas_fastsim_integration.md).

`scripts/research_upstream_mirror.py` now provides access without copying GitHub
credentials to the Pod:

1. The local bridge fetches `survival-simulator/lucas-experimental` using the
   existing local Git authentication. It updates only its remote-tracking ref;
   it does not switch branches or merge the working tree.
2. It sends a Git bundle over pinned SSH. The first bundle contains the branch's
   history; subsequent updates use incremental bundles when possible.
3. The Pod verifies the branch name and commit, imports the bundle into
   `/workspace/lucas-bridge/upstream.git`, and extracts a pinned models/docs
   snapshot with a file-hash receipt. Transport bundles are removed on the Pod
   after successful verification.
4. The Pod repository's `origin` now points to that mirror. The original URL is
   preserved in `/workspace/lucas-bridge/original-origin.json`. The existing
   major-boundary fetch/comparison code can read the mirror normally.
5. Future LLM reviews are instructed to inspect
   `/workspace/lucas-bridge/current.json`, its actual GitHub check time and its
   pinned source directory. Imports still require the normal paired evaluation.

Refresh runs every five minutes while the local bridge and computer are running.
It stops when the campaign leaves development, after ten hours, or on an error.
If the computer sleeps or the bridge fails, the last verified mirror remains
available but may be stale. Its timestamp must not be described as a fresh GitHub
check. Restarting the bridge resumes synchronization without restarting the
campaign or rerunning simulations.

Local process status, logs and PID are stored in the Git-ignored launch folder:
`lucas-bridge-status.json`, `lucas-bridge.stdout.log`, `lucas-bridge.stderr.log`,
and `lucas-bridge.pid`. From the repository root, a foreground restart is:

```powershell
survival-simulator\.venv\Scripts\python.exe -B survival-simulator/scripts/research_upstream_mirror.py local --repo . --folder survival-simulator/runs/runpod-launch-20260918
```

The failed opening review, campaign protocol, candidate snapshots, previous best
and final holdout were not rewritten. The repair was verified with a real
bundle transfer/import, matching GitHub/Pod branch heads, pinned source export,
and unchanged hashes of the campaign configuration, prompts and source manifests.
