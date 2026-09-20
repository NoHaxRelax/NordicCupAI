# Policy sweep fleet

Runs the `seed-aware-policy` bench across a RunPod fleet so a mode comparison can use
enough seeds to mean something.

**Why this exists.** The seed-aware-policy README rejects modes 27, 35-37, 39-42, 49 and 51
on **16 paired seeds**. Per-seed score sd is about 333, so 16 seeds gives a standard error
near 83 and can only detect effects above roughly 166. A genuine +50 gain would have been
invisible. At 1000 paired seeds the standard error drops to about 10.5, so those rejections
are worth re-testing rather than inherited.

Every arm runs the **same** seed set, so all comparisons are paired.

## Use

```sh
# sap.tar = git archive of origin/codex/seed-aware-policy-2200 survival-simulator/seed-aware-policy
python fleet.py provision 7     # bring pods up and bootstrap them
python fleet.py launch          # assign one arm-set per pod and start
python fleet.py status          # rows completed per pod
python fleet.py collect         # pull the csvs back
python fleet.py kill            # TERMINATE EVERYTHING -- always run this
```

## What bootstrap_pod.sh fixes

The bench does not build as committed. Four things:

1. `policy_bench.cpp` includes `../lucas_integration/native/_nengine.cpp`, a path that does
   not exist on that branch; the engine is at `../native/_nengine.cpp`.
2. `build.py` hardcodes `-lpython3.12`; the image ships 3.11.
3. Several `native/*.hpp` rely on `<map>`, `<set>` and `<array>` arriving transitively,
   which newer GCC does not do.
4. The replay writer emits **~114 MB per game**. At 33,000 games that is 3.7 TB. The sweep
   needs only `result.json`, so `capture()` becomes a no-op and `save()` writes the summary.

NumPy is pinned to **2.3.5**. The engine routes `sin/cos/arctan2/hypot` through NumPy's own
compiled loops, and the seed-aware README states that 1.26 results must not be pooled with
2.3.5. Absolute scores are therefore comparable only within one NumPy version; paired deltas
within a run are what matters.

## Two silent failure modes worth knowing

Both were hit on the first launch and both looked like success.

- **Embedding the job list in the ssh command line truncates it.** A 3000-line heredoc was
  cut to 1196 lines by the argv limit, so the terminator never arrived and nothing launched
  — while a naive status check still reported "running". `launch()` now builds `jobs.txt` on
  the pod.
- **`ssh` consumes the stdin of a `while read` loop**, so only the first pod in a loop gets
  its command. Use `ssh -n`.

Verify a launch with `pgrep -c policy_bench`, not with the launch command's exit code.

## Cost note

These use GPU pods because the toolchain image was already validated on one, but the
benchmark is **entirely CPU-bound** and the GPU sits idle. Eight 4090s cost roughly
$5.92/hr against maybe $0.50-1.50/hr for CPU-only instances of similar total core count.
Prefer CPU pods; the only reason to take a GPU box is that they come with 96 vCPUs each,
which CPU tiers may not match per instance.
