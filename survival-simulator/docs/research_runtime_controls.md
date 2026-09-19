# Generation ceiling and BO parallelism

The user requested a maximum of **eight major generations**, retaining four
subgenerations per major. `scripts/research_config.json` now uses eight for newly
prepared campaigns. Early stopping, time, storage, financial limits and the
review-call cap still apply; eight is an upper bound, not a completion promise.

For the already running `research-night-1` campaign, the historical configuration
and protocol remain immutable. `scripts/research_runtime_controls.py` applies a
separate, audited override to the supervisor's in-memory scheduling settings.
Only `schedule.major_generations` may change. The override binds to the original
protocol checksum and the external runtime's checksum, and rejects stale or
unrelated changes. The trusted evaluator and source manifests are reverified.

The deployed runtime is `/workspace/research-runtime/research_loop.py`; the
immutable amendment is
`/workspace/research-night-1/operator-controls/max-generations-8.json`.
Successful activation is recorded in `runtime-controls-applied.json`, campaign
state and the cumulative journal. The dashboard uses the applied value from
campaign state, rather than displaying an unactivated amendment.

`scripts/research_supervisor_handoff.py` performs a bounded coordinator handoff
only while an independently launched search or LLM review wrapper is running.
It refuses an inline comparison, final evaluation, stale heartbeat or near-expired
step. It suspends and rechecks the old coordinator, replaces only that coordinator
and its launcher, and leaves the independent wrapper and simulations running.
Saved step deadlines, campaign start time, incumbent and shutdown guard remain
intact. The Pod launcher is updated to use the external runtime for later resumes.
The original launcher and handoff receipts are retained under `operator-controls/`.

The existing limits are **$40**, nine campaign hours, a two-hour final evaluation
reserve, sixteen LLM review calls, and the previously armed Pod shutdown deadlines.
No additional Pods or experiments are created just to apply the ceiling change.
Seed rotation and asynchronous scheduling are separate, unimplemented changes.

## How BO uses CPUs

The costly simulation evaluations run in separate processes. The current main
Pod allows sixteen simultaneous games. With four screening seeds per candidate,
the study proposes batches of four candidates. NumPy/BLAS thread counts are capped
at one per worker to avoid multiplying each simulation into another CPU pool.
The main Pod's actual CPU affinity contains 32 logical CPUs across 16 distinct
physical cores. Sixteen game slots are therefore a reasonable initial setting;
increasing to 32 requires benchmarking and does not double physical capacity.

The small Gaussian-process model chooses a diverse batch from a pool of proposed
settings; its fitting/acquisition step runs in the study coordinator. The current
major-boundary search allows 48 trials and scores a pool of up to 512 proposals at
each selection. Simulation evaluation is the primary parallel workload.

A local synthetic timing check (16 scalar dimensions, 48 completed observations,
512 proposals, four selections, single-threaded BLAS, ten timed calls after
warmup) took a median 0.0166 seconds and maximum 0.0279 seconds per suggestion
batch. This measures only GP scoring/selection on an already built proposal
pool; it excludes constructing and validating that pool, simulation evaluations,
and startup imports. It does not measure optimization quality or Pod hardware
performance. The receipt is `runs/runpod-launch-20260918/bo-suggestion-timing.json`.

This is batch parallelism on one Pod, not an asynchronous distributed BO service.
The three satellite Pods run independent focused studies and Python checks; their
evidence can inform later main-campaign reviews. They are not workers in the main
BO queue. A study waits for its entire current batch before proposing more
candidates, so stragglers leave slots idle. The measured impact and recommended
rolling-queue improvement are documented in
[the performance audit](research_performance_audit_20260919.md).
