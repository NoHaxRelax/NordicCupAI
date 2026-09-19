# Continuous four-Pod research — 2026-09-19

**STOPPED by user request on 2026-09-19:** the fleet had expanded to eight Pods;
all eight are now verified `EXITED`. Optimization is halted and the queue has a
`STOP` marker. Completed data remains on the shared volume. Do not restart
without new user authorization. The following deployment details are historical.

The user authorized additional Runpod spending and requested that CPU workers
keep doing useful work during LLM reviews and while slower games finish. The
old $40 monetary cutoff and auxiliary 2.7-hour Pod shutdown are retired.

## Running deployment

- Campaign: `/workspace/research-resume-20260919`, target major generation 8.
- Main Pod: `a0aruuldun08t9`; launcher 10508, coordinator 10509.
- Worker Pods: `nc6e1yh96rqylc`, `3ziz66qp102nyh`, `jluij0qgnz2tys`.
- Queue: `/workspace/research-dispatch-20260919`.
- Background studies: `/workspace/research-capture-fleet-throughput-20260919`.
- Operational code: `/workspace/recovery-ops`, outside every frozen snapshot.
- Shared volume: `u9kre0adae`, expanded to 500 GB for cumulative evidence.
- Dashboard: http://127.0.0.1:8765, including measured container CPU usage.

Each Pod admits up to 30 shared/earlier games, leaving some CPU headroom. The
worker counts games already running outside the queue before admitting more.
Earlier study runners may briefly start another local batch, exceeding that
target during the transition; background admission pauses until those games drain. The
priority monitor gives background job processes and their descendants nice 8,
so normal-priority main evaluations retain CPU precedence.

The versioned `research_dispatch_v2.py` workers can admit up to eight urgent jobs
above their normal slot target, preventing a full background pool from delaying
a new main evaluation. New background executors set nice 8 at startup. Four
`research_worker_service.py` processes restart failed queue daemons while already
running case interpreters keep their locks and continue. Five daemon failures in
ten minutes request fleet shutdown instead of silently paying for idle workers.
Per-Pod `worker-*.v2-handoff.json` receipts record the executable hash and preserve
the earlier queue client/runner hashes used by the coordinator.

All four Pods share the queue. Final assessment has priority 0, main paired
comparisons 5, main screening 10, background Python confirmation 20, and
background screening 40. Running games are not killed because they survive
longer. Priority changes admission order and CPU scheduling, not scores.

The current pre-deployment BO wrapper is allowed to finish unchanged. Subsequent
main studies use the asynchronous driver: every completed game frees space for
more work without a whole-batch barrier. The GP trains only on fully completed
candidate panels; pending trials are not scored as losses. Candidate configs,
parents, case IDs, completed results, and the scheduler hash are recorded.

Six supplementary capture studies can run concurrently and replenish whenever
one finishes while development remains active. They test guide delivery, bait continuity
and capture/feeding allocation. Each immutable study plan pins the source and
incumbent and allocates fresh, disjoint screening and Python confirmation seeds.
It runs up to 24 focused candidates, followed by an eight-map paired Python
comparison when a supported finalist exists. Completed reports are advisory;
only the main supervisor can promote a candidate using its normal comparison.
Scoped Codex guidance points future reviews to these results and requires
careful interpretation across different source snapshots and seed panels.

## Integrity and recovery

The queue stores immutable job requests. Cross-Pod `flock` ownership prevents
duplicate execution; the job interpreter inherits the lock so a worker-daemon
restart cannot release a still-running case. A second cache-case lock prevents
two different comparisons from writing the same cached case concurrently.
Completed job requests move to an archive, keeping pending-queue scans bounded.
Each job uses a fresh interpreter for its candidate's frozen source, verifies
source hashes and environment/case identity, and runs the existing evaluator.

The new coordinator observes existing bounded step wrappers rather than
duplicating or killing their games. `operator-controls/throughput-handoff.json`
records the unchanged original config, protocol and accepted-policy hashes.
`runtime-adapter-throughput.json` records the adapter, queue and async-driver
hashes. `operator-controls/throughput.json` records authorization and deployment
parameters. The cumulative main journal includes the scheduling handoff.

Development jobs cannot use retired holdouts. Current holdout jobs require an
already frozen final-selection file, its exact original/selected source and
config, Python execution, and the final-evaluation control directory. The
dashboard exports worker counts and CPU usage without exposing held-out outcomes.

## Spending and shutdown

The four existing Pods cost $3.84/hour for compute, plus storage. No new Pods or
LLM API credentials were added. Astra xhigh still uses the verified ChatGPT login.
The runtime uses $3.95/hour for conservative whole-fleet accounting and retains
$10 for earlier spending. This charges the full fleet from main-Pod creation,
so it intentionally overestimates the period before auxiliary Pods existed.

Normal completion is generation 8 followed by the final assessment. The old
financial cutoff no longer stops development. A 72-hour infrastructure failsafe
from main-Pod creation ends at **2026-09-22 06:08:06 UTC**. The internal $300
ceiling covers that entire failsafe window, including the $5 reserve; it is not
an additional amount being spent or prepaid. The main's 4-hour final reserve
remains. Campaign storage has a 180 GiB allowance with 30 GiB reserved for final
assessment; new background studies pause when their root reaches 180 GiB.

The new controller guard is PID 9112, using `guard.json` and `guard.jsonl` in the
queue directory. It verifies the four Pod identities, stops workers before the
controller on completion, and stops the fleet after a sustained coordinator
failure or thirty minutes with no simulation work. The local deadline-only
backup uses `throughput-guard.json` and `throughput-backup-guard.jsonl` in the
Git-ignored launch folder. Controller guard 7327 and both old local guard
processes 21720/4688 were verified and retired after the replacements armed.
The original auxiliary launcher markers no longer stop their Pods; their shared
workers continue taking useful jobs after those earlier studies end.

## Verification

Twenty-six local tests passed for dispatch safety, asynchronous refill, complete
panels, runtime budgets, guards, and dashboard holdout exclusion. A live cross-Pod
locking test passed while the controller held a verified lock. The shared queue
completed 34 real games without executor errors during rollout, with container
CPU samples around 90–100% across all four Pods. Later verification found 129
completed queue jobs without executor errors, and all 24 main Python comparison
games started (five already complete, nineteen running, none waiting). Existing results and accepted
policy were preserved; the final holdout was still unopened.
