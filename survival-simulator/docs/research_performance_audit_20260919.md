# Research performance audit — 19 September 2026

Measured at approximately 01:02–01:04 Europe/Paris (23:02–23:04 UTC on 18 September).
This was a read-only audit of development cases and live processes. No new
experiments were launched, and no holdout results were opened.

## Findings

The clearest throughput problem is the barrier between search batches.
`research_study.py` proposes four candidates at a time, each evaluated on four
seeds. `evaluate_batch()` fills 16 game slots but waits for every game in that
batch before the study proposes more candidates. Completed slots remain empty
while the slowest games run. Saving completed candidates early does not refill
these slots.

For the first 16 games on each satellite, occupancy was:

| Study | Batch elapsed | Occupied game slots, averaged over batch |
| --- | ---: | ---: |
| Guide delivery | 9.15 min | 56.8% |
| Bait continuity | 10.73 min | 48.1% |
| Capture allocation | 6.21 min | 59.7% |

Occupancy is summed game wall time divided by 16 times the batch elapsed time.
Batch starts were reconstructed from result timestamps minus game wall time;
these values are approximate. This measures game-slot occupancy, not physical
CPU utilization. Perfectly filling those slots at unchanged game speeds would
give a theoretical 1.7–2.1x throughput improvement for these batches. Adaptive
proposal dependencies, hardware contention and validation barriers mean that
this is an upper bound, not a measured speedup.

Python policy decisions dominate individual C++ games. The C++ engine accelerates
physics; the policy still runs in a separate Python process and exchanges JSON
observations/actions every 0.1 simulated seconds.

| Completed development cases | Count | Median wall time | Policy CPU / game wall time | Policy request wall time / game wall time |
| --- | ---: | ---: | ---: | ---: |
| Main, C++ | 46 | 4.67 min | 75.1% | 84.6% |
| Guide delivery, C++ | 26 | 4.10 min | 72.3% | 82.3% |
| Bait continuity, C++ | 18 | 5.11 min | 76.7% | 85.7% |
| Capture allocation, C++ | 32 | 4.15 min | 75.4% | 84.3% |
| Main, Python validation | 20 | 5.80 min | 59.7% | 67.9% |

Fractions are ratios of summed timings across completed cases. These are different
trajectories and workloads, so comparing rows does not measure engine or feature
speedup. The longest completed main C++ game took 11.17 minutes. All completed
games in this sample ended in extinction before the 3,000-second simulation limit;
longer survival can naturally take more wall time.

The policy request time includes policy computation, transport, validation and
debug export. The difference from policy CPU time is not a measurement of JSON
overhead alone. Decision CPU time was over 99.9% of decision wall time, and live
process samples showed little runqueue waiting. These measurements do not suggest
CPU scheduling contention as the main cause.

Diagnostics have a measurable cost. In the existing matched native seed-0 pilot,
the same trajectory took 162.80 seconds without recording and 200.46 seconds with
recording: approximately 23% extra runtime. This includes collection and other
instrumentation effects; screenshots alone cannot be blamed. Current collection
timers account for roughly 10–13% of native wall time, but exclude policy export,
engine hooks and some finalization work. Clip timing partly overlaps collection
and must not be added to it.

The 20-second live samples showed ample memory (about 3–4 GB used versus 64 GB on
the active Pods where cgroup measurements were available), no reported cgroup CPU
throttling, and low CPU/I/O pressure. The guide Pod did not expose the expected
cgroup files; its per-process measurements still showed little scheduling delay.
Processes that started or ended during the sample are excluded from CPU deltas.

The main campaign had just completed its Python promotion comparison and entered
the LLM review for subgeneration 1.2. Simulation inactivity during that review is
part of the sequential research workflow. The satellite searches continued.

## Recommended order of changes

1. Replace the search batch barrier with a bounded rolling queue in a new,
   versioned scheduler. Refill slots from fully specified immutable candidates;
   allow partial results to free capacity without treating a partially evaluated
   candidate as a winner. Persist proposal ordering, search RNG, completed parent
   evidence and pending trials so asynchronous completion does not lose provenance.
   Maintain the same worker, cost, deadline, cancellation and storage limits.
2. Profile the Python policy on representative development trajectories. Optimize
   repeated planning, geometry and serialization only after identifying functions
   responsible for the cost. Verify unchanged decisions on replay where an edit
   is intended to preserve policy behavior. Keep the observation-only process
   boundary intact.
3. Benchmark diagnostic tiers for future screening: retain energy/death metrics
   and a terminal ring buffer, while reserving richer traces and screenshots for
   finalists or failures. Measure costs before reducing diagnostic detail; do not
   discard the evidence needed to explain predator-capture failures.
4. Benchmark worker-count changes separately. Sixteen active games does not imply
   sixteen unused physical cores on a 32-vCPU allocation. More Pods increase
   independent throughput, not the speed of a single sequential policy call.

The current frozen campaign sources and protocols were left intact. A scheduler
change requires its own verified version and adoption at a controlled boundary.

## Evidence and rerunning the audit

`scripts/research_performance.py` is a standard-library, read-only Linux collector.
It checks case metadata against development seeds before opening case results.
It prints only selected numeric result fields and classified process statistics;
it does not print command lines, credentials or simulator world states.

The deployed helper is `/workspace/research-perf-audit.py`, outside the immutable
source snapshots. Example on a Pod:

```sh
/workspace/predator-search-venv/bin/python -B /workspace/research-perf-audit.py --sample-seconds 20 --aggregate
```

Local numeric receipts are in the Git-ignored
`runs/runpod-launch-20260918/performance-{main,guide,bait,allocation}.json` files.
The matched pilot is `pilot-native-report.json` in the same folder.
