# Bait nursery prototype

Optional `--nursery-size 2`, default off pending paired evaluation. Two young
gatherers settle near recently observed trees on the rear side of the trap,
120–300 units from bait and within 220 units of the rear entrance. The chosen
food area needs a clear agent-width route to that entrance. If no suitable
observed trees exist, the nursery does not activate. No food is artificially
spawned and no evaluator information enters the policy.

Nursery members gather Orchard-assigned ripe fruit within 160 units of their
camp, use wall pathfinding, avoid predators and are exempt from guide/bait
recruitment while assigned. At age 55 they return to the ordinary role pool.
When bait is running low and no replacement is travelling, a nearby parent
with at least max(300, 70% capacity) energy can reproduce. Birth requests are
spaced by 15 seconds and limited when two children younger than 15 seconds
already occupy the food area. Children join the normal population and bait
selection; none waits at the trap entrance. This is a prototype, not verified
continuous bait production. It does not yet preferentially allocate all local
food to nursery parents or explicitly track which parent produced each donor.

Local every-frame pilot: `logs/entrapment-iteration/nursery-pilot-20260919`,
viewer port 9075. Green rings identify nursery gatherers.

Completed pilot: score 877.88, extinction 858.9 seconds, estimated bait gaps
274.6 seconds. Same-seed post-anchor baseline scored 872.19 at 821.3 seconds.
There is no demonstrated improvement from this single game. One nursery birth
occurred at 111.2 seconds (parent 21, newborn 27, confirmed by next-frame normal
agent IDs). Child 27 was never dispatched as bait and died at age 122.8 as a
gatherer. Thus this version produces population near the trap but does not yet
ensure those offspring become bait; explicit offspring reservation is needed
in the next version. The remote paired batch uses this immutable first version.

Free remote CPU experiment:

- `ssh pc` reports 12 CPUs. Six workers run twelve full games: baseline and
  nursery on seeds 204871, 917263, 605319, 148027, 730951, 392681.
- Isolated remote source: `/home/lucas/entrapment-nursery-20260919`.
- Interpreter: `/home/lucas/entrapment-research-20260918/.venv/bin/python`.
- Results: `logs/nursery-paired6/results.json` plus per-game summaries/manifests.
- Native extension uses the same pinned C++ source and Python 3.13.14 ABI.
  The existing portable compiled module imports and runs; the host lacks a C++
  compiler. No server or unrelated process was stopped.
- `scripts/compare_bait_nursery.py` runs isolated subprocesses, bounds each job
  to 1800 wall seconds, and reports infrastructure errors separately.
- Remote games use `--summary-only`; only the local pilot records every frame.

Runpod MCP and CPU catalog were verified. No new pod was rented: the free PC
is the first paired screen. The existing ledger estimates $4.88 spent of $10,
leaving approximately $5.12 for broader validation if justified. Other users'
pods are not part of this work and were left untouched.

## Offspring reservation prototype (v2)

Explicitly tracks newborns through normal parent observations, keeps them
gathering locally, protects them from guide recruitment, and prioritizes
viable nursery children for bait. The pilot scored 791.73, ended at 762.9
seconds, and accumulated 190.9 estimated bait-gap seconds. No nursery births
occurred: the stricter birth rules do not yet provide a reliable donor supply.
This is an unsuccessful experimental iteration, still disabled by default.
The first-version paired PC batch is unchanged.

## Scheduled donor prototype (v3)

Keeps one observed nursery child available while an earlier replacement
travels. Parents may reproduce below 75 seconds of current bait lifetime,
with at least max(250, 60% capacity) energy. Normal predator avoidance vetoes
reproduction during escape; a trapped predator behind a wall no longer blocks
all births merely through proximity. Requests are recorded only after the
final action/energy checks. Tree memory lasts 60 seconds and children remain
reserved if the temporary food center disappears. Children leave the reserve
at age 45; all route, energy, direct-entry and overlap rules still apply.

The first local v3 pilot has observed child 61 becoming bait at 197.4 seconds
(83.9 energy, age 27.5). Child 27 did not make the journey. A completed pilot
and paired maps are needed before drawing a performance conclusion.

A temporary 16-vCPU Runpod CPU3 pod, n0d6kz5mjuzd6a, was created at
08:00:41 UTC for the paired comparison. Compute price is $0.48/hour plus
disk; the previous pod's actual billing was $4.8917 including disk. Budget
remains $10 total. The free PC continues the immutable v1 batch.

Completed v3 development-seed pilot: score **1095.08**, extinction at **1056.9
seconds**, versus 872.19 / 821.3 for nursery disabled. Two observed nursery
children, one successful bait arrival. Eighteen total bait arrivals and sixteen
overlaps, but **284.3 seconds of estimated gaps**, almost all one terminal
coverage collapse. Mean energy 28.4%, 90.8% ripe fruit; fifteen non-guide
predator victims, including two replacement baits (native counts in summary). This
single-map increase is not evidence of general reliability. Every-frame replay
is `logs/entrapment-iteration/nursery-v3-pilot-20260919`, port **9076**.

Runpod v3 paired comparison uses 12 workers and these 12 seeds:
204871, 917263, 605319, 148027, 730951, 392681, 584107, 829631, 176453,
963217, 418759, 256903. Each seed runs nursery sizes 0 and 2 for the full
3000-second horizon or natural extinction, with normal energy and predators.
Source bundle frozen at 36d731e; Python 3.13.14, NumPy 2.5.3, Pydantic 2.13.5,
SciPy 1.18.1, Shapely 2.1.2 and pygame 2.6.1. The portable pinned native
extension imports and steps successfully on the CPU pod.
