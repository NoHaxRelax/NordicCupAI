# Interrupted C++ benchmark: 42 of 100 games completed

Stopped at the user's request on 2026-09-19. **This is not a completed 100-game
benchmark.** Slow runs are missing, so the completed subset is biased toward
faster games. Do not interpret its mean as the mean across all 100 maps.

| Measurement | Completed games |
|---|---:|
| Games | 42 / 100 requested |
| Mean score | 1157.46 |
| Median score | 1101.68 |
| Minimum–maximum score | 524.49–2491.77 |
| Mean simulated lifetime | 1154.52 seconds |
| Reached 3000 seconds | 0 / 42 |
| Guide handoff arrivals / assignments | 200 / 1143 (17.5%) |
| Games establishing bait | 41 / 42 |
| Established-bait games with a later occupancy gap | 41 / 41 |
| Premature predator deaths with sprint energy available | 1005 |

Handoff arrival means reaching the stationary delivery point while a predator
is observed following. **It does not prove capture or permanent retention.**
Twenty-six games had a predator continuously within 40 units of occupied bait
for 30 seconds; the largest such count was four. This is a proximity metric.
Sprint-available deaths cover all roles and exclude intentional stationary
delivery sacrifices. There were 163 such intentional sacrifices.

Bait gaps include previously occupied sites that were subsequently abandoned;
the evaluator does not distinguish these from an interrupted replacement at
the current primary site. The longest measured gap was 2210.4 seconds.

The controller does not meet the requested 95% reliability or continuous-bait
requirements. The completed orchard-only PC reference averaged 1563.78 over
100 games, but it is **not a matched comparison with these 42 cloud games**.
The cloud baseline never started. A separate 25-game PC entrapment partial is
archived nearby; it is not pooled into these figures.

## Code and conditions

Frozen controller/evaluator: `04173f8ee9d957ff947e6f7d5e73f76ef27750b0`, on
`survival-simulator/lucas-experimental`. Simulation and per-tick decisions run
in C++17. Games have predators, ordinary energy, aging and reproduction. Policy
inputs contain observed information; evaluator truth never feeds decisions.

The C++ controller is an adaptation of Oscar's native mapping/survival with
observed trap discovery, route-aware bait replacement, fruit detours, shared
avoidance and three-tick guide search. It is not a bit-for-bit port of the prior
Python controller/Nikolaj explorer. Corners and replacement scheduling for
secondary traps after group merges remain unsupported.

## Runtime and saved evidence

The 32-core Runpod batch ran from approximately 12:11 to 13:19 UTC. A separate
45-second local profile of seed2026091950 measured 77.37% of sampled time inside
A*; `clear_of` and `clear_segment` accounted for 49.62% and 22.94% self time.
Repeated collision work is the measured bottleneck in that sample. The
spatial-index candidate in `../../entrapment_iteration/unvalidated-astar-spatial.patch`
only compiled: it was **not behaviorally validated, timed, merged or benchmarked**.

All owned evaluation workers were stopped. Pod `gmmphzfgodxci2` was deleted at
13:21 UTC; a subsequent query returned404. Estimated final-pod cost is at most
about **$1.26**, including a disk margin; posted billing was still incomplete.
The conservative research total is **$6.51**, below the authorized $10.

- `games.jsonl`: exactly42 completed rows, sorted by seed.
- `shards/`: original interrupted manifests, raw rows and logs.
- `interrupted-status.json`: missing seeds, hashes and termination evidence.
- `interrupted-partial-summary.json`: measured summary.
- `score-histogram.svg`: distribution of the42 completed scores.
- `source-provenance.json`: all native inputs checked against frozen Git blobs.
- `runtime-profile.speedscope.json`: raw diagnostic samples, separate from scores.

The development replay remains at http://localhost:9084/. It uses the frozen
controller on development seed1883894846; it is not one of these benchmark games.
