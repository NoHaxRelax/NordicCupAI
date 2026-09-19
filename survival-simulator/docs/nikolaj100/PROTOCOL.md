# Frozen rank-1 benchmark

Policy: Nikolaj's `rank1_pod-03.json` from commit `6e49081`, unchanged.
Both engine and policy run in C++; normal energy, aging, food and natural predator spawning.
100 fresh world seeds: 5001–5100. Independent policy RNG seed: 0.
Horizon: 3000 simulated seconds or extinction. No competition service is contacted.
No tuning or candidate selection uses these maps. Report mean, median, spread,
bootstrap interval, survival, and all individual scores.

The runner follows the native campaign initialization (policy sees the initial
public agent states at time zero). Six Python configuration keys have no native
parser entry: five are unused, and `wait_tol` only increments a diagnostic counter.
They do not change decisions. The score result measures the current C++ port,
not a fresh reproduction of the historical Python training runs before fixes.

Runpod CPU pools rejected 32/16/8 CPU3c and 16 CPU5c requests, and an A4500
request was also unavailable. Dedicated pod zmml46gemukrm5 has 16 CPUs attached
to an L40S; GPU is unused. $1.09/hour, 10 GB disposable disk. Leave running after download, as explicitly requested by the user. Ongoing cost: $1.09/hour.
