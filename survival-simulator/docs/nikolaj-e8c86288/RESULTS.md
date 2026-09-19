# Nikolaj rank1: 1,000 fresh maps

Commit e8c8628862066d3e59226a3d35bd365b635be81d; recommended configuration models/best_policies/rank1_pod-03.json. Unchanged native engine and policy. Seeds 19001–20000, natural predators, policy seed 0, 3000-second horizon. All 1000 unique seeds verified.

Mean score: **1474.4**, 95% bootstrap CI **1449.3–1499.0** (10,000 resamples).
Mean survival: 1494.3 seconds.
Elapsed: 16.17 minutes using 12 workers on Ryzen 5 3600 (Zen2).

Compute, summed timing divided by summed ticks:
- ns_policy: 458.3 microseconds/tick
- ns_engine: 312.3 microseconds/tick
- ns_interface: 0.4 microseconds/tick
- ns_loop_cpu: 760.1 microseconds/tick
- ns_loop_wall: 771.0 microseconds/tick

Loop wall time includes CPU scheduling under concurrent load. Timing is not directly comparable to a different CPU.

The sharedfood20 Runpod panel is still running. Matching seed numbers do not guarantee matching trajectories across CPUs: upstream documents NumPy floating-point differences between CPU families. Policy, engine revision, Python/NumPy and CPU provenance differ. A cross-panel score difference must not be presented as a clean isolated policy effect. This is the checked-in recommended rank1, not an unavailable winner from untracked studies.
