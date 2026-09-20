# Runpod runtime sample: committed sharing optimizations

Source `f31f586`; 32 fresh seeds 30001–30032 per policy, ordinary full games with predators/energy, 3000-second horizon or extinction. One existing 32-vCPU Runpod CPU pod, 32 game workers, randomized job order. BO broker drained and resumed; BO game processes were not suspended.

| Policy | Mean seconds/game | Bootstrap 95% CI | Median | P90 | CPU µs/tick |
|---|---:|---|---:|---:|---:|
| shared_food_control | 25.68 | 23.04–28.51 | 24.66 | 34.97 | 1251.0 |
| congestion_pricing | 24.78 | 22.06–27.61 | 21.77 | 34.18 | 1284.8 |

Times cover the native game loop, excluding map initialization; profiling is enabled. Intervals describe map-sampling uncertainty on this host/load, not between-host or repeated-measurement uncertainty. A 32-map sample is not a new 1,000-map runtime average. Later jobs may experience lower concurrency as the sample drains.

Only committed behavior-preserving optimizations were used; no new tuning or reduced game horizon. Source build provenance, raw rows and BO-broker resumption record are stored here.
