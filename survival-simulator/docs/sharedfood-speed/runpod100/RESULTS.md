# Runpod runtime sample: committed sharing optimizations

Source `bc3069d`; 100 fresh seeds 30001–30100 per policy, ordinary full games with predators/energy, 3000-second horizon or extinction. One existing 32-vCPU Runpod CPU pod, 32 game workers, randomized job order. BO broker drained and resumed; BO game processes were not suspended.

| Policy | Mean seconds/game | Bootstrap 95% CI | Median | P90 | CPU µs/tick |
|---|---:|---|---:|---:|---:|
| shared_food_control | 17.57 | 16.37–18.81 | 17.11 | 25.85 | 990.5 |
| congestion_pricing | 17.56 | 16.44–18.71 | 16.74 | 25.37 | 1003.5 |

Times cover the native game loop, excluding map initialization; profiling is enabled. Intervals describe map-sampling uncertainty on this host/load, not between-host or repeated-measurement uncertainty. This sample is not a new 1,000-map runtime average. Later jobs may experience lower concurrency as the sample drains.

Only committed behavior-preserving optimizations were used; no new tuning or reduced game horizon. Source build provenance, raw rows and BO-broker resumption record are stored here.

The under-20-second mean target is met on this sample for both policies; both map-bootstrap upper confidence limits are below 20 seconds. This does not guarantee every game or every CPU host finishes in under 20 seconds (P90 is about 25–26 seconds). The laptop action checks cover six full games/109,229 ticks, and all 64 games overlapping the prior Runpod sample match score/lifetime/tick count exactly.
