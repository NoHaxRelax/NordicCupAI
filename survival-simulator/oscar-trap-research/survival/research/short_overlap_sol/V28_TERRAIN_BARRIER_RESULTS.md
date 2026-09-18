# V28 all-action terrain barrier

Frozen source: `policy_v28_terrain_barrier.py` (`edbe508a325fd953fc6178a3b3193d9e56f0c802e29f95566dd3f6b683473607`). It inherits the frozen integrated V27 controller and applies an 8-unit erosion check around every proposed endpoint when a fresh predator DTO is under 65 units. If an action would enter slower terrain, it substitutes the collision-clear same-or-faster endpoint with greatest observed predator separation.

The motivating V26 failure on map 10164 died at 5.8 seconds. Its endpoint-only terrain gate fired at 5.6, but the native endpoint was in river at 5.7; separation fell from 39.1 to 17.4. V28 avoids that transition and makes a causal handoff at 35.6 seconds.

All runs use native dynamics, every-frame streaming recordings, 300-second horizons, and causal scorer V4.

| Map / fixture | Set | Death | Entry / delivery | V4 result |
|---|---|---:|---:|---|
| 10164 / 20164 | fitted V26 failure | 35.5 | 35.6 / 35.6 | pass |
| 10139 / 20139 | fitted terrain regression | alive | 19.6 / 20.2 | pass |
| 10212 / 20212 | fresh | 71.9 | 159.7 / 159.7 | **fail: no unbroken guide-to-bait chain** |
| 10213 / 20213 | fresh | 34.3 | 34.5 / 34.5 | pass |
| 10214 / 20214 | fresh | 21.6 | 21.7 / 21.7 | pass |
| 10215 / 20215 | fresh | 36.9 | 37.0 / 37.0 | pass |

All legacy harness receipts say success and all captured predators remain held through 300 seconds. The causal result is 3/4 fresh: map 10212 is an autonomous capture 87.8 seconds after guide death and is retained as a failure. This sample cannot support a 95% claim.

Receipts, V4 score sidecars, and frame streams are in `results/short_overlap_sol/v28_terrain_barrier/`.
