# V26 integrated terrain-gate results

Frozen policy: `policy_v26_integrated.py` (`72174cbd18059bd7d21deefe392c400ead3eaf97234c0a81ff0f3067563aa424`).

This candidate combines the replaceable/short-overlap site selection and final-lane behavior, the release mixin, and the V25 terrain-transition gate. It uses only observations plus the static map. All runs used native predator dynamics, infinite guide energy, 0.1-second ticks, 300-second horizons, 320-pixel native frames, and streaming recorder v2 with every frame retained.

| Map / fixture | Set | Physical entry | Delivery | Guide death | Result |
|---|---:|---:|---:|---:|---|
| 10139 / 20139 | fitted terrain failure | 22.8 | 23.2 | alive at 300 | pass |
| 10040 / 9640 | fitted emergency failure | 17.5 | 17.7 | alive at 300 | pass |
| 10048 / 9748 | fitted emergency failure | 42.6 | 42.6 | 42.4 | causal handoff pass |
| 10180 / 20180 | fresh | 24.3 | 24.7 | 24.2 | causal handoff pass |
| 10181 / 20181 | fresh | 23.1 | 23.9 | 221.8 | pass; death long after capture |
| 10182 / 20182 | fresh | 44.4 | 44.5 | 242.6 | pass; death long after capture |
| 10183 / 20183 | fresh | 2.0 | 2.0 | 1.9 | causal handoff pass |

All seven retained the predator through 300 seconds with zero physical or target losses after delivery. The frozen fresh sample is 4/4, but four cases are far too few for a 95% reliability claim. The three fitted cases establish regression repair only.

The fresh direct-death cases are not delayed autonomous captures: entry follows guide death by 0.1 second on maps 10180 and 10183. Map 10048 similarly switches to bait at guide death and physically enters 0.2 seconds later.

Important integration limitation: V26 copied the replaceable-site controller and collapses the staged site fallback into one `min_gap=10.1, min_overlap=10.3` call after the default selector fails. A later integrated policy should bind root V24's exact staged selector (default, then narrow overlap >=20, then short overlap only if needed). V26 remains frozen for reproducibility.

Receipts and frame streams are under `results/short_overlap_sol/v26_integrated/`.

Subsequent fitted probe 10164 / 20164 failed: guide death 5.8 seconds, no delivery through 300. This disproves general terrain-transition repair by V26 despite its 4/4 fresh pilot. See `V28_TERRAIN_BARRIER_RESULTS.md` for the boundary-eroded repair and its own fresh failure.
