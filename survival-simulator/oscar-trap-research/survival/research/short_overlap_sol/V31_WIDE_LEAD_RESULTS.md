# Wider chase-lead experiment

`policy_v31_wide_lead.py` inherits the wide terrain route and compact site fallback, raises predicted moving separation from 35 to 60, stationary separation to 65, the upper spacing bound to 75, the hold point from 75 to 95, and the slow-terrain guard from 65 to 95. It keeps continuous predator gaze.

The candidate is rejected. Causal scorer V4 reports 0/4:

| Map / fixture | Set | Result |
|---|---|---|
| 10212 / 20212 | fitted boundary failure | fail; death 64.5, no capture |
| 10224 / 20224 | fitted one-step barrier failure | fail; death 2.5, later autonomous legacy capture |
| 10280 / 20280 | fresh | fail; guide alive but no delivery through 300 |
| 10281 / 20281 | fresh | fail; death 1.0, later autonomous legacy capture |

The predeclared 10282 run was interrupted at 10 seconds after the candidate was conclusively rejected; it has no published receipt. Map 10283 was not started. Larger nominal lead can stall route progress and does not replace collision/terrain viability.

Receipts, causal sidecars, and every-frame streams for completed cases are under `results/short_overlap_sol/v31_wide_lead/`.
