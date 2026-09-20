# Full-game 200k burst comparison

Local simulation only, run on mypc Ubuntu WSL with the native nightsim C++ engine and native orchard/evasion policy. Python orchestrates the harvester and batch. No hosted validation.

Policy: `survival/nightsim/serve/pred_best.json`. This is the existing orchard/evasion baseline, not a verified copy of Lucas's requested roughly 1700-score strategy. That policy identification remains outstanding.

32 paired seeds 101–132; horizon 3000 simulated seconds or extinction. Harvester: budget 200000, contact_nearest, trigger 30, closing 6, max_harvests 100, default cooldown 50 seconds and min_free 6. Maximum harvest count was raised to avoid the earlier two-burst limit. Simulation and policy use C++; no HTTP serialization or API capacity is measured.

| Metric | Base | 200k bursts |
|---|---:|---:|
| Mean score | 1526.30 | 8072.68 |
| Mean survival seconds | 1515.33 | 1441.30 |
| Total bursts | 0 | 362 |
| Actual negative-energy predator meals | 0 | 210 |
| Mean cumulative predator-seconds resting below -1000 energy | 0 | 5534.63 |

Per-burst transfer rate: 210/362 = 58.0%. All 210 successful targets were eaten during the burst tick; all 152 failed targets died of starvation one tick (0.1 simulated seconds) later. Thus these failures did not come from failure of the list-mutation skip during the initial burst. They came from failing to get eaten before the next energy check. Individual geometry/target-switch/rest causes are not yet instrumented.

Observations are computed before predators move. Distances used by the next policy call can therefore be stale by one predator movement, and skipped agents can retain older observations. The predator chooses its nearest visible prey after all agent actions, with movement constrained by steering and obstacles. A distance-only trigger cannot guarantee contact. Predator identity and resting state are not directly exposed in the observation.

The earlier 17/32 statistic meant games with negative aggregate predator penalty, not successful bursts divided by attempted bursts. It should not be described as a per-burst success rate. Gaps in ascending agent IDs also do not establish a difference between sorted live IDs and list order.

The higher score does not establish longer survival: mean survival changed by -74.03 seconds on this batch. Predator-seconds sum over predators and can exceed game duration. Raw per-game data and per-burst death outcomes are in `results.jsonl`.
