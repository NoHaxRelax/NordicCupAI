# Escape from sightings before forecast traffic, 19 September 2026

The three sprint-available non-guide captures in the control replay were
avoidable. `bystander_avoidance.py` combined the guide's predicted future
predator positions with real sightings and ranked their maximum danger
together. Escaping a hypothetical future position could consequently steer
the agent straight into the predator it was currently observing.

The fix keeps those two sources separate. Immediate contact, hearing and
vision exposure from current observations take priority. Guide corridors and
the bait keep-out zone still steer agents outside immediate danger. Forecast
traffic alone no longer forces sprinting; a safe walking detour is eligible.
No engine state enters the policy, and the survival module is unchanged.

## Native counterfactual verification

`scripts/replay_escape_failures.py` replays every original action from the
same seed, verifies the living-agent IDs and positions, and changes only the
victim's fatal action. For agents 9 at 127.5 s, 92 at 444.4 s, and 109 at
577.1 s, the original action kills the agent and the corrected action leaves
it alive after that native tick. Maximum pre-change position error is zero.
This establishes an avoidable immediate failure, not subsequent survival.
Exact output: `escape-current-sightings-counterfactuals.json`.

## Full native game

Seed 1883894846, maximum 3000 seconds, default guide and survival settings.
The fix is `e6485d9`. Every frame is retained at
`logs/entrapment-iteration/escape-current-sightings-v1-20260919`;
replay: `http://localhost:9081/`. Summary and manifest are adjacent JSON files.

| Measurement | Control | Corrected escape |
| --- | ---: | ---: |
| Score | 784.88 | 1077.50 |
| Lifetime (seconds) | 750.4 | 1027.5 |
| Premature sprint-available captures, all roles | 3 | 0 |
| Delivery arrivals / assignments | 2 / 19 | 3 / 28 |
| Guide predator deaths | 9 | 14 |
| Other predator deaths | 11 | 13 |
| Estimated bait gaps after first arrival (seconds) | 81.7 | 223.4 |
| Maximum predators within 40 of bait for 30 seconds | 4 | 4 |
| Mean agent energy / maximum | 0.288 | 0.275 |
| Ripe fruit fraction (age at least 20 seconds) | 0.923 | 0.910 |

Runtime 414.95 seconds, 10,275 saved frames. The longer game still ended in
extinction. This is a promising single-seed result, not a reliability claim.
Zero sprint-available deaths also does not mean agents escaped after losing
sprint energy. The requested population-wide outcome remains unachieved.

At 720 seconds, only five agents remained: bait had energy 5 and age 82;
the richest gatherer had energy 214 and was roughly 828 estimated units away.
No viable replacement could reach the bait with the required reserve. The
trap went unbaited for much of the remaining game. This needs a better-fed
donor supply, not merely a last-second dispatch rule.

## Performance and further screens

`8189bb1` removes exactly duplicated observed wall segments from avoidance.
All 69 checked recorded sighting inputs returned identical actions; local
elapsed time for those calls fell from 0.144 to 0.065 seconds. This is a small
profile sample, not a whole-game speedup measurement.

Oscar's latest remote branch was checked at `4ee1aa0`. Best survival settings
are unchanged from `a62e04e`; new commits chiefly concern his separate native
guide experiments. Their scenario results are not our full-game results.

The prior free-PC screen using the regressed v3 guide was stopped after one
completed game and five 1800-second infrastructure timeouts. Six live jobs
and four queued jobs were cancelled; partial artifacts and the stop record
are saved locally. They cannot provide a paired comparison.

An initial replacement launch failed before any game because the engine
checkout's old `models` package shadowed the evaluated policy. `0f348ad`
fixes the import order and records the actually loaded module paths in each
manifest. A one-second remote smoke run verified both paths before retrying.

The corrected free-PC batch uses frozen source
`/home/lucas/entrapment-escape-0f348ad`, three workers, seeds 204871, 730951
and 605319, and current vs newer Oscar survival parameters (six full games).
It uses the default v2 guide, corrected escape, a 10800-second wall limit per
job, and `logs/paired3`. Native engine source is the unchanged verified copy
in `/home/lucas/colony-settings-20260919`. No paid pods were started.

Two additional local replays test existing options separately on the dev
seed: `--release-trap-food` and `--bait-reserve 90`. They are not yet adopted.

The release-food replay completed at **441.04 score / 418.1 seconds**, with
18.5 estimated seconds without bait and zero sprint-available captures.
It regressed markedly and remains off. The 90-second reservation replay also
completed: **790.58 score / 746.2 seconds**, with **7.8 estimated bait-gap
seconds**. Its shorter lifetime reduces the opportunity for later gaps, so
that gap total is not sufficient evidence of an improvement. It remains off.

## Observed-terrain bait ETA probe

The all-river estimate may reject a donor that can reach bait through faster
visited terrain. `--bait-terrain-estimate` tests a more local estimate using
`models/entrapment/bait_travel.py`: sample the planned route every eight units,
use the slowest observed biome within 12 units (including pose uncertainty),
and retain river speed on unknown stretches. Bait selection, revalidation and
fruit-detour checks share the same estimate. The normal overlap, 15-second
useful-lifetime requirement and six-energy travel reserve remain in place.

These are visited points, not biome polygons; interpolation can miss a border.
The estimate is not a safety proof, remains opt-in, and is recalculated while
travelling. A simple numerical check gave 0.8 seconds for 80 observed forest
units at walking speed 10 and 2.67 seconds for the same unknown route. The
full native replay completed in
`logs/entrapment-iteration/escape-bait-terrain-20260919`: **626.58 score /
601.3 seconds**, 86.6 estimated bait-gap seconds, and one premature guide
capture with sprint available. The victim had sprint speed 10.56 in swamp,
below predator sprint speed 15 even on equal terrain. This option regressed
on the dev seed and remains off. Summary and exact source manifest are saved.
