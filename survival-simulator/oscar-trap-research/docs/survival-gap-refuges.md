# Alternative trap: enter a passage the predator cannot fit through

17 September 2026. Local personal Nordic AI Cup research against unchanged upstream `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`.

**The strongest new alternative is a size-selective refuge between two obstacles.** Agents have radius 5, predators radius 10. A physical gap wider than 10 but narrower than 20 units can admit the agent while excluding the predator. The agent walks approximately 30 units into a long passage and stands still. The predator continues detecting it from the mouth, but collision prevents entry. This uses two ordinary obstacles; neither needs to be a thin wall, and no narrow river is required.

The current experiment establishes mapped-site acquisition and holding from arranged approaches. It does not implement discovery from ordinary startup observations, colony recruitment or indefinite renewal. It is a bait-maintained trap: withdrawing the bait allows the predator to leave.

## Evidence

| Test | Result |
|---|---|
| Constructed gaps 12/15/18, lengths 60/90, headings ±0.3 | 12/12 full 60-second holds, no captures |
| Negative control: gap 22, same lengths/headings | 4/4 captured at 0.5 seconds |
| Three predators at one constructed gap | 2/2 full 60-second holds |
| Native maps 1–20, one selected site per eligible map, headings ±0.3 | 18/18 full 60-second holds across 9 maps |
| Fresh native maps 21–40, unchanged selection/controller | 20/20 full 60-second holds across 10 maps |
| Three predators at native sites on seeds 1/7/8/9 | 4/4 full 60-second holds |

All successful runs maintained both proximity to the refuge mouth and detection of the bait for 100% of post-acquisition checks, including active-predator ticks. Merely surviving after disengagement does not count. The single bait used normal founder traits and 150 starting energy. Dry-land entry cost **2.25 movement energy**; terrain-dependent cases cost 2.81–4.5. It spent no further movement or turning energy while holding, although passive energy and later senescence still apply. At 60 seconds the bait retained approximately 85.5–87.75 energy. No tested bait needed food during this horizon.

The native tests retain the original obstacles, biomes, trees and food dynamics. They include slower terrain, including swamp and desert. Initial agent and predator positions are arranged, and additional predator spawning is disabled to isolate the mechanism. The three-predator tests explicitly arrange three valid, unblocked starting positions. No predator starts inside an obstacle.

## Map availability

A fixed oracle scan finds opposing obstacle faces with a physical gap between **11 and 19 units**, at least **55 units of overlapping passage length**, a clear straight agent route, and clear initial predator/agent placements. It considers both orientations and both entrances. The one-unit clearance margin excludes exact tangency. Boundary obstacles are included.

| Maps | Maps with a candidate | Valid entrance candidates | Maps with a full-speed entry route |
|---|---:|---:|---:|
| Seeds 1–20 | 9/20 | 15 | 4/20 |
| Seeds 21–40 | 10/20 | 21 | 5/20 |
| Total | **19/40** | 36 | 9/40 |

Some candidates are opposite entrances to the same passage. This is not 36 independent refuges. For simulation, the longest eligible passage on each map was selected before looking at its outcome. All 19 selected sites held in the two tested headings, including non-full-speed approaches. Map selection still reads true obstacle geometry; the two approach headings and aligned starts do not cover arbitrary encounters.

The nominal size difference alone is not enough: a passage must be long enough to put the bait beyond the 15-unit contact distance, and other nearby obstacles must leave the approach usable. Very narrow gaps demand accurate localization. The 11-unit lower screening bound leaves only 0.5 units of center clearance on each side for a radius-5 agent. Wider accepted gaps provide more tolerance.

## Why this differs from the water approach

Water control repeatedly moves two baits to alternate the nearest target. The passage refuge lets collision do the holding. A young stationary bait pays about 1 passive energy per second, versus roughly 5.8 total energy per second for the previously tested moving water pair. Those figures exclude scouts, nursery workers, replenishment and senescence, so they are service-cost comparisons rather than complete strategy budgets.

Several predators can hear the same protected bait and congregate outside the same entrance. Native experiments with three predators passed, but this does not show that all approach directions or additional spawn locations are safe. Predators arriving at the opposite entrance or workers inadvertently becoming nearer targets remain untested failure modes.

## Is it a permanent cage?

No permanent cage has been established. In two explicit withdrawal controls, the bait left at 10 seconds. The predator escaped the mouth region at 13.3–13.4 seconds; one later caught the fleeing bait. A holder must remain and eventually be fed or replaced. Legal gap handoffs and newborn preparation have not yet been tested in this namespace.

A separate exploratory idea was a one-way pocket: coax a predator in with its 15-unit chase move, then leave it unable to exit with its 11-unit untargeted walking move. A one-unit raster census of 20 native maps found enclosed predator-clear pockets, but no screened opening in that 11–15 range. The closest borderline candidate on seed 16 measured about 15.16 units after a local 0.025-unit refinement, still beyond a dry-land chase step. This is a finite geometric search, not a proof that every possible one-way trap is impossible. No usable walk-away prison was demonstrated.

Other ideas have clearer limits in the engine: agents and trees are non-solid, so they cannot form a body wall; predator exhaustion causes temporary rest rather than permanent incapacitation. Existing thin-wall pinning remains a separate promising mechanism, documented in `survival-wall-deployment.md`; its natural-food replacement results should not be credited to these new gap experiments.

## Controller and measurement boundaries

The gap controller receives the mapped target and initial own pose. It dead-reckons requested movement using public own biome and uses legal walking actions followed by zero movement/turn. It does not read the predator's hidden energy or resting state to choose actions. Pose error was zero in the retained native entry tests because the selected routes were clear. Actual geometry, predator coordinates/rest and native observations are used for diagnostic scoring.

One action is issued for the one living bait each tick. There are no paid births, mutations, energy injections or arranged mature fruits. The generated world remains native except for arranged creatures and disabled additional predator spawning. Containment is defined as all tested predators remaining within 75 units of the entrance, jointly with all of them detecting the bait. Attention is sampled after the bait action and before the native world tick, with containment measured afterward. The 60-second endpoint is a cutoff, not a demonstrated maximum duration.

## Files and reproduction

Implementation:

- `survival/research/water_deployment/alternative_traps.py`
- `survival/research/water_deployment/pocket_survey.py`

Results use `survival/results/water_deployment/alternative-gap-*.json` and `alternative-pocket-*.json`. The initial eight dry-site native runs are a subset of the 18 all-terrain development runs and must not be double-counted.

```sh
survival/.venv/bin/python survival/research/water_deployment/alternative_traps.py controlled
survival/.venv/bin/python survival/research/water_deployment/alternative_traps.py survey
survival/.venv/bin/python survival/research/water_deployment/alternative_traps.py native --include-slow
survival/.venv/bin/python survival/research/water_deployment/alternative_traps.py survey --start 21 --stop 41
survival/.venv/bin/python survival/research/water_deployment/alternative_traps.py native --start 21 --stop 41 --include-slow
survival/.venv/bin/python survival/research/water_deployment/pocket_survey.py
survival/.venv/bin/python survival/research/water_deployment/pocket_survey.py --refine
```

The native three-predator replay is `survival/results/water_deployment/alternative-gap-native-three-predators.replay.json.gz`. It uses the existing recorder and original `Environment.draw`. No new viewer or renderer was added.

All completed gap experiments now record automatically, including the too-wide-gap and bait-withdrawal failure controls. Unique state recordings go to `survival/results/water_deployment/replays/` and appear in the live Survival Lab catalog. An explicit `record` argument selects native rendering for demonstrations. The historical audit and per-row receipts are `survival/results/water_deployment/replay-audit.json` and `survival/results/water_deployment/reproductions/`. Backfilled footage is labeled **NEW reproduction**, with the saved configuration and original result-file hash; it is not recovered historical footage. See the water deployment report for audit and resume commands.

**Next practical step:** recognize a two-face passage from observed edges, test a safe approach with a real chase in progress, and supply a replacement route before committing an aging holder. The strongest current result is a much cheaper holding mechanism with more candidate maps in this local survey, not yet an autonomous full-game trap policy.
