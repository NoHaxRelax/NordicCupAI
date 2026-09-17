# Water deployment: native sites, acquisition and replacement

17 September 2026. Personal Nordic AI Cup research, local execution only.

**Water control now works at selected native river sites, including a land approach and legal reserve handoffs. It is an opportunistic mapped-site tactic, not a finished startup policy.** On seed 37, both baits started on the same dry bank, acquired the predator in 0.9 seconds, and sustained continuous water containment plus bait attention until 87.9 seconds with three replacements. Native terrain, obstacles, trees and fruit were retained. A second site reached 80.7 seconds with two replacements. Neither run ended in capture; a serving bait depleted.

The broad limitations remain material: candidate sites are uncommon under the current search criteria, short approach gaps fail, food is uneven between banks, and automatic new generations are not reliably better than using existing reserves. No full-game win or competition score is claimed.

## What is implemented

Code is isolated in `survival/research/water_deployment/`; evidence is in `survival/results/water_deployment/`. No vendor, debugger, frozen baseline or other strategy files were changed.

- `sites.py`: scans arbitrary river angles, checks a common water interior and full-speed retreat lanes, measures obstacle coverage and native food near both shores.
- `probe.py`: a scout physically crosses a nominated site three times and reconstructs bank bounds from its own public biome readings.
- `controller.py`: assigns active baits independently of agent IDs, approaches behind a retiring bait, switches the assignment before depletion, recovers at native shore food, and obtains newborn pose and heading through public Agent observations.
- `base_snapshot.py`: a local fork of the earlier containment/acquisition controller. Additions include a local predator association filter and optional experimental corridor/entry changes. Historical controller snapshots are retained.
- `run.py`: exact one-action-per-living-agent execution, native births, energy accounting, joint containment/attention measurement, worker outcomes and native replay capture.
- `association.py`: two independent water stations, each following its own cached predator track.
- `comparison.py`: a fixed-horizon local comparison with the frozen observation-only nursery baseline and a nursery fallback after water control loses a bait.

All 18 vendored upstream source/document files still match their Git blob hashes in `survival/source-tree.json`, commit `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`. See `source-verification.json`. Results carry engine and experiment hashes. The engine is not patched. Controlled fixtures explicitly replace the map/entities and disable random new trees/predators; native fixtures retain map, obstacles and food dynamics but arrange initial creatures and, unless stated otherwise, suppress additional predator spawns.

## Information boundary and legality

**Privileged preparation:** the availability scan reads the biome map and obstacles. Tests supply a shared initial coordinate frame, site orientation and initial agent poses. Collision-aware dead reckoning uses the surveyed static obstacle rectangles. Starts, predator count and initial predator energy are arranged. Thus these are not observation-only full games.

**Controller decisions:** the active pair uses cached public predator `distance`, `angle` and `rel_dir`, public own traits/biome/energy/age, and tracked agent poses. It does not receive live true predator coordinates, energy, resting state, or future fruit. Foraging uses observed fruit, and replacement eligibility uses public agent state. Newborn localization uses another agent's observation ID, distance, angle and relative heading, rather than reading the child's true pose.

The loop submits exactly one finite action for each living agent present at the start of a tick. A newborn receives its first action on the next tick. Births are requested through native `spawn_agent=True`: the parent must still have more than 100 energy after movement, pays 100, and the child starts at 75 with native random traits and maximum age. Waiting, travel, turning, passive drain and senescence all cost real energy. No energy or fruit is replenished by the controller.

True coordinates, rest state and energy changes are used only for evaluation. A hold requires **both actual river occupancy and selection of a designated bait by the native nearest-detectable-agent rule**. Selection is measured after agent actions and before the native non-agent pass; water occupancy is measured after that pass. Death during the pass can change selection at the final 0.1-second boundary, so terminal depletion ticks should not be interpreted as additional usable service. Keeping agents alive after the predator disengages is not credited as containment. Metrics also include the active-predator subset; rest ticks cannot hide active failures.

## Native site availability and geometry

The new scan samples river ridge points and cross-sections every 10 degrees, rather than considering only horizontal/vertical rivers. Its screening footprint has an 80-unit along-river span, at least 32 units of common water, at most 62 units of bank envelope, full-speed banks, and retreat lanes extending about 65 units beyond each bank. Obstacle checks sample this whole footprint with a 10-unit buffer. This is a finite search and a sampled clearance check, not an exhaustive geometric certificate.

| Maps | Maps with candidates | Candidate sites | Completely clear sampled corridors |
|---|---:|---:|---:|
| Development seeds 1–40 | 5/40 | 26 | 2 |
| Held-out seeds 41–80 | 4/40 | 12 | 0 |
| Total | 9/80 | 38 | 2 |

Both clear corridors are on seed 37. Most sites are obstructed somewhere in the working area. A clear bait corridor does not guarantee a clear nursery or approach: a seed-37 site that works for two baits is rejected when a four-agent nursery start intersects an obstacle.

The native geometry pilot identified a concrete motion error. Uncorrected dead reckoning differed from true pose by tens of units after the engine deflected a move around an obstacle. On seed 23, site 3, collision-aware tracking changed a 27.7-second capture into a 51.9-second full-water, full-attention hold ending at depletion. The engine uses expanded **rectangles**, including square corners, rather than circle-to-rectangle distance. Matching that rule reduced the measured pose error to numerical precision in mapped trials. The first approximate collision model and its results remain in `native-pilot.json`.

In the corrected development batch, six of ten tested sites held with 100% joint containment until depletion, around 52–68 seconds. Others still failed despite accurate pose tracking. The held-out batch selected the least obstructed candidate on each unseen candidate-bearing map and tested two headings: four of eight starts were blocked, and only one of the four executable cases maintained a full-water hold to depletion (seed 70, 51.9 seconds). A 72.3-second survivor with only 53.8% joint containment is a failure, not a success.

### Paid local surveying

At three nominated seed-37 sites, a single scout completed three crossings in 8.1–8.4 seconds, spending approximately 48–50 energy. It measured 41–44 units of common water and 61–64 units between conservative dry-bank samples. Diagnostic position error was negligible. At the seed-23 test, an obstacle caused about 7 units of unobserved drift; that survey is not accepted.

Candidate nomination and initial approach remain privileged. The scout's measurements use only its own biome, but the experiment's acceptance check includes diagnostic truth. This does not yet solve searching the whole map, recognizing odometry failure online, or sharing a coordinate frame from arbitrary startup positions. Surveying also consumes a substantial fraction of a 150-energy agent's budget and is not free preparation.

## Entry from land

The two-bait dry-shore launch was tested at five native sites with both baits initially on the predator's shore. With a 45-unit predator offset from the shore and initial heading −0.25 radians relative to the cross-section, the mapped controller acquired the site. The corresponding 30-unit offsets failed at 0.3 seconds: after paying for the approach to the shoreline, there was too little clearance while the predator was still taking full-speed land steps.

A two-step lateral dogleg was tested as a concrete repair. Both directions failed, in 0.1–1.0 seconds. Those outcomes are retained in `dogleg-negative-pilot.json` and `dogleg.json`. The dogleg is experimental and is not recommended. Close-gap entries need a longer land approach or another bait to establish separation before crossing.

Complete native approach plus reserve trials, with four initial 150-energy agents and cached-observation association enabled:

| Native site | Acquire | Duration | Joint containment after acquisition | Handoffs | Outcome |
|---|---:|---:|---:|---:|---|
| Seed 37, site 8 | 0.9 s | 87.9 s | 100% | 3 | Bait depletion |
| Seed 37, site 10 | 0.9 s | 80.7 s | 100% | 2 | Bait depletion |
| Seed 23, site 2 | 1.0 s | 27.0 s | 96.5% | 0 | Capture |
| Seed 70, site 5 | – | – | – | – | Nursery start blocked |

These are ordinary short approaches within an already mapped site, not arbitrary-distance travel from game startup. Evidence: `approach-relay.json`.

## Food, replacement and senescence

A replacement approaches on the retiring bait's own bank, about 12 units behind it. The assignment changes only when the two are within 17 units and the incumbent has less than 50 energy, or is older than 52 seconds. The old bait then retreats to forage. Readiness is requested earlier, below 60 energy. No clone or teleport occurs during an ordinary reserve handoff.

In the no-food, width-50 fixture, four starting agents provide the same initial 600-energy budget in each comparison:

| Policy | Hold duration over three seeds | Replacement result |
|---|---:|---|
| Two active, two waiting | 51.7 s | No handoff |
| Existing reserves | 65.0–69.3 s | Two successful handoffs |
| Legally born successors | 54.2–54.4 s | Two births and two successful handoffs |

The newborn experiment spent 200 parent energy to obtain two 75-energy children. Delaying births until the serving bait fell below 70 energy improved the earlier 51.6–51.7-second result, but still underperformed direct reserve use. This is a resource finding, not a failure of physical handoff. Newborn pose reconstruction was accurate without hidden coordinates. Native mutations were retained, and agents too slow for the 9-unit ordinary movement request were not selected.

With untouched native food and four arranged starting agents, the reserve policy reached 88.2 seconds at seed-37 site 8 and 90.6 seconds at site 10, both with 100% joint containment. Total absorbed fruit energy across the four agents was approximately 1,384 and 1,153 respectively. These numbers include nursery workers and retired baits; they are not free energy delivered to the active pair. A resource-poor or inaccessible shore can still starve while the other side accumulates food.

There is also a demonstrated native newborn handoff: at site 8, a shore parent legally produced a child at 24.9 seconds; the child took over at 51.2 seconds with 106.5 energy after native foraging. That run reached 89.2 seconds with three handoffs and one birth, at an absorbed food cost of 823.8 plus the 100-energy birth. It does not establish indefinite generation turnover.

Senescence remains a limiting factor. The public DTO does not expose maximum age; the policy begins seeking a younger replacement at age 52 to leave margin before the native minimum maximum-age value of 60. The more aggressive `renewal-urgent` experiment lowered the parent reserve to 112 energy and permitted an extra child. It worsened two native cases into captures around 38 seconds and is not promoted. `native-renewal-v3.json` preserves these failures; its historical case field says `renewal`, referring to the then-current urgent revision. The reproduction script now names it explicitly `renewal-urgent`.

The planted-tree `orchard`/`renewal` suites are mechanism fixtures with six placed native trees and no placed mature fruit. Their ordinary growth, fruit production and death remain active, but these are not evidence of natural tree density. They do not replace the native-food results above.

## More than one predator

Two nearby predators at one station defeated the ordinary pair in about 1–2 seconds. A single pair must not be presented as handling multiple predators.

Two independent pairs at separate stations held two predators for about 52 seconds in all six controlled width/heading cases, using four 150-energy baits. Initially, the same arrangement failed on the native map in two seconds. The cause was **observation association**: when a bait could not see its own predator, it sometimes saw the second station's predator. Averaging the two estimates moved the supposed local predator between stations.

The corrected estimator reconstructs each observation in the shared local frame and associates it with the previous local track, using a movement-based distance bound. It uses public observations and history, not predator IDs or true positions. With this filter, the two native-station cases lasted 52.1 and 52.4 seconds, ending in depletion rather than capture. Joint containment of **both** predators was 100% and 99.81%; the latter had one out-of-water tick and is not an absolutely continuous hold.

Allocation to those two stations is arranged. Automatically detecting and acquiring a newly spawned predator, moving it to a spare station, and financing four replacement chains remain unimplemented. Tightening the lateral anchor to ±15 units was separately tested and did not consistently repair failures; `corridor.json` retains that ablation. The association fix, rather than anchor clamping, explains the successful two-station change.

## Workers and score

Two matched 150-second native assays use the same arranged four-agent starts and native food. The comparison is the frozen observation-only nursery policy. The water variant uses the reserve controller, then falls back to that same nursery policy when an active bait is lost. Random additional predator spawns are disabled in both. This is a local protection assay, not full-game validation.

| Site | Nursery score | Water + fallback score | Total captures, nursery / water |
|---|---:|---:|---:|
| Seed 37, site 8 | 143.65 | 150.55 | 8 / 7 |
| Seed 37, site 10 | 151.45 | 153.96 | 2 / 1 |

At site 8, the two initial shore workers lasted only 5.5 and 12.4 seconds under nursery control; with water assignment they lasted 89.1 and 94.4 seconds. At site 10, their lifetimes increased from 23.4/95.8 to 90.6/137.8 seconds. Those workers eventually served as baits, so this is explicitly not a claim that they were kept permanently out of danger. Population and birth counts differ between policies. The sample is too small and too selected to infer an overall score advantage.

Water fallback began at 88.2 and 90.6 seconds. Subsequent survival and score are included in the 150-second assay but **not** counted as continued successful water control. The controller's native senescence and resource failures remain visible.

## Replays and reproduction

Every completed water, gap and transect run now records automatically, including sweeps and failures. The default is a lightweight state replay with one-second sampled frames plus exact critical-event frames; the recorder is called after every step. Passing the existing `record` argument or `--record` requests native-rendered demonstration footage. Both modes use unique policy/version/seed filenames under `survival/results/water_deployment/replays/`, preserve prior recordings, and return the replay path in the result. The live Survival Lab catalog discovers these files automatically.

The historical replay audit is `survival/results/water_deployment/replay-audit.json`. It distinguishes original recordings, new reproductions, missing files and cases rejected before simulation. Per-row reproduction receipts under `survival/results/water_deployment/reproductions/` retain the historical source hash, controller limitations, new metrics and metric differences. Archived controller/base files are used when their hashes match the historical metadata. Otherwise, the replay explicitly discloses a current-controller reproduction. The old metric files remain unchanged; missing historical footage has not been recovered.

To inspect or resume the bounded historical backfill:

```sh
survival/.venv/bin/python survival/research/water_deployment/backfill_replays.py
survival/.venv/bin/python survival/research/water_deployment/backfill_replays.py --run --workers 3
```

Existing debugger-compatible replays use the original `Environment.draw`; no new UI or renderer was added:

- `survival/results/water_deployment/native-hold.replay.json`: 67.9-second native single-predator hold.
- `survival/results/water_deployment/approach-relay-native.replay.json`: native same-bank approach and reserve handoffs.
- `survival/results/water_deployment/two-stations-native.replay.json`: corrected association with two native stations.

The saved native images `native-hold-40s.png`, `approach-relay-native-40s.png` and `two-stations-native-40s.png` were visually inspected. The recorded approach replay reproduced 87.9 seconds with three handoffs; the two-station replay reproduced 52.1 seconds with 100% joint containment. Replay recording is read-only and does not consume engine RNG.

Run from the repository root:

```sh
survival/.venv/bin/python survival/research/water_deployment/sites.py --start 1 --stop 41
survival/.venv/bin/python survival/research/water_deployment/sites.py --start 41 --stop 81
survival/.venv/bin/python survival/research/water_deployment/run.py native
survival/.venv/bin/python survival/research/water_deployment/run.py relay
survival/.venv/bin/python survival/research/water_deployment/advance.py heldout
survival/.venv/bin/python survival/research/water_deployment/probe.py
survival/.venv/bin/python survival/research/water_deployment/approach.py
survival/.venv/bin/python survival/research/water_deployment/association.py
survival/.venv/bin/python survival/research/water_deployment/comparison.py
survival/.venv/bin/python survival/research/water_deployment/approach.py --record
survival/.venv/bin/python survival/research/water_deployment/summarize.py
```

Seeds, cases and trajectories are saved, and action legality is checked on every tick. Native object sets can affect ordering in tied interactions, so seeds are not a guarantee of bit-identical long food histories across Python processes. Preserve the recorded trace when comparing changes. Early pilot files also predate the corrected food accounting for the engine's skipped-agent pass on list removal; promoted reruns use observed age advancement to avoid inventing passive-cost recovery as food.

## Next iteration

Use a **single station with mapped collision tracking, the association filter, a sufficiently separated dry-shore entry, and existing shore reserves** as the current demonstration configuration. Keep the failed dogleg, urgent birth and anchor-clamping options experimental.

The next useful change is a shore-specific nursery planner: locate reachable fruit and trees, estimate each reserve's paid travel time and remaining service time, delay births until a food-backed slot is available, and prepare the next generation before the serving bait reaches senescence. That should be paired with online landmark checks so scouting can detect odometry errors without diagnostic truth. A complete startup policy and automatic extra-predator allocation should follow only after those components work on held-out sites. The current 80-map availability survey and mixed held-out results do not justify calling full-game integration complete.
