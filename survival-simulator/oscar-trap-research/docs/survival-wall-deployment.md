# Wall deployment through ordinary observations

17 September 2026. Personal Nordic AI Cup research, local only. The upstream engine is unchanged at `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`; all 18 vendored source files were checked against their recorded Git blob hashes. No competition requests, validations, submissions, publishing or pushes occurred.

**An observation-only crew can feed and replace a wall holder for 300 seconds on natural generated terrain. Finding and acquiring that opportunity during ordinary play remains unsolved.** The final controller held both development sites for five minutes with native food, ten paid births per site, five or six descendant generations, and no captures when additional predator spawning was disabled. It also held one site with native predator spawning enabled. Other predator cases failed, and six unarranged game starts produced no trap acquisition during their 180-second evaluation windows.

The final version has **36 retained simulation runs**, plus six focused regression checks. Earlier exploratory versions and failures remain in the same results directory. These small, selected and dependent cases are engineering evidence, not an estimated competition win rate.

## What changed from the orchard feasibility controller

[The controller](../survival/research/wall_deployment/controller.py) accepts only native `ObservationResponse` dictionaries and public simulation time. It receives no environment object, world coordinates, wall assignment, fruit energy, predator rest flag, hidden age threshold or fixture seed. Setup, diagnostics and native replay rendering do use true state and are separate from the policy.

Each initially disconnected founder starts an arbitrary local coordinate frame. Observed full obstacle edges correct movement estimates. Relative agent bearings and `rel_dir` register nearby workers and newborns into a shared local map. Known adjacent perpendicular edges identify a rectangular wall and its opposite face. Maps and path searches are bounded; disconnected established groups do not yet merge.

The controller recognizes a held opportunity when an observed predator is across a long wall edge and close enough to retain by hearing. It can also infer a possible resting window after 0.3 seconds of stationary predator observations, identify a thin wall from two adjacent edges, and route to its opposite face. Stationarity is a heuristic: the remaining rest time is hidden.

Foragers stay on the protected half-plane and search within 350 units of the anchor. Fruit observations contain neither IDs nor ripeness, so spatial tracks use time since first sighting as a conservative age estimate. Workers normally wait 15 seconds, accept younger fruit below 85 energy, and stop deliberately collecting near capacity. An incoming holder reaches the anchor before the outgoing holder leaves. Old workers retire, creating replacement slots. Births cost the native 100 energy and children receive native traits, energy and lifetimes. There are at most three active and six total members per connected group, with an overall ceiling of 18 living agents for ordinary games.

Several deployment bugs were fixed through retained failures:

- A first idle tick obtains native initial observations instead of moving without sensor data.
- A newborn inside the planner's clearance margin can move outward without crossing the wall.
- An inferred far face and a bounded correction distance prevent confusing equal-length wall faces.
- A newborn exactly on its parent's position uses mutual zero-distance bearings to recover its heading. The ordinary nonzero-distance formula had invented a second predator on the protected side and caused false withdrawal.
- Retirement renews old foragers before they occupy every active slot. A protected-side predator triggers abandonment of an unsafe trap.

The final controller hash is `484f61dfee082c05c5673ea1646b129cbd3d2614a35837e695c79aa30cc584e1`. The [summary](../survival/results/wall_deployment/summary.json) records source and policy hashes and compact run metrics. All general-harness final runs agree on that policy hash.

## Natural food and repeated replacement

These tests retain each seed's original generated biome map, obstacles, initial trees and fruit. New trees spawn at their native global locations; there is no arranged forest, injected fruit, energy refill or free replacement. The three founders retain their native 150 energy and randomly generated old-age thresholds.

**The initial trap and nearby crew positions are still arranged.** An evaluator selects the physically valid wall face with the highest deterministic initial protected-side tree/fruit ranking. This selection uses true setup coordinates and is not an online site selector. The controller receives only its subsequent ordinary observations. This distinction remains essential.

| Generated seed | New predators | Held through 300 s | Captures | Paid births | Maximum descendant generation | Handoffs | Native fruit eaten | Gross food energy | Living at 300 s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | Disabled | Yes | 0 | 10 | 5 | 8 | 130 | 6,781.4 | 4 |
| 8 | Disabled | Yes | 0 | 10 | 6 | 9 | 85 | 4,274.6 | 4 |

The crews pay 1,000 energy in births apiece. Their gross food collection averages 22.6 and 14.2 energy per second across the entire crew, including retirees and capacity waste. This demonstrates food-supported renewal at these patches; it does not prove that allocating that food and those workers to holding is optimal. Maximum sampled localization error was 0.47 units at seed 7 and effectively zero at seed 8. The 300-second cutoff is not evidence of indefinite sustainability.

Evidence: [natural-renewal-v7.json](../survival/results/wall_deployment/natural-renewal-v7.json).

## Additional predators and paired colony controls

The paired control uses the same foraging, reproduction, mapping and avoidance controller with wall recognition/acquisition disabled. Each pair begins with the same generated map and arranged three-founder/one-predator setup. Native tree and predator spawning remain enabled.

| Seed | Wall crew: survival / first containment loss | Colony control survival | Wall score | Control score | Wall / control captures |
| --- | --- | ---: | ---: | ---: | ---: |
| 7 | 300 s / none | 300 s | 306.64 | 287.85 | 0 / 13 |
| 8 | 286.9 s / 139.5 s | 300 s | 268.57 | 288.04 | 11 / 9 |
| 14 | 69.6 s / 68.8 s | 300 s | 63.44 | 278.66 | 6 / 23 |

Seed 7 retained the original predator throughout the horizon with ten births and eight handoffs while a second predator appeared elsewhere. Seed 8 withdrew when a predator approached the protected side and later went extinct. Seed 14 briefly operated the trap but lost the colony much earlier than its control. **This is a mixed result, not a demonstrated general score improvement.**

An additional stress suite deliberately places a second awake predator about 200 units into the protected side at the start. Both wall crews fail: seed 7 dies at 5.9 seconds versus 3.5 for its control; seed 8 dies at 98.8 versus 99.2. Withdrawal alone is inadequate when predators approach from both sides.

The native score is elapsed seconds plus gross fruit energy divided by 1,000, minus captured agent reserves divided by 100. Holding a predator has no direct score reward. Final score accounting agrees with the recorded meals and captures within rounding in every general-harness run.

Evidence: [native spawning pairs](../survival/results/wall_deployment/natural-native-v7.json), [protected-side stress](../survival/results/wall_deployment/protected-side-stress-v7.json).

## Acquisition without a rest flag or supplied wall map

A separate twelve-case suite starts one 150-energy agent at an arranged corner vantage, where native sensors can reveal two adjacent wall edges and a nearby predator. Physical boundaries, native motion, sensing, energy and aging apply. There is no food or additional spawning. The initial vantage, predator position, heading and rest phase are setup privileges; the policy is observation-only.

| Opportunity | Cases | Acquired and held through 60 s |
| --- | ---: | ---: |
| Fresh rest, width 30 or 35, length 70 or 100, headings ±0.3 | 4 | 3 |
| Late rest at the same geometries | 4 | 0 |
| Width 38, deliberately outside the acquisition filter | 4 | 0 |

For eligible widths the agent reached the opposite face at 1.6–1.7 seconds, using ordinary walking actions. A fresh predator woke at about 3.5 seconds. The late-rest predator woke at about 0.9 seconds, so the same inferred opportunity could be lethal. One fresh-rest case survived without sustained containment and did not count as a hold.

Evidence: [rest-acquisition.json](../survival/results/wall_deployment/rest-acquisition.json). This is observation-based acquisition in a prepared opportunity, not an encounter discovered during a full game. The earlier low-energy sacrificial-guide findings remain separate; this controller does not implement deliberate guide sacrifice.

## Unarranged generated-game starts

The final controller and its paired control were each run from native five-agent, zero-initial-predator starts for 180 seconds, with all native terrain, food and spawning. Seeds 11–14 were inspected in predecessor evaluations; seeds 21–22 were selected after the final controller was frozen.

**None of the six wall-enabled runs recognized or attempted a wall trap.** All final runs reached the 180-second cutoff. This is a bounded game prefix, not a completed 3,000-second game. The surviving agents are operating the ordinary colony component, so their scores do not establish a wall benefit.

| Seed | Wall-enabled score | Colony-control score | Wall acquisitions |
| --- | ---: | ---: | ---: |
| 11 | 183.99 | 154.00 | 0 |
| 12 | 183.50 | 182.27 | 0 |
| 13 | 194.81 | 194.57 | 0 |
| 14 | 189.61 | 189.83 | 0 |
| 21, fresh held out | 182.80 | 158.71 | 0 |
| 22, fresh held out | 188.33 | 188.59 | 0 |

The policies can differ during a short rest assessment even without committing to acquisition. Native object sets and divergent action/birth histories also produce different realized food and predator histories. These paired seeds are not fixed exogenous-event counterfactuals, and rerunning a seed is not guaranteed to reproduce every numerical result across processes or platforms. Retained replays document their actual runs.

Evidence: [evaluation seeds](../survival/results/wall_deployment/fullgame-v7.json), [fresh held-out seeds](../survival/results/wall_deployment/fullgame-fresh-heldout-v7.json).

## Limits and the next implementation

Recognition is asymmetric: a predator hears out to 60 units while a founder hears only 50. A 30–35-unit wall can retain pursuit while hiding the predator from its holder. An already acquired wide-wall opportunity may therefore be physically usable but invisible to this detector. Nearby natural fruit also does not guarantee a safe route or adequate recruitment after initial trees die.

Global exploration still accumulates localization errors away from landmarks. Established disconnected maps do not merge, and a single held anchor does not protect workers from another direction. Neither sensor-based rest timing nor emergency escape is reliable enough to commit a whole colony to a wall by default.

The next bounded iteration should add **an active scout that records two faces and a protected food route before allocating a holder**, carry observed predator rest transitions into a conservative acquisition deadline, and require a confirmed replacement route before releasing an elder. Evaluate this on newly selected complete generated starts, measuring encounters discovered, eligible opportunities rejected, successful acquisitions, uninterrupted hold duration and paired score. Keep the wall action optional until that end-to-end test improves on the same colony controller. Two-predator escape and map confidence are the next failure controls, not more sweeps of an already acquired wall.

## Code, reproduction and native replays

- [controller.py](../survival/research/wall_deployment/controller.py): ordinary-observation policy.
- [experiments.py](../survival/research/wall_deployment/experiments.py): natural-site fixtures, generated starts, legality assertions and passive accounting.
- [acquisition.py](../survival/research/wall_deployment/acquisition.py): corner/rest-window suite.
- [checks.py](../survival/research/wall_deployment/checks.py): unequal-heading and colocated registration, landmark correction, clearance exit and native hold checks.
- [adapter.py](../survival/research/wall_deployment/adapter.py): `make_policy(seed=...)` factory for the existing debugger. It performs no network I/O.
- [summary.json](../survival/results/wall_deployment/summary.json): compact final evidence; [experiment ledger](../survival/results/wall_deployment/experiment-ledger.json): exploratory and interrupted-run notes.

From the repository root:

```sh
survival/.venv/bin/python survival/research/wall_deployment/checks.py
survival/.venv/bin/python survival/research/wall_deployment/acquisition.py --record
survival/.venv/bin/python survival/research/wall_deployment/experiments.py --seeds 7,8 --seconds 300 --control wall --no-native-predators --output renewal-repeat.json --record
survival/.venv/bin/python survival/research/wall_deployment/experiments.py --seeds 7,8,14 --seconds 300 --output native-repeat.json
survival/.venv/bin/python survival/research/wall_deployment/experiments.py --mode generated --seeds 21,22 --seconds 180 --output generated-repeat.json
```

The harness also accepts `--policy-file survival/results/wall_deployment/policy-v7.py` for an archived source version. It asserts exactly one finite legal action per living agent per tick; children first act on a subsequent tick. Generation, paid birth, death, food and role histories remain in the detailed JSON files.

The following replays use the existing recorder's `native_render=True` and the original `Environment.draw`. Their final native images were visually inspected. Drag a `.json.gz` file into the existing local debugger; no new viewer or renderer was created and its files were not edited.

- [300-second natural seed 7 renewal](../survival/results/wall_deployment/natural-renewal-v7-seed7-wall.json.gz)
- [300-second natural seed 8 renewal](../survival/results/wall_deployment/natural-renewal-v7-seed8-wall.json.gz)
- [Observation-based resting-window acquisition](../survival/results/wall_deployment/rest-acquisition-native.json.gz)
- [Native final frame, seed 7](../survival/results/wall_deployment/native-renewal-seed7-300s.png)

All implementation changes are confined to `survival/research/wall_deployment/`, `survival/results/wall_deployment/` and this report. The vendor, frozen policies, debugger, orchard namespace and other strategy tasks' files were left unchanged.
