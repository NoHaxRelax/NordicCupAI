# Water baiting: tuning, observations, food and robustness

Local investigation, 17 September 2026. Upstream commit `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`; original movement, perception, predator logic, collisions, energy and fruit lifecycle are unchanged. No external game API or score submission.

**The water idea works in controlled conditions. The first 18 failures were insufficient evidence against it.** A controller using cached public observations held a predator entirely in a 50-unit river for 63.5 seconds before the first bait starved. A more tolerant variant, supplied with a deliberately rich fruit bank and young enough baits, achieved 120 seconds in several constructed cases. The improved land-to-water acquisition procedure worked through first depletion in 11/12 new straight-river fixtures. However, all three unblocked tests at sampled natural river sites failed through capture within 9.2–17 seconds. This is a demonstrated mechanism with a substantial geometry and deployment gap.

This bounded iteration covers arranged containment, land-to-water acquisition, and an initial generated-map transfer test. Natural food sustainability, replacement, and a complete policy remain follow-up work.

## Evidence and reproduction

Code: `survival/research/water_tuning.py`.

```sh
survival/.venv/bin/python survival/research/water_tuning.py --search
survival/.venv/bin/python survival/research/water_tuning.py --tune
survival/.venv/bin/python survival/research/water_tuning.py --observe-tune
survival/.venv/bin/python survival/research/water_tuning.py --held-out
survival/.venv/bin/python survival/research/water_tuning.py --resources
survival/.venv/bin/python survival/research/water_tuning.py --acquisition
survival/.venv/bin/python survival/research/water_tuning.py --geometry
survival/.venv/bin/python survival/research/water_tuning.py --native-sites
```

The main staged searches evaluated 72 initial configurations, 280 predictive configurations, 72 heading-triggered configurations, and 321 observation-based configurations, each on three training fixtures: 2,235 training runs. Additional lifetime, observation-lag, speed, food and held-out tests are saved separately. These are deterministic controlled experiments, not independent competition evaluations.

Results are `survival/results/water-tuning-*.json`. The `preboundary-*` files are retained as **rejected exploratory evidence**: the original helper had visible boundary edges but lacked physical boundary rectangles, so drifting predators could benefit from simple coordinate clamping. All promoted results use the actual four 30-unit physical boundary obstacles. The successful 50-unit observed case stayed around x=778–820, y=570–615, far from those boundaries.

Playable recordings, suitable for the local debugger's import control:

- `water-tuning-observed-success.replay.json`: 50-wide river, 150 energy per bait, no food, all-water attention until starvation at 63.5 seconds.
- `water-tuning-observed-failure.replay.json`: wider-river failure using the same narrow-site controller.
- `water-tuning-fed-success.replay.json`: 120-second observed-controller case with deliberately abundant mature fruit and maximum age 120.
- `water-tuning-fed-success-native.replay.json`: the same prepared-food demonstration using the original simulator renderer.
- `water-tuning-acquisition-shore-native.replay.json`: original-renderer demonstration of both baits beginning on land, dry-shore launch, entry and containment until depletion. This is a known straight-river fixture; it includes the subsequent failure after the first bait starves.

## What changed from the first attempt

The original controller alternated between fixed near and far positions separated by 50 units. That causes the replacement bait to arrive too late and wastes energy. The revised controller instead:

1. Tracks the predator's position and heading.
2. Predicts its approach to a shore and switches early, with hysteresis and a minimum hold time.
3. Places the desired bait near its own shore and makes the other bait **just far enough away that the desired bait is the nearest detectable agent**.
4. Lets both agents adjust their positions along the banks, or remain near a fixed lateral anchor, depending on the variant.
5. Continues an emergency escape when a handoff fails, rather than stopping at the nominal retreat point.

The real predator selects the nearest currently detectable agent, not a previously assigned target. Across the recorded successful case there were 40 actual predator target switches; the intended bait was the selected bait in 94.6% of active ticks. Minimum predator/bait center distance was 22.24, compared with a 15-unit contact threshold.

Five trajectory families were explored: adaptive retreat across opposite banks; sideways changes along the banks; leading the predator down the river; a single bank runner; and switching when the predator completes a prescribed part of its turn. Opposite-bank adaptive retreat produced the best observation-based results. Merely circling or moving sideways is not sufficient.

## Observation access matters

There are two explicitly different controller settings:

- **Privileged diagnostic:** receives the current true predator position and heading.
- **Observation-driven containment:** receives cached `distance`, `angle`, and `rel_dir`; reconstructs predator pose from those observations; tracks its own movement using public traits and current biome; predicts the predator's next movement from observed displacement and the known chase-turn rule. It does not read true predator energy, resting state, position or heading when choosing actions.

Both settings still assume known initial agent poses, a straight river of known width, and a shared local coordinate frame. The observation-based setting therefore controls an arranged/mapped site, including a tested land-to-water entry procedure, **not a complete policy that discovers the site from ordinary game startup**.

The engine caches agent observations before moving predators. Ignoring that one-tick lag made the otherwise successful controller fail in roughly 1–2 seconds: the pursued bait stayed one movement behind the distance required for a handoff. Explicit velocity/model extrapolation and larger initial clearance were necessary. Near a bank, a failed switch also requires escape rather than waiting for the second bait.

The observation predictor assumes the near-range direct-chase branch. It is less trustworthy when the predator is farther away, becomes visually occluded, leaves the river or changes its movement mode. Those are remaining practical limitations, not privileged fields silently fed to the controller.

## Best no-food demonstration

Configuration: cross-bank control; fixed lateral anchor; target-distance reserve 12; switching threshold 3; no positional lookahead; maximum ordinary request 10; emergency distance 22; 3-tick minimum role hold; cached-observation velocity prediction. River width 50. Two baseline agents started at 150 energy each. Predator initially at the river center facing one shore.

- Predator remained in water for every tick through 64 seconds.
- A bait was detectable for every active predator tick.
- No captures occurred.
- The first bait starved at 63.5 seconds; the other at 64.4 seconds in the longer run.
- At 64 seconds the recorded run had spent 171.84 total movement/turn energy plus roughly 128 total passive energy.
- Predator stayed near the starting site, rather than being pinned by a map boundary.

This establishes feasibility for one geometry. It does **not** establish general robustness: this narrow-site configuration failed most perturbed held-out starts. Increasing its initial energy to 300 delayed starvation but a dynamic failure eventually occurred around 76.7 seconds; energy alone does not repair every unstable controller.

## More tolerant controller and held-out tests

The more tolerant observation-based configuration selected from training uses:

| Parameter | Value |
|---|---:|
| Target-distance reserve | 18 |
| Switching threshold about river center | 9 |
| Predictive horizon for shore switching | 3 ticks |
| Maximum ordinary movement request | 9 |
| Minimum role duration | 4 ticks |
| Along-bank position | 70% of predator displacement from initial anchor |
| Emergency escape distance | 22 |
| Shore margin | 2 |
| Observation prediction | Chase-turn model, speed inferred from observed motion |

The training fixtures had widths 50/60, headings 0 or 0.2 and one small offset. After selecting the configurations, we evaluated 52 new straight-river fixtures per controller: widths 30,40,44,48,54,56,64,72,90,100,120,160,200 and four different heading/offset combinations. Width 30 is a stress test below the generator's nominal 40-unit minimum. Headings included ±0.35 and ±π/2; some starts also had x offsets ±3 and y offsets ±12.

For the tolerant observation controller, among the **20 held-out fixtures with widths 40–56**:

- 19/20 spent at least 98% of elapsed time in water.
- 19/20 had a bait detectable in at least 99% of active predator ticks.
- All reached their first bait depletion without a predator capture. First starvation was approximately 44.7–52.1 seconds.
- Three predators subsequently caught the remaining bait after the first bait starved.
- One 40-wide case spent only 89% of time in water, so it is not counted as dependable containment.

There were **zero full 60-second successes with both unfed baits alive** across the 52 held-out fixtures. In the narrow band, depletion was the main obstacle. Wider rivers also produced acquisition/sensing failures: a bait on the other bank often cannot enter the predator's 60-unit omnidirectional detection range. Survival after the predator wandered away is not counted as containment.

The privileged heading-triggered controller achieved four full 60-second holds in its 52 held-out fixtures, but failed many other starts. Current control remains geometry-sensitive even with perfect information.

## Why “just faster than 4.5” was insufficient

Water reduces a predator's sprint movement to 4.5 per tick. A dry-bank bait can certainly outrun that by requesting 4.6. However, to transfer pursuit across two opposite banks, the pursued bait may need its distance to increase as fast as the opposite bait's distance grows: close to twice the predator's cross-river motion, around 9 units per tick.

Holding the tolerant controller's other parameters fixed:

| Maximum ordinary request | Result across widths 44/50/56/60 |
|---:|---|
| 4.6 | Capture in 0.8–0.9 seconds |
| 6 | Capture in 1.8–2.1 seconds |
| 7 | Three early captures; one later capture |
| 8 | Two depletion-limited holds; two captures |
| 9 | All four lasted until bait depletion around 52 seconds |
| 10 | Similar depletion-limited performance to 9 |

The request is a ceiling, not constant full-speed motion. These controllers remain within baseline walking speed for ordinary actions; sprint mutations are not required for the demonstrated steady-state hold.

## Food, age, and newborn-energy replacements

The tolerant controller uses roughly 2.9 energy per simulated second **per bait**, including passive cost. One fully mature 60-energy fruit therefore buys roughly 20 seconds if the bait can collect it without breaking control.

No-food energy sensitivity at width 50:

| Initial energy per bait | Approximate first depletion |
|---:|---:|
| 75, normal newborn energy | 26.1 seconds |
| 150, starting-agent energy | 51.7 seconds |
| 300, a fed reserve | 99.3 seconds |

Newborn energy is below the normal 100-energy sprint cutoff, but the ordinary 9-unit walking request is still available. It is the short energy budget and failed-handoff escape margin that matter. These tests arranged new agents in place; they **do not yet verify replacement travel and handoff**. Reserves waiting on the sidelines also lose energy at 1 per second, so early pre-spawning is not free.

For food tests we deliberately placed mature native fruits on both bank lines. Their energy, radius and age match mature fruit, and normal native eating and rotting rules remain active. This is an abundant prepared fixture, **not proof of naturally occurring food density**. Mature fruit starts at age 40 and rots after roughly another 30 simulated seconds; there is no secret replenishment or injected energy.

Across 12 arranged food-site variants with maximum agent age 90:

| Initial fruit spacing along each bank | Mean run duration | Longest run |
|---:|---:|---:|
| No fruit | 49.3 s | 56.9 s |
| 100 units | 66.3 s | 75.5 s |
| 50 units | 77.1 s | 91.6 s |
| 25 units | 88.7 s | 101.5 s |

Those durations include failures and are not equivalent to successful confinement. In a separate 12-case upper-age test with 25-unit fruit spacing and maximum age 120, **7/12** held both agents alive for the full 120 seconds with at least 99% water time. One additional run survived 120 seconds but only kept the predator in water 86% of the time, so it is excluded. Other cases failed or depleted earlier.

This supports the user's intuition that fruit-rich shores and younger replacement baits can extend control. It also explains why a complete strategy must include access to food, replacement timing and safe entry, rather than simply assigning two agents permanently.

## Acquisition from land: a discrete shoreline launch matters

Both baits and the predator now begin on the same land bank. One bait crosses the river to draw pursuit into water; the other waits 75 units along the bank, approaches after detecting entry, and takes over pursuit once the crossing bait retreats beyond the opposite shore. Entry and switching decisions use the cached public predator observations. Initial poses and river geometry remain known fixture information.

An early direct crossing often failed. The important correction was to take a short first step exactly onto the dry shoreline, then sprint into the water. Native movement uses the biome at the **start** of each movement, so a 20-unit sprint starting on dry ground moves 20 units even when it ends in water. A step starting in water moves only 6. The engine itself was unchanged. This is a useful gameplay edge case, although it requires accurate bank localization.

After 36 initial acquisition cases, a 16-case perturbation check and 15 shoreline-launch probes, the improved procedure was frozen for 12 new fixtures. These combined widths 46/54/58 with predator distances 37/53/67/83 beyond the near bank, headings offset by ±0.15 or ±0.3 radians, and lateral offsets ±8 or ±14. Each bait started with 150 energy; there was no food.

- All 12 entered the containment phase in 0.8–1.1 seconds.
- **11/12 maintained 99.4–100% water time while both baits remained alive**, after acquisition. A bait remained detectable throughout those periods.
- Their first loss was starvation at 33.7–40.4 seconds from startup. Acquisition consumed roughly 37–53 energy from the crossing bait, leaving 97–113 at the handoff; the other retained approximately 145–147.
- The width-46, distance-83, heading −0.3, lateral-offset −14 case escaped and captured a bait at 6.5 seconds. Completing the entry phase is therefore not itself proof of stable containment.
- The remaining bait was subsequently captured after the first starvation in all 11 depletion-limited cases. No case achieved a full 60-second two-bait hold.

These results support acquisition in a suitably mapped straight site. A replacement schedule would need to account for the crossing cost and act before about 30 seconds in these fixtures. That is a planning implication, **not a validated live replacement procedure**. Results and exact inputs: `water-tuning-acquisition-final-held-out.json`.

## Natural river availability and first transfer test

An oracle survey used 40 genuine generated maps, seeds 1–40. The deliberately conservative site criterion required a 120-unit corridor with at least 40 units of common water across nine sections, at most 64 units of total bank envelope, and full-speed land on both banks. Both horizontal and vertical orientations were checked. Only **2/40 maps** contained qualifying corridors, yielding four coarsely deduplicated sites; nearby candidates can overlap. This is evidence that the current straight narrow-site requirement is restrictive, not proof that other curved sites are unusable. The survey reads complete terrain and is not an observation-driven discovery policy.

Each candidate was then checked in a complete native generated environment, retaining its terrain, physical obstacles, initial trees/fruits and ongoing food/tree dynamics. Two baits and one predator were arranged at the site with 150 bait energy; only additional predator spawning was disabled. A known local coordinate transform supplied the controller's initial map frame. Trials stopped at first bait loss.

| Map/site | Outcome | Predator water time | Fruit energy collected |
|---|---|---:|---:|
| Seed 23, horizontal corridor at x=1200 | Capture at 17.0 s | 93.5% | 48.2 |
| Seed 23, horizontal corridor at x=1320 | Capture at 9.2 s | 81.5% | 0 |
| Seed 37, vertical corridor at y=900 | Capture at 15.9 s | 90.6% | 0 |
| Seed 37, vertical corridor at y=960 | Skipped: initial creature positions intersect physical obstacles | n/a | n/a |

All three tested failures occurred with substantial bait energy remaining; starvation was not the cause. Predator attention stayed above 99%. Thus merely collecting natural food did not solve the more immediate containment failure. Native terrain boundaries depart from the controller's fixed bank lines, and its dead reckoning currently has no collision correction. These are concrete transfer limitations; this small batch does not isolate their individual causal contributions. Source data and trajectories are in `water-tuning-generated-geometry.json` and `water-tuning-native-sites.json`.

No sustained natural orchard/replenishment result or native birth/replacement handoff is claimed. Next work should make control follow mapped bank shape and handle blocked movement, then validate actual food and replacement under those conditions. Repeating only straight-fixture parameter sweeps would not resolve the demonstrated native-map failure.

## Current conclusion

Water containment has a working observation-based mechanism, a useful shoreline movement edge case, and a tested entry procedure for known straight sites. It can work for a minute without food and for two minutes in sufficiently rich constructed food sites. It is **not ready as a generated-map strategy**: all three natural-site transfer trials failed quickly, and live replacements remain untested. The next iteration should address bank shape, localization and collisions, then natural feeding and handoff. Arranged-fixture performance must not be reported as a generated-map score advantage.
