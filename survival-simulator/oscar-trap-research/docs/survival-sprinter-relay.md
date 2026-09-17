# Predator relay: acquire a turning circle, then replace its bait

17 September 2026. Personal Nordic AI Cup research, entirely local. Exact upstream commit: `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`.

**The substantial improvement is an observation-only controller that moves a bait inside the predator's turning circle. A prepared reserve can then replace it while it rests, and a real 75-energy newborn can be fed and take over. This establishes a useful occupation mechanism and replacement sequence. It does not yet establish an economical, reliable service from ordinary 150-energy starts or on generated maps.**

In the final six held-out single-predator cases, prepared pairs retained a bait as the actual target for **99.7% of active predator ticks**, with no bait or worker captures. Ordinary 150-energy pairs retained **77.7%**, suffered one bait capture, and sometimes left long target gaps. Worker life and food results remain mixed. The two-predator tests expose additional interference and captures.

The final evaluation contains **71 recorded cases**: 36 held-out comparisons, 9 equal-population comparisons, 18 two-predator comparisons, 4 longer resource tests and 4 recorded examples. Another 44 exploratory cases preserve the unsuccessful steps leading here. An incorrectly dispatched baseline is explicitly excluded.

## What changed from the prior relay

The previous straight-running relay could hand over once but often let the reserve become the wrong target, lost attention, or exhausted the outgoing bait. Three iterations here led to a different movement pattern:

1. **Radial staging and joint pursuit prediction failed.** Prepared pairs could approach full retention briefly, but moving both agents along the pursuit trajectory spent too much energy and carried them into boundaries. Eighteen stronger center/assignment penalties did not solve this: 30-second retention ranged roughly 45–84%, with many captures. Longer survival sometimes came from losing the predator.
2. **Use the predator's turning limit.** During close pursuit the original engine limits each turn to 0.3 radians, then moves in that turned direction. At a 15-unit step, constant turning has radius `15 / (2 sin(0.15)) = 50.188`. A bait near its center can remain within the predator's 60-unit hearing radius, outside the 15-unit capture radius, while the predator keeps circling. At an 11-unit step the corresponding radius is 36.805. This is the original chase rule, without walls or water.
3. **Move the reserve during an observed pause.** Keep the active bait near the estimated circle center; stage and forage outside the orbit. After consecutive stationary observations, bring a prepared replacement toward the center, then send the outgoing bait back to safe forage. Confirm success using the predator's actual selected target, not the controller's role label.

Eight early orbit-acquisition fixtures, spanning 150/500 energy and four headings, retained attention for all 60 seconds without captures. Combined bait movement/turning cost was about 51–91 energy, excluding passive living costs. Those promising pilots justified the longer and more varied tests below; they are not pooled into the final means.

A separate privileged geometry check placed a stationary bait exactly at the theoretical center. The original predator maintained a 50.188-unit gap for 25 full-speed ticks to numerical precision. That check proves the geometry; the evaluated controller must acquire its center from observations.

## Controller and information boundary

The controller reads normal cached agent states and time. It reconstructs relative positions and headings from sightings, establishes an arbitrary coordinate frame, and integrates commanded movement. It estimates predator movement from consecutive sightings and accounts for the observation being one predator move old. It never reads true map coordinates, predator energy, or the engine's resting flag.

The state sequence is:

- **Acquire and hold:** select a turning side, estimate the circle center, and choose among bounded movement candidates with capture-distance penalties. Keep looking toward the predator to maintain observations. Retain the last moving center during an observed pause.
- **Stage and recover:** keep the reserve about 120 units from the center. Pursue observed food only outside the orbit exclusion zone. After a handoff, the outgoing bait returns to this role.
- **Prepare and request:** require incoming energy of at least `max(140, max_energy / 5 + 25)`. Request a handoff when the active bait falls below 120, or after age 60 when a suitable younger replacement exists. Use a ten-second dwell between planned handoffs.
- **Approach and replace:** start the approach after at least two stationary sightings. Once the incoming bait is within eight estimated units of the center, release the outgoing bait. The predator's closest-perceived-agent rule still decides the actual target.
- **Renew:** a distant reserve with energy above 225, age above 25 and recently observed food can pay the native 100-energy reproduction cost. The real newborn begins at 75 with native random placement and mutations. It must forage before planned activation. The nursery temporarily permits a third live bait per pair.

Initial bait IDs and pair ownership are assigned roles. All starting positions, trees, boundary walls and predator placement are **privileged experimental fixtures**, not knowledge passed into the policy. The policy therefore has an observation-only interface, but deployment still needs map acquisition, localization through collisions, obstacle handling and predator association. Apparent immobility near an obstacle is not reliable evidence of rest.

## Final single-predator comparison

Six held-out seeds, 101–106, each run for 120 seconds at both 150 and 500 bait energy. Starting gaps were 45, 80, 55, 90, 110 and 50; headings were −0.7, 0.4, 1.8, π, −2.2 and −1.3 radians. Reserve distances were 65–150 with varied bearings. Each condition had the same two 150-energy worker roles using a common reactive forage/escape controller, and the same arranged tree positions.

| Bait starting energy | Controller | Mean active target retention | Bait captures / six runs | Worker captures / six runs | Mean worker-seconds alive | Mean food eaten by baits / workers |
|---|---|---:|---:|---:|---:|---:|
| No bait | Two workers | 0% | 0 | 4 | 174.3 | 0 / 377.1 |
| 150 | One orbit bait | 67.1% | 0 | 3 | 157.1 | 37.4 / 321.9 |
| 150 | Two-bait relay | 77.7% | 1 | 3 | 159.3 | 483.0 / 323.2 |
| 500 | One orbit bait | 97.4% | 0 | 2 | 161.7 | 40.2 / 321.9 |
| 500 | Two-bait relay | **99.7%** | **0** | **0** | 142.4 | 590.1 / 407.1 |

Food figures are nominal nutrition consumed, not guaranteed retained energy. The raw results separately report received energy and capacity waste. Worker-seconds measure life, including idle time, rather than productive work. The no-bait rows are identical across nominal bait-energy settings and are shown once.

The prepared relay eliminated captures in this sample, but its workers lived **31.9 fewer combined seconds** than the no-bait workers on average. They mostly died from energy exhaustion. Its additional food collection therefore does not establish improved worker survival or colony economics. For ordinary starts, the relay improved retention over one bait but did not reduce worker captures, and worker life remained below the no-bait control.

The longest uninterrupted sequence of active ticks without a bait target was **22.5 seconds** for ordinary pairs and **0.4 seconds** for prepared pairs. Gaps include acquisition, accidental worker targeting, disengagement and time after bait death. Sleeping time is excluded. A surviving agent after disengagement receives no occupation credit.

Prepared energy is an experimental condition. Filling two ordinary founders from 150 to 500 would require at least **700 additional nutrition**, before collection costs or waste. That preparation process was not simulated. Births and food during the runs are charged normally.

## A real newborn takeover

The final `prepared500-newborn-handoff` replay is the cleanest complete sequence:

- Parent pays 100 and creates agent 4 at **25.0 s**, with exactly **75 energy**.
- Native mutation raises its capacity to 739.416. Its readiness threshold therefore becomes **172.883**, rather than the ordinary 140.
- It reaches **204.05 energy at 34.8 s** by eating physical fruit.
- The predator actually changes target from agent 0 to agent 4 at **62.2 s**.
- Retention is **100% of active ticks over 120 s**, with no bait captures and no target gap.

This run consumed 1,039 nominal nutrition in the bait group, spent 543.43 on movement/turning, paid 100 for the birth, and peaked at three baits plus two workers. Both workers eventually starved. It proves the replacement sequence, not a net worker benefit or perpetual population renewal.

The ordinary-start replay deliberately preserves a failure: it completes a founder-to-founder handoff, produces a 75-energy child, but fails to feed that child to readiness. The child starves; the remaining baits later starve as well. The controller must not count that birth as successful renewal.

## Resource duration and opportunity cost

All ordinary tree cases start with six mature bait-area trees plus one per worker. They produce fruit and die using native rules; new tree generation is disabled in those fixed-stand fixtures. Finite-fruit exploratory cases contain eight real ripe fruits around the bait area, not energy injections. Every fixture retains the 1600×1200 physical world and its boundary walls. New predator spawning is disabled to isolate the tested service.

In two additional 180-second `renewing` cases, native stochastic tree generation was enabled:

| Starting energy | Retention | Largest active gap | Bait food consumed | Birth cost | Baits alive at 180 s |
|---|---:|---:|---:|---:|---:|
| 150 | 82.5% | 23.7 s | 6,142.6 | 100 | 1 |
| 500 | 99.6% | 0.5 s | 5,087.6 | 200 | 2 |

These are **one seed each**, without workers. The ordinary surviving bait did not continuously occupy the predator. The prepared case demonstrates longer operation with natural replenishment, at a substantial resource cost; it does not prove unlimited sustainability. The original tree-generation rate also decays over time.

The equal-initial-population comparison assigned four founders as four workers, one bait plus three workers, or two baits plus two workers. Across three ordinary-energy seeds, mean combined worker life was **306.7 / 258.2 / 194.4 seconds**, and worker captures totaled **5 / 3 / 0** respectively. The relay prevented captures but consumed an average 844.7 nutrition in its bait group, while its workers collected 156.5. The four-worker control collected 1,080.3. Lower capture counts alone do not justify removing two foragers.

## Two predators

Three new seeds, two assigned predator locations 350 units apart, and one bait or one pair allocated to each. Each case retains two worker roles and runs for 120 seconds. This tests already assigned local controllers, not global predator discovery or optimal assignment.

| Starting bait energy | Allocation | Mean active retention across predators | Total bait / worker captures | Mean worker-seconds alive |
|---|---|---:|---:|---:|
| No bait | Two workers | 0% | 0 / 5 | 100.1 |
| 150 | One bait each | 69.7% | 0 / 0 | 152.6 |
| 150 | One pair each | 85.5% | 1 / 1 | 151.1 |
| 500 | One bait each | 97.5% | 0 / 0 | 152.6 |
| 500 | One pair each | 96.9% | 5 / 1 | 166.0 |

The prepared paired strategy is **not reliable with nearby predators**. A high overall target fraction can coexist with bait deaths because another bait remains targeted. Reserves, newborns and outgoing agents can interfere with the other orbit. The current identityless track association and local safety calculation do not provide joint multi-predator safety.

## Measurement, failures and reproducibility

Final measurements observe the original engine's actual `Predator.step`, `kill_agent` and `remove_fruit` calls, then delegate exactly once with unchanged arguments. They do not infer capture from positive remaining energy or count a target before the native starvation pass. Source-line assertions distinguish capture, starvation, food consumption and rot. Observers never supply information to policy decisions or draw randomness.

Every run saves complete parameters, source and controller hashes, one-second diagnostic traces, controller events, actual target changes, gap lengths, population cost, food consumption/receipt/waste, births, mutations and deaths. Actions are checked for one finite command per living observed agent. The read-only recorder stores original decision observations alongside actions.

Earlier result files remain for failed-hypothesis review. The first baseline labelled `orbit_single` accidentally used the old pursuit controller; `heldout-invalid-single-baseline.json` is excluded. The final comparison dispatch was corrected and rerun. Earlier exploratory instrumentation and fixture ordering also differ, so historical numbers are not interchangeable with the final tables. Paired policies share seeds and starting arrangements, but births and policy-dependent engine RNG consumption change future food streams. These are small controlled samples on the local macOS interpreter, not official score estimates.

Verification passed the 25-tick turning-circle geometry check, 50 ticks of observation immutability and action-contract checks, and schema/time-order checks for all saved replays. Vendor code, existing policies, the dedicated-sprinter work, and the debugger were not edited. No hosted validation, competition calls, submissions, publishing or pushes occurred.

## Artifacts and next work

- [Executable controller and reproduction instructions](../survival/research/sprinter_relay/README.md)
- [Final machine-readable summary](../survival/results/sprinter_relay/summary.json)
- [Prepared newborn takeover replay](../survival/results/sprinter_relay/prepared500-newborn-handoff.json.gz)
- [Ordinary handoff followed by failed newborn preparation](../survival/results/sprinter_relay/ordinary150-handoff-starvation.json.gz)
- [Ordinary food-exhaustion replay](../survival/results/sprinter_relay/ordinary150-food-exhaustion.json.gz)
- [Two-predator replay with replacements and a worker loss](../survival/results/sprinter_relay/two-predator-pairs.json.gz)

Use the existing Survival Lab file loader. No new viewer or manifest was created.

The next useful iteration is to retain the orbit mechanism while improving **food and population allocation**: feed young replacements before aging reserves, stop expensive elder foraging from consuming nursery food, and use a reserve only when the active bait's remaining service warrants it. Multi-predator work needs a shared danger model and disjoint recovery routes before adding more pairs. Generated-map acquisition and collision-corrected localization remain untested. The evidence supports continuing this controller family; it does not support deploying the current relay as a generally beneficial colony policy.


## Replay availability update

Every relay harness now records every completed run by default, including controls, failures and sweep cases. The original report statistics above remain unchanged. New runs publish unique replays and per-run metrics, so a later controller version cannot overwrite an earlier recording.

The audit found 147 metrics-only cases across the final and historical result files. All 147 were reproduced locally and clearly labelled as **new reproductions**, not recovered original footage. Four reported case rows already had their original recordings. The closest preserved controller was used for historical cases whose exact full hash was unavailable; these limitations appear in the replay metadata and case receipts.

**All 156 relay replay files were confirmed in the live Survival Lab catalog at 2026-09-17T13:49:02.663397+00:00.** This total also includes the older preserved demonstrations, two recorded verification checks, and a new native-rendered takeover demonstration. Native rendering uses the upstream engine's drawing code. No viewer, catalog, manifest or vendored engine file was edited by this task.

- [Open live Survival Lab](http://127.0.0.1:9053/)
- [Complete case-to-replay coverage and live discovery audit](../survival/results/sprinter_relay/replay-audit.json)
- [Native-rendered prepared takeover reproduction](../survival/results/sprinter_relay/replays/orbit-v96d8a373a2-seed101-20260917T134718357371Z-f5582ba258.json.gz)

The native takeover reproduction retained attention for 100% of active ticks over 120 seconds. It is a fresh run of the saved configuration. Original metrics, original recordings and failed cases are preserved separately.
