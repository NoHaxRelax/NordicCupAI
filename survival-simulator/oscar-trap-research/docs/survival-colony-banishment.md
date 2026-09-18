# Managed colonies and temporary predator banishment

17 September 2026. Personal Nordic AI Cup strategy research. Exact source:
`acfc31a4003a5f91bf11032a02cd98c178ddbd7e`; interpreter:
`survival/.venv/bin/python`. All work is local. No competition API, validation,
score submission, publishing or push was performed. The vendor, existing
policies, other investigations and debugger were not edited.

**Result:** a working managed colony and repeatable guide service, with a
promising result in arranged obstacle orchards: 4/4 held-out score wins, median
+11.92 points. Open orchards and river fixtures were inconsistent. On eight
ordinary generated maps, the conservative guide activated on only one seed;
it has not established a reliable general improvement over the nursery baseline.

## What was built

This iteration replaces the earlier stationary-worker fixture with a managed
colony. Workers collect real fruit, share visible-fruit claims, select a rich
observed tree region, spread across nearby trees, scout when hungry, flee local
predators and reproduce within a population ceiling. The default ceiling is
four; an unmodified generated map still starts with its normal five founders.

Both paired arms reserve food/energy for a healthy individual. With banishment
enabled, the controller chooses an adequately fueled candidate near an arriving
predator, attempts to acquire its attention, leads it away from the orchard,
looks for repeated observed immobility, escapes beyond the release location,
keeps workers outside a temporary exclusion radius and returns by a detour.
The process can repeat using another fed worker. Energy is never injected.

The final version rejects old guides, multiple visible predators, candidates
near known edges, and known obstructed initial escape routes. An unsuccessful
acquisition has a four-second limit. This is temporary displacement, not a
permanent occupation policy. No controller uses hidden predator energy or sleep.

Implementation and reproducible commands are in the
[research folder](../survival/research/colony_banishment/README.md). Intermediate
algorithms, executable runners and
[failed hypotheses](../survival/research/colony_banishment/ITERATIONS.md) are kept.

## Evidence and scope

Results below use the corrected, frozen version-5 protocol. Training used seeds
1–3. Version 4's earlier held-out evaluation is preserved separately because it
exposed order-sensitive mapping decisions, including large differences when no
guide was selected. Version 5 canonicalizes observation order and rounds DTO
numbers/actions to six decimals. It keeps version 4's strategy and parameters.
The corrected protocol was frozen before a fresh set of cases in
[heldout-v5-freeze.json](../survival/results/colony_banishment/heldout-v5-freeze.json):
minimum nominal guide energy 300, plus a gate requiring at least 250 above its
sprint cutoff; population ceiling 4; remote release radius 450; guide lead limit
18 seconds; rest-based release; conservative risk gate enabled.

The final generated-map suite uses seeds 201–208 and a 600-second horizon. Each seed
runs the same colony with banishment off/on and the existing nursery policy.
These maps retain default dimensions, obstacles, terrain, five 150-energy
founders, fruit, trees, reproduction, aging and stochastic predator spawning.
No arranged predator is introduced on generated maps.

The final managed-orchard suite uses held-out seeds 21–24, three terrain variants and
300 seconds. The fixture has 12 initially productive trees, 28 finite fruits,
four normal 150-energy founders and normal-range aging thresholds. Initial
tree growth is uniform 20–35; initial fruit growth is uniform 0–40. Original
fruit maturation, rot, tree death, new trees and new predators remain enabled.
Two scheduled arrivals at 20 and 100 seconds use normal zero-energy, initially
resting predators. Their exact positions and headings are saved in every run.

The obstacle variant adds ten ordinary-size obstacles beyond the orchard. The
river variant adds the same obstacles and a 100-unit river strip at x=1040–1140.
These are geometry fixtures, not default generated maps or solved river
isolation. The observation-only policy is not given those coordinates. Separate
oracle pairs on seeds 21–22 receive true **agent poses only**, with the same
cached observations; their results must not be merged with deployable runs.

### Observation boundary

The policy interface is ordinary cached observation dictionaries and simulation
time. Local maps start in arbitrary frames. Odometry uses requested movement
and the reported current biome; visible trees/edges correct translation, and
identified agent observations connect coordinate frames. Recent observed edges
support local avoidance. Neither full map geometry nor absolute positions are
silently supplied. Unseen obstacles, ambiguous landmarks and collisions can
still corrupt this map. Returned predator observations retain the engine's
normal one-move delay.

Diagnostics may read true geometry, energy and resting state. They measure
agent/worker survival time, harvest energy, score, positive-energy captures,
worker proximity below 100 and 50 units, actual predator targeting, guide
movement/turning expenditure, release locations, return times, spawning,
boundary exposure and map error. Occupation is recorded as predator-seconds
actually selecting the guide; a guide surviving or a predator wandering far
away is not counted as occupation.

Worker metrics exclude whichever individual is currently the guide. Reduced
raw worker exposure can therefore partly reflect lost worker time. The paired
summary also reports exposure per worker-second and all-agent captures; guide
time, movement energy and food consumption expose the opportunity cost.
Capture classification uses positive remaining energy at removal; a rare
negative-energy capture could be classified as energy death. Harvest accounting
uses score changes adjusted for time and recorded capture penalties, and omits
the identical first empty-step harvest in paired arms.

The paired colony control keeps the same population ceiling and healthy-agent
reserve. It measures the incremental cost of dispatching the guide. It does
not isolate the cost of preparing that reserve in the first place. The nursery
comparison includes broader policy differences, including its population cap.

Same-seed pairs share initial conditions, not an identical future random stream.
Different actions consume the engine's shared RNG differently. The earlier
version-4 comparison also exposed substantial null-arm divergence from unordered
observations and order-sensitive map choices. Canonical ordering and numeric
precision address this in version 5; separate null-arm and shuffled-observation
checks are saved. The version-4 score differences are not reliable evidence of
banishment benefit and are excluded from the final tables. Treat the
paired bootstrap intervals as descriptive, not confirmatory causal estimates.
The horizon censors surviving colonies; none of this is a full 3000-second or
official Linux evaluation.

## Concrete progress beyond the prior fixture

The recorded version-3 forest seed-2 run demonstrates repeated service by fed
workers. Agent 0 was selected at 25.6 seconds with 468.89 energy and released at
34.0 seconds with 314.51. It separated at 35.9 and returned at 40.6. The predator
reentered a 250-unit radius of the release-time worker centroid at 40.9, only
6.9 seconds after release. Agent 2 was then selected at 43.8, released the same
predator at 61.7 and completed its return at 75.3. The second return-to-colony
event was not observed before the 140-second recording ended, which is a
censored result rather than permanent banishment.

The colony harvested 139 fruits worth about 6,126 energy during this run,
replaced three agents and still had four alive at 140 seconds. Guide jobs took
53.2 agent-seconds and about 576.5 movement/turning energy. Predators actually
selected the guide for 20.9 active predator-seconds. A second predator later
caught a new guide in a corner at 112.7 seconds, and two workers were lost.
This is a working multi-cycle managed-colony demonstration with a visible
failure, not proof of net benefit or continuous occupation.

Replay: [repeat release, return and later corner failure](../survival/results/colony_banishment/iteration3-failure-forest-2-banish.json.gz).
All 18 vendored source files matched their pinned Git blob hashes. Interface
checks cover serialized observation-only decisions, shuffled observation order, unchanged inputs, one
finite action per agent, relative-frame alignment and non-mutating diagnostic
sensing. Replay files use the existing recorder and format; no new viewer,
renderer or manifest entry was introduced.

The map is small relative to predator travel speed: ordinary dry-land wandering
covers 11 units per 0.1-second tick, or 110 units/second. Even a 1000-unit direct
return is only about nine seconds of active movement. Terrain, resting and
heading can delay that, but distant release alone cannot promise a long safe
harvest window. A guide's return trip can consume much of the time it bought.

<!-- HELDOUT_RESULTS -->

## Corrected held-out results

229 local runs are retained across five controller iterations, diagnostics, null comparisons and two held-out protocols. The tables below use only the final 60 version-5 runs: 24 generated-map runs, 24 orchard runs and 12 separately labelled oracle runs.

### Default generated maps: eight seeds, 600-second horizon

| Policy | Mean score | Mean duration | Reached 600 s | Mean harvest energy | Mean captures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing nursery | 490.29 | 500.6 s | 4/8 | 25,726 | 18.00 |
| Managed colony, no banishment | 478.01 | 467.9 s | 2/8 | 14,634 | 5.25 |
| Managed colony with banishment | 489.62 | 479.2 s | 2/8 | 15,036 | 5.62 |

Banishment won 1/8 paired scores and tied 7/8. Mean score difference: +11.61; descriptive paired-bootstrap 95% interval [0.0, 34.819]. Mean duration difference: +11.3 seconds.

| Seed | Control score | Banishment score | Guide selections | Release attempts |
| --- | ---: | ---: | ---: | ---: |
| 201 | 430.91 | 430.91 | 0 | 0 |
| 202 | 605.03 | 605.03 | 0 | 0 |
| 203 | 441.58 | 441.58 | 0 | 0 |
| 204 | 455.08 | 455.08 | 0 | 0 |
| 205 | 493.19 | 586.04 | 3 | 0 |
| 206 | 615.42 | 615.42 | 0 | 0 |
| 207 | 379.51 | 379.51 | 0 | 0 |
| 208 | 403.39 | 403.39 | 0 | 0 |

Seven generated-map pairs matched exactly because the conservative gate never dispatched a guide. All measured benefit came from seed 205: three guide jobs, two controller-reported remote contact losses, and no planned rest-based release. Those contact losses are separate events; this harness does not measure their return delay. This is limited evidence for occasional displacement, not reliable acquisition on ordinary maps. The nursery still had the highest mean score and reached the horizon more often.

A separate **training-only** acquisition ablation removed the conservative gate on seeds 1–3 at 300 seconds. Seed 1 then dispatched five guides, achieved one planned release and gained 2.18 score points, but lost two guides and spent about 777 movement/turning energy. The other two pairs tied with no guide jobs. This suggests acquisition restrictions are a real bottleneck; it does not validate the aggressive variant on held-out maps. [Ablation data](../survival/results/colony_banishment/iteration5-acquisition-ablation.json).

### Managed orchards: four paired seeds per terrain, 300 seconds

Differences below are banishment minus the same colony control. Exposure is the percentage-point change in worker time spent within 100 units of any predator.

| Fixture | Score wins | Mean score difference | Harvest difference | Capture difference | Exposure change |
| --- | ---: | ---: | ---: | ---: | ---: |
| forest | 2/4 | -1.75 | -1,243 | -1.75 | -4.46 pp |
| obstacles | 4/4 | +43.11 | +4,369 | -3.75 | -2.71 pp |
| river | 1/4 | -4.79 | +311 | +4.00 | -0.23 pp |

The obstacle result has a median paired gain of 11.92 points. Its mean is strongly affected by seed 22, where the control died at 162.1 seconds and the banishment colony reached 300. Excluding the largest gain leaves a mean of 8.96 points across the other three cases. One obstacle case had only brief guiding and no formal remote release, so 4/4 is a whole-controller outcome, not proof that every gain came from a completed banishment. This is still a small arranged-fixture sample.

### Protection, cost and return

| Evaluation | Guide jobs | Planned remote release attempts | Observed returns / attempts | Median observed return delay | Mean guide time per run | Mean guide movement/turning energy per run | Mean predator-seconds targeting guide |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Generated maps | 3 | 0 | 0/0 | Not observed | 8.0 s | 54.5 | 1.5 |
| Managed orchards | 39 | 14 | 12/14 | 38.2 s | 47.1 s | 469.2 | 17.8 |

Unobserved returns are right-censored by extinction or the horizon. A release attempt and surviving guide are not proof of exclusion or permanent occupation. Return radius is 250 around the release-time actual worker centroid, so worker relocation changes its interpretation.

### Exact-pose diagnostic, kept separate

| Fixture | Oracle banishment score difference | Score wins |
| --- | ---: | ---: |
| forest | -48.19 | 1/2 |
| obstacles | -4.67 | 1/2 |
| river | -1.39 | 1/2 |

Exact localization alone did not make the strategy uniformly useful. This diagnostic does not grant geometry, hidden rest or energy to the observation-only policy.

## Reviewable artifacts

- [Frozen final protocol](../survival/results/colony_banishment/heldout-v5-freeze.json), [paired summaries](../survival/results/colony_banishment/final-summary.json), and [all-run inventory](../survival/results/colony_banishment/experiment-inventory.json).
- [Generated-map outcomes](../survival/results/colony_banishment/heldout-v5-generated.json), [orchard outcomes](../survival/results/colony_banishment/heldout-v5-fixtures.json), [oracle outcomes](../survival/results/colony_banishment/heldout-v5-oracle.json).
- [Three exact null comparisons](../survival/results/colony_banishment/iteration5-null-verification.json). The score, duration, metrics, events, releases and traces match in all three inactive pairs.
- [Generated-map control replay](../survival/results/colony_banishment/heldout-v5-generated-201-control.json.gz) and [matched banishment-enabled replay](../survival/results/colony_banishment/heldout-v5-generated-201-banish.json.gz). Seed 201 illustrates the policy when no guide is dispatched.
- [Managed forest control replay](../survival/results/colony_banishment/heldout-v5-forest-21-control.json.gz) and [banishment replay](../survival/results/colony_banishment/heldout-v5-forest-21-banish.json.gz).
- [Obstacle orchard banishment replay](../survival/results/colony_banishment/heldout-v5-obstacles-22-banish.json.gz).
- [Active generated seed-205 banishment replay](../survival/results/colony_banishment/final-active-generated-replays-generated-205-banish.json.gz) and [its control](../survival/results/colony_banishment/final-active-generated-replays-generated-205-control.json.gz). These are separate recording reruns; [reproduction check](../survival/results/colony_banishment/active-replay-reproduction.json) compares their game results with the original held-out runs.
- [Source verification](../survival/results/colony_banishment/source-verification.json), [checks](../survival/results/colony_banishment/checks.log), [replay verification](../survival/results/colony_banishment/replay-checks.json).

Load recordings through the existing Survival Lab file picker. Inspector annotations were wrapped into its existing rule/detail structure after recording; positions, actions, observations, energy and event data were retained.

## Recommendation and remaining limits

Keep this as an optional, conservative strategy experiment. The code now performs actual managed harvesting, fueled guide selection, acquisition attempts, repeated displacement, release avoidance and return. The evidence supports temporary displacement as a possible service, but not routine banishment as a general improvement or cheap permanent parking. Preserve the no-banishment colony and existing nursery as controls.

The main unresolved costs are poor acquisition geometry, guide return routes that can bring pressure back, remote food/aging costs, changing orchard productivity, unknown terrain, and additional predators arriving from an unobserved direction. Only one guide can be active. There is no dedicated return-route guard, no guaranteed opposite-corner navigation, no robust river-isolation planner and no permanent occupation. These were considered; they are not silently claimed as solved.

A further iteration should pay for a guide only when the predicted extra harvesting/evacuation time exceeds its food and replacement cost, and verify a safe return or independent food patch before departure. It needs new held-out seeds and the full 3000-second horizon. The current bounded samples do not justify replacing the baseline or making a hosted submission.
