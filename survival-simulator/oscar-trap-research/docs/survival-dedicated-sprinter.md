# Dedicated single-predator bait: executable next iteration

17 September 2026. Personal Nordic AI Cup research. **The useful new strategy is a cheap hold inside the predator's turning circle, with controlled food excursions. Ordinary founders can perform it; forced speed breeding is not a prerequisite.** The controller and local experiments are complete. This is an opt-in research policy, not a demonstrated solution to the full game.

The representative final founder starts with **150 energy**, holds the selected predator continuously for **114.7 seconds**, consumes **13 fruits**, and spends **218.80 movement energy**. It eventually starves. The old adaptive controller lasts 9.5 seconds in the same orchard. With no food the new controller holds for 68.7 seconds, versus 2.2 seconds for the old controller. These comparisons use original physics, cached observations, physical boundaries and finite resources.

## Scope and evidence

All work stayed in the assigned research/results directories and this report. Original source is `survival/vendor/survival-simulator`, commit `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`; interpreter is `survival/.venv/bin/python`. No competition API, hosted validation, submission, publishing, pushes, vendor changes, worker-policy edits, debugger changes or new renderer. Water baiting, wall acquisition and two-bait handoff are outside this experiment.

The final evidence comprises **85 primary engine runs**: 9 matched controls, 36 held-out cases, 16 trait/energy cases, 18 fixed-horizon worker-protection runs and 6 generated-map games. Five thousand optimistic mutation searches and seven final representative replays supplement them. Earlier controller screens and failed variants are preserved separately.

**Geometry construction is privileged; action decisions are observation-only.** Controlled arenas are 1600×1200 forest with 30-unit boundary walls. The default bait is at (800,600), predator 60 units west, facing east with 102 energy; bait faces west. Agent senescence threshold is 90 seconds unless explicitly varied. Initial predator wake state is arranged, not inferred from gameplay. No new predators or trees are added in these fixtures. Original fruit growth, aging, tree death, collision, hearing, predator control and energy rules run unchanged.

Food fixtures: `line` contains four mature fruits east of the bait at offsets 140/300/460/620. `orchard` contains eight ordinary trees aged 20 seconds on a radius-100 ring plus six mature fruits on a radius-65 ring. The trees produce fruit using the engine's stochastic rules and then die. These are generous **finite arranged orchards**, not free ongoing nutrition or evidence that a randomly spawned bait can acquire such a patch. Final generated-map tests use the unmodified default map generator and all normal spawns.

## What the controller does

[`controller.py`](../survival/research/dedicated_sprinter/controller.py) takes one ordinary agent DTO and simulation time. It never receives true pose, predator identity, hidden predator energy/rest, fruit age, tree age, a true terrain map or assigned world coordinates.

1. Reconstructs a local frame from self motion, current biome and remembered static edges/trees. Fruit positions are remembered for interception but excluded from pose correction because fruit has no identity and can disappear. Facing and movement directions remain independent.
2. Infers predator motion from consecutive cached sightings and rest after two nearly stationary sightings. Advances the delayed predator pose before evaluating the next action. It also considers a full-speed waking predator instead of treating inferred rest as a guarantee.
3. Scores finite single-action candidates for energy, safety, hearing contact, observed obstacles, known slow terrain and progress toward remembered fruit. Requested movement is clamped to the observed sprint trait and current cutoff. No repeated-action trick is used.
4. In a cheap hold, targets the interior of the predator's turning circle. A predator advancing 15 while turning 0.3 radians follows a discrete circle of radius `15 / (2 sin(0.15)) ≈ 50.2`. A bait that safely gets inside that circle can stay selected within hearing range while barely moving. The controller recentres as observed speed changes.
5. Switches to food-seeking curved pursuit when fruit is remembered and fuel is below `max(150, 0.7 × capacity)`. It maintains a cost for leaving 58-unit predicted hearing range and a 24-unit predicted capture margin. These are soft objectives, not a proof of safety. Resting windows create opportunities to collect food.

Energy balance and nutrition caps were separately audited for 67 final controlled/replay records, and corrected terminal aging accounting was checked against a fresh run. The prediction function matched 200 original clear-ground predator actions, across headings, ranges and both normal speed and the low-energy cutoff, to floating-point error below 1.2e-13. A final correction preserved heading while modelling rest and used the source's requested-speed pivot geometry even at low actual speed. Safety still depends on imperfect observation association and terrain/collision inference.

On final held-out cases, inferred-rest precision was 94.0% and recall 90.8%. The rest classifier can confuse a blocked predator with sleep. No hidden state corrects policy decisions; hidden fields are used only for diagnostic metrics.

## Matched training controls

All rows below start with 150 energy, walking 10, sprinting 20 and capacity 500. `adaptive` is the prior direct escape policy. `nofood` uses the new controller but disables deliberate food pursuit; incidental eating is still possible. `predictive` is the final automatic hold/forage controller.

| Food | Policy | Lifetime s | Active target share | Longest occupied s | Fruits eaten | Movement energy |
| --- | --- | --- | --- | --- | --- | --- |
| none | adaptive | 2.2 | 100.0% | 2.2 | 0 | 50.8 |
| none | nofood | 68.7 | 100.0% | 68.7 | 0 | 26.9 |
| none | predictive | 68.7 | 100.0% | 68.7 | 0 | 26.9 |
| line | adaptive | 20.4 | 100.0% | 20.4 | 4 | 284.9 |
| line | nofood | 68.7 | 100.0% | 68.7 | 0 | 26.9 |
| line | predictive | 14.5 | 44.1% | 3.3 | 0 | 83.7 |
| orchard | adaptive | 9.5 | 100.0% | 9.5 | 2 | 141.4 |
| orchard | nofood | 68.7 | 100.0% | 68.7 | 0 | 26.9 |
| orchard | predictive | 114.7 | 100.0% | 114.7 | 13 | 218.8 |


“Active target share” counts real predator selections before its original movement, including ticks when it wakes. “Longest occupied” is the longest uninterrupted interval without an active selection away from the bait, including intervening rest time. Death ends the interval. Survival without selection does not count as occupation. No-food and line results expose why lifetime alone is inadequate: food detours can lose attention even while the bait survives.

The representative orchard founder receives 522.2 nutrition after capacity clipping, from 522.2 available nutrition in eaten fruits. Movement costs 218.8, turning 87.4, and passive/aging drain 366.5. Unconsumed fruit and nutrition wasted at capacity are not credited as fuel.

## Untuned geometry, heading, fuel and terrain

The final split was drawn with fixed generator seed 403971 and engine seeds 401–412. It varies position, rotation, heading ±2 radians, separation 50–85, predator energy 40–200, bait energy 150/250/400, senescence 60–120, obstacles, desert and forest/swamp boundaries. Parameters are stored per run. They were not used for parameter selection. Each of the three controllers receives the same construction parameters; controller-dependent fruit interactions can alter subsequent shared RNG consumption.

| Policy | Median lifetime s | Mean active target share | Median longest occupied s | Runs with ≥60 s occupation /12 | Median fruits |
| --- | --- | --- | --- | --- | --- |
| adaptive | 72.0 | 43.7% | 4.4 | 0 | 0.0 |
| nofood | 102.4 | 99.9% | 102.3 | 12 | 0.0 |
| predictive | 112.8 | 99.6% | 84.0 | 12 | 7.5 |


These tests validate a held encounter and some local acquisition, not a guaranteed search for predators or orchards. One split-terrain case actually entered swamp; the other two stayed on the forest side. Seed 407 was captured at 64.5 seconds with 450.6 energy still available, so fuel reserve alone does not make the controller safe. The largest measured odometry error was 0.0 units. Only the current and previously visited biomes are known; an unseen terrain transition remains dangerous. Observation payloads lack predator IDs, so nearest-neighbour motion inference can switch individuals when several predators approach.

## Worker protection at a common horizon

Six new geometries/headings, each with two stationary 150-energy workers, were run for the same **90 seconds**, including after a bait died. Worker starvation and capture are separated; no worker starved in this set. This avoids rewarding a failed bait merely because its experiment stopped early.

| Policy | Workers alive /12 | Worker captures /12 | Worker-seconds /1080 |
| --- | --- | --- | --- |
| none | 2 | 10 | 553.1 |
| adaptive | 3 | 9 | 681.3 |
| predictive | 12 | 0 | 1080.0 |


This is a controlled protection result, not a forage-colony score. Workers are arranged away from the initial chase. A worker entering the predator's path can become the closest target; the bait has no privileged control over target selection.

## Small specialist caste and preparation cost

[`colony.py`](../survival/research/dedicated_sprinter/colony.py) keeps the existing nursery worker policy and limits bait roles to at most two and at most one-third of the living colony. It attempts one bait per predator observation cluster using relative Agent links to align local frames. Disconnected observation components can still duplicate a predator. Roles expire on death or lost contact; there is no coordinated handoff. Readiness requires observed sprint speed ≥18 and fuel at least 70 above cutoff. Selecting the best nearby prepared agent is supported; a dedicated breeding programme is not justified by these runs.

| Walk | Capacity | Initial energy | Food | Lifetime s | Active target share | Fruits |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 500 | 75 | none | 0.4 | 100.0% | 0 |
| 10 | 500 | 75 | orchard | 0.4 | 100.0% | 0 |
| 10 | 500 | 150 | none | 68.7 | 100.0% | 0 |
| 10 | 500 | 150 | orchard | 114.7 | 100.0% | 13 |
| 10 | 300 | 75 | none | 29.9 | 100.0% | 0 |
| 10 | 300 | 75 | orchard | 104.4 | 100.0% | 9 |
| 10 | 300 | 150 | none | 68.7 | 100.0% | 0 |
| 10 | 300 | 150 | orchard | 109.6 | 100.0% | 8 |
| 12 | 300 | 75 | none | 31.9 | 100.0% | 0 |
| 12 | 300 | 75 | orchard | 101.2 | 100.0% | 9 |
| 12 | 300 | 150 | none | 70.5 | 100.0% | 0 |
| 12 | 300 | 150 | orchard | 100.8 | 98.2% | 6 |
| 16 | 500 | 75 | none | 36.9 | 100.0% | 0 |
| 16 | 500 | 75 | orchard | 115.9 | 100.0% | 10 |
| 16 | 500 | 150 | none | 72.2 | 100.0% | 0 |
| 16 | 500 | 150 | orchard | 115.2 | 100.0% | 9 |


A normal newborn begins at 75 while capacity 500 implies a 100-energy sprint cutoff. The final newborn replay dies after 0.4 seconds despite nearby orchard resources. Lower capacity 300 gives a 60 cutoff and only 15 energy of initial sprint reserve. It does not guarantee safe deployment. Cheap walking speed can help, but selection for life alone is unsafe: the replay using an actual sampled mutant (walk 13.934, sprint 20, capacity 309.452) lives 112.1 seconds yet keeps attention for only 43.0% of active ticks.

The original `Environment.spawn_agent` was used for 1000 searches per target. The optimistic nursery has one best parent, unlimited replacement food, no elapsed game time, no senescence and no predators. Failed children are removed. Therefore these are **optimistic mutation-effort estimates**, not feasible preparation demonstrations.

| Mutation target | Births p10 | Births median | Births p90 | Median parent birth debits |
| --- | --- | --- | --- | --- |
| capacity375 | 4 | 26.0 | 77 | 2600.0 |
| walk12 | 4 | 24.0 | 68 | 2400.0 |
| walk12_capacity375 | 14 | 44.0 | 93 | 4400.0 |
| walk15_1 | 15 | 48.0 | 109 | 4800.0 |
| walk12_capacity250 | 29 | 66.0 | 129 | 6600.0 |


Each birth debits its parent 100 energy and endows the child with 75. The debit column is cumulative parent spending, not net energy destroyed or an exact external food requirement. Useful failed offspring, replacement-parent energy, collection costs, time and mortality need a real nursery simulation. The search conserves neither time nor food, and cannot establish that 44 births can be produced before an encounter.

A founder starts at 150. Preparing a 500-energy founder needs at least 350 added nutrition, at least six fully mature 60-energy fruits before movement/living costs. Raising a newborn from 75 to 150 needs 75 added nutrition, at least two mature fruits. Filling larger capacity is useful only if that food is actually available. The final controller's headline arranged result uses the ordinary founder and needs no mutation preparation.

The 114.7-second window is not permanent occupation. Even if it could be repeated with perfect handoffs, a single predator over 3000 seconds would require at least 27 bait lifetimes, hence 26 replacement births and 2600 parent birth debits, plus food and safe staging. No such replacement chain was executed. Multiple predators and aging increase the practical burden.

## Generated-map games

Three paired default generated maps ran to extinction or the normal 3000-second horizon. The baseline is the unchanged nursery policy; the candidate adds the bounded specialist assignment layer and final single-bait controller. Births and fuel are obtained through ordinary gameplay; no prepared agents, trait injection, arranged food or hidden geometry are supplied. There is no forced specialist breeding in these games.

| Seed | Nursery score | Caste score | Difference | Nursery survival s | Caste survival s | Caste births | Distinct assigned baits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1014.6 | 849.4 | -165.2 | 1023.7 | 858.0 | 56 | 21 |
| 2 | 506.0 | 898.8 | 392.8 | 496.7 | 899.4 | 77 | 16 |
| 3 | 858.9 | 856.6 | -2.3 | 857.5 | 867.7 | 73 | 24 |


Mean score: nursery 793.2, caste 868.3. The candidate wins score on **one of three seeds**, and none of the six games completes 3000 seconds. This sample does not establish a reliable score improvement. Assignment episodes include reacquisition of the same agent, so they must not be read as replacement births. Logs preserve assignment time, energy, age and traits. An earlier unbounded assignment prototype reached four baits and performed badly on seed 1; it is preserved in `fullgame-caste-1-v3.json`, excluded from the final comparison.

The original engine shares its RNG between ecology, movement and reproduction, so policy changes diverge future maps/events. These are paired starting seeds, not matched future random events. Local macOS execution also does not prove equivalence with hosted Linux scoring.

## Failed hypotheses and remaining limits

- Straight adaptive sprinting wastes fuel and cannot deliberately harvest a patch. It remains a useful paired control.
- Curved pursuit alone improved occupation but spent hundreds of movement energy. The turning-circle hold reduced that cost substantially.
- Weak food attraction plus quiet facing often increased lifetime by releasing the predator. Strong food attraction could similarly destroy retention. Those screens remain in `center-food-screen.json`; the final controller prioritizes contact and retains normal facing.
- Fruit-based pose correction produced spurious shifts when unlabelled food changed. Final localization uses static trees/edges only. Unknown collisions and ambiguous landmarks remain possible.
- Moving less is insufficient if aging dominates. Finite trees die and fruit rots. The representative founder eventually starves despite successful holding and collection.
- Fast or low-capacity mutants are not automatically better baits. The actual-mutant replay demonstrates high survival with poor occupation. Prefer readiness and measured retention over a single trait threshold.
- The assignment layer cannot yet guarantee one globally unique predator per bait under disconnected observations. Contact churn, distractors, several simultaneous predators and replacement staging remain unsolved.

The concrete deliverable is the tested observation-only controller, an optional small-caste wrapper, reproducible scenarios, diagnostics and replays. The next justified deployment gate is reliable assignment/acquisition and replacement under generated-map observations; the holding controller itself has substantially stronger evidence than the earlier 39.1-second, 500-energy/four-fruit demonstration.

## Artifacts and reproduction

Final controller SHA-256: `7b24bb76439679778a46acc0fa609be0c4d6648f868e48c1117e74e3c52e3afe`.

Primary final records are `controls-final-v5.json`, `validation-final-v5.json`, `traits-final-v5.json`, `protection-v5.json`, `fullgame-caste-{1,2,3}-v5.json`, the three `fullgame-nursery-*.json` controls, `genetics.json`, `replays-final-v5.json`, `summary.json` and `verification.json`, all under `survival/results/dedicated_sprinter/`. Earlier v1–v4 screens are development evidence and are not pooled into final statistics. Source snapshots are retained as `controller_v1.py` through `controller_v4.py`.

From the repository root:

```sh
DEDICATED_RUN_TAG=v5 survival/.venv/bin/python survival/research/dedicated_sprinter/controls.py
DEDICATED_RUN_TAG=v5 survival/.venv/bin/python survival/research/dedicated_sprinter/validation.py
DEDICATED_RUN_TAG=v5 survival/.venv/bin/python survival/research/dedicated_sprinter/protection.py
DEDICATED_RUN_TAG=v5 survival/.venv/bin/python survival/research/dedicated_sprinter/fullgame.py --mode caste --seed 1
survival/.venv/bin/python survival/research/dedicated_sprinter/fullgame.py --mode nursery --seed 1
survival/.venv/bin/python survival/research/dedicated_sprinter/genetics.py
DEDICATED_RUN_TAG=v5 survival/.venv/bin/python survival/research/dedicated_sprinter/replays.py
survival/.venv/bin/python survival/research/dedicated_sprinter/verify.py
survival/.venv/bin/python survival/research/dedicated_sprinter/report.py
```

Repeat the two full-game commands for seeds 2 and 3. All commands run locally. `DEDICATED_CONTROLLER=controller_v2` selects an archived controller for development comparisons.

Load these files with the existing Survival Lab viewer's file loader:

- [`final-founder-orchard.json.gz`](../survival/results/dedicated_sprinter/final-founder-orchard.json.gz): 150-energy founder, actual food excursions and eventual starvation.
- [`final-founder-no-food.json.gz`](../survival/results/dedicated_sprinter/final-founder-no-food.json.gz): cheap holding without nutrition.
- [`final-protected-workers.json.gz`](../survival/results/dedicated_sprinter/final-protected-workers.json.gz): fixed-horizon protection of two workers.
- [`final-specialist-orchard.json.gz`](../survival/results/dedicated_sprinter/final-specialist-orchard.json.gz): synthetic walk-12/capacity-300 specialist and its failure.
- [`final-newborn-cutoff.json.gz`](../survival/results/dedicated_sprinter/final-newborn-cutoff.json.gz): normal newborn fails before feeding.
- [`final-real-mutant.json.gz`](../survival/results/dedicated_sprinter/final-real-mutant.json.gz): feasible sampled traits, survival exceeding occupation.

- [`final-heldout-capture.json.gz`](../survival/results/dedicated_sprinter/final-heldout-capture.json.gz): untuned obstacle/heading case that loses a well-fuelled bait.

Replays use the existing `survival/debugger/recorder.py` format, record cached action observations and diagnostic truth separately, and do not alter the viewer or manifest.
