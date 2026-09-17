# Investigator 05: fruit and tree timing

Tested locally on 17 September 2026 against the unmodified vendored simulator at commit `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`. No network, validation attempt, evaluation, submission, or vendor edit was used.

Run:

```sh
survival/.venv/bin/python survival/research/mechanics_hunt/05_fruit_tree.py --trials 300 --seeds 1 7 42 --horizon 300
```

Machine-readable evidence is in `survival/results/mechanics_hunt/05_fruit_tree.json`.

## Ranked findings

### 1. Dense orchards are replenished aggressively in the early game

This is the strongest ordinary-play lead in this track, but it is not yet an end-to-end score advantage.

`spawn_tree` is attempted with raw per-tick probability
`100 / max(1, tree_count / 2) * dt * 0.5 ** (time / 300)` before the biome acceptance check. At `dt=0.1`, 50 trees at time zero cause an attempt with probability 0.4 every tick. With 10 or fewer trees, the raw probability is at least 1, so there is an attempt every tick. Source: `environment.py:447-468,737-753`.

On generated 1600 by 1200 maps with normal obstacles, biomes, tree births, tree deaths, fruit growth/rot and predator spawning:

| Seed | Trees created from 50 initial attempts | Peak sampled trees at 60 s | Trees at 300 s | Tree births/deaths over 300 s | Fruit births over 300 s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 103 | 65 | 393 / 360 | 1,167 |
| 7 | 38 | 105 | 73 | 422 / 387 | 1,214 |
| 42 | 19 | 97 | 52 | 351 / 318 | 867 |

The survey deliberately removed agents, so the observed 105–182 fruits still present at 300 seconds are an upper bound with no consumption. Tree and fruit creation counts are engine diagnostics, not observable policy inputs.

The actionable, contract-compliant hypothesis is to prefer locally dense tree observations, revisit productive clusters, and keep exploring as old clusters turn over. Trees may overlap because tree placement checks obstacles but not other trees. In these three maps, the densest sampled 50-unit tree neighbourhood held 3–5 trees. At 300 seconds the densest 15-unit fruit neighbourhood held 3–4 fruits. A controller can see repeated `Tree` and `Fruit` observation entries, although neither type has an ID or age.

This extends the earlier “small orchard colony” result. It does not prove that camping one orchard beats continued exploration, because the survey had no consumers or predators affecting agents.

### 2. Maturation is worth exactly 40 energy only when birth time and reserve are credible

The known timing result reproduces: a fruit grows from 20 to about 60 energy over 20 simulated seconds and is removed at about 50 seconds. Fresh and ripe fruit at the same location produce the same observation payload: only type, distance and angle. Source: `fruit.py:4-25`, `creature.py:94-179`, `environment.py:730-735`.

The extended realistic-energy fixture compares both choices over the same 20.1 seconds, from 150 energy and 20 units away:

| Case | End energy | Direct fruit bonus | Outcome |
| --- | ---: | ---: | --- |
| Known-fresh, eat now | 144.4 | 0.02 | Survives |
| Known-fresh, wait 20 s | 184.4 | 0.06 | Survives, +40 energy and +0.04 score |
| Already 40 s old, eat now | 184.4 | 0.06 | Survives |
| Already 40 s old, blindly wait 20 s | 124.4 | 0 | Fruit rots |

A separate 18-energy control survived by walking to the fresh fruit immediately but died after 18.1 seconds when waiting. The practical rule is therefore gated waiting: only delay a fruit first seen in a continuously covered patch, only when the agent has a safe energy reserve, and stop at maturity. First sighting without prior coverage does not reveal birth time.

The controller actions used one request per observed living agent. Fruit age was used only to configure and diagnose the fixtures. The proposed gate needs only first-seen history and the exposed energy value.

### 3. Fruit and tree observations are stale after same-tick removal

Agent observations are computed before fruit collection, fruit rot and tree death. The returned state can therefore contain:

- a fruit that the same agent has already consumed in that tick;
- a tree removed by its death check later in that tick.

Both disappear from the next response. This is an ordinary-observation, contract-compliant timing fact. A controller that blindly chases the returned fruit can waste one movement action and its energy. Maintain a short claimed/consumed set based on expected endpoint contact and energy increase, and require a second sighting before committing further travel when contact should already have occurred. Trees deserve a similar one-response grace period before treating disappearance as movement error.

The stale observation is not a hidden-state exploit: hidden list membership was used only to prove that removal had occurred.

### 4. One endpoint can collect a fruit cluster automatically

Collection iterates through every local fruit whose centre is within the strict sum of radii. A synthetic five-ripe-fruit stack was consumed in one tick by one idle, full-energy agent and added 0.30 score even though none of the energy could be stored. The returned observation still contained all five stale fruit entries.

Exact co-location is an artificial fixture, but the generated no-consumer maps did contain natural clusters of 3–4 fruits within 15 units of one fruit. A practical controller can favour a multi-fruit endpoint when several observations have nearly equal relative coordinates. This is a modest route-efficiency opportunity, not evidence for manufacturing stacks.

### 5. Tree death is a repeated hazard, not a fixed 50–100 second lifetime

Every tree tick draws a new threshold `50 + 50 * sqrt(random())` and removes the tree when its current age exceeds that draw. A tree does not receive one permanent death age. Source: `environment.py:745-753`.

Across 300 one-tree engine fixtures at the normal local `dt=0.1`, new-tree death age was:

- median 58.1 seconds;
- 10th–90th percentile 54.3–62.1 seconds;
- range 51.7–66.1 seconds.

A new forest tree produced a median 4 fruits before death, with a 10th–90th percentile of 1–6. Unrelated global tree/predator creation was disabled, and agents and consumption were absent.

The death hazard is also step-size-sensitive because the random check is not scaled by `dt`: median death was 56.5 seconds at `dt=0.05`, 58.1 at 0.1, and 60.0 at 0.2. The documented/local server uses 0.1, so this is mainly a simulator correctness finding. It supports revisiting clusters rather than assuming a tree remains productive for a fixed hidden lifetime.

All initial trees that were successfully created were pre-grown to ages between roughly 20 and 80, so they are immediately productive but some are already exposed to a high death hazard. Also, 50 initial spawn calls created only 19–38 trees in the three tested seeds because biome acceptance can reject an attempt.

## Fixture-only effects and rejected leads

- **Synchronized rot list skip:** eight fruits all set just past the rot threshold decayed as 8, 4, 2, 1, 0 across four ticks because the loop removes from the list while iterating. The last gained only 0.3 seconds. Natural fruit birth times are not synchronized enough to make this a useful tactic.
- **Blocked tree creation:** `spawn_tree(..., max_attempts=0)` can create a tree inside an obstacle because its final obstacle-free result is not checked, while fruit creation at the same blocked point fails. Normal tree calls get 50 attempts, so this is a synthetic negative control, not a strategy.
- **Full-energy scoring:** ripe fruit still adds about 0.06 score when energy storage is full. This reproduces the known fact but is tiny next to 1 score per survival second until survival is reliable.
- **Blind 20-second waiting:** rejected. It can forfeit a fully ripe fruit or kill a low-energy agent.
- **Waiting past maturity:** rejected. Energy and radius stop increasing once energy reaches about 60.
- **Treating trees as permanent orchards:** rejected. Individual trees turn over quickly even though the global system replenishes them.

## Evidence boundaries

All movement, collection, growth, scoring, rot, tree-death and generated-map claims call the original engine methods. Arranged positions, pre-aged fruit/tree objects, disabled unrelated spawning, absence of consumers, object identity, and exact hidden ages are marked above. The generated-map policy-relevant counts use seeds 1, 7 and 42, but no controller was benchmarked and no hosted behavior was tested.

The best next test is an observation-only controller that timestamps genuinely new fruit in continuously covered orchard patches, chooses cluster endpoints, ignores one-response post-contact ghosts, and falls back to immediate eating under an energy or predator-risk threshold. Compare it with the current nursery policy across the existing ten-seed benchmark, without using fruit IDs, ages, absolute map coordinates or engine handles.
