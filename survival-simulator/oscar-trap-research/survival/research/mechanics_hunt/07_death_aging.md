# Mechanics hunt 07: death, aging, and list iteration

Tested locally on 17 September 2026 against the unmodified vendored simulator at commit `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`. No network service, competition API, submission, or remote target was used.

## Result

The known in-loop removal bug is real in ordinary generated games, but it is not a useful tactic to optimize. The stronger practical result is a defensive timing rule: an agent must finish its action with **strictly more than the next passive drain** to reach fruit collection. At the standard `dt=0.1` on every current biome, that means post-action energy must be `> 0.1`. At exactly `0.1`, passive drain reaches zero and kills the agent before it can eat fruit at its endpoint.

There is a narrow, contract-compliant exception for elderly agents. The engine checks death after passive drain, then applies old-age damage, then collects fruit. An agent that remains slightly positive at the first gate can become negative from old-age damage and still be restored by an overlapping fruit later in the same tick. This is proven in a fixture, but reaching the fruit safely is the actual strategy problem.

## Ranked findings

| Rank | Finding | Evidence and magnitude | Observability | Contract status | Recommendation |
| --- | --- | --- | --- | --- | --- |
| 1 | Action-end energy must exceed passive drain to eat | Starting at `0.100000` energy while overlapping a 20-energy fruit dies and leaves the fruit; `0.100001` survives and ends at `20.000001`. Movement and turning costs happen before this gate. | Energy and biome are ordinary status; action cost is calculable. | Compliant, one action. | Add an emergency-food feasibility check: `energy - action_cost > 0.1 * biome_energy_modifier`. The current source sets all biome energy modifiers to 1. |
| 2 | Fruit can rescue post-gate old-age negative energy | Age 100, max-age 60, energy 0.5 becomes `-0.601` yet remains alive for the response. An overlapping fresh fruit instead leaves it alive at `19.399`. A seed-42 default map transfer using its actual agent max-age and spawned 20-energy fruit ended at `19.2186`. | Age, energy, and fruit bearing/distance are ordinary. `max_age` is hidden, but onset is inferable from the previous energy delta. | Compliant, one action per agent. Both rescue checks assign timing/overlap; the seed-42 transfer retains the generated map, traits, and fruit. | Use as a safety-margin rule for old agents already heading to reachable food, not as a reason to run energy to zero. |
| 3 | Starvation removal skips the next list entry's lifecycle | The follower saves one tick of age/passive drain and gets no fresh observation or fruit handling. Its action still executes and costs energy. Four untouched default-map idle runs produced **10 skipped updates across 20 deaths**, with at least one in every seed. | A zero age delta and unchanged energy are visible in ordinary status after the fact. The internal list position is correlated with creation/ID order but not part of the formal contract. | The observed effect occurs with one action per returned agent. Deliberately sacrificing a predecessor is permitted by the action schema but has no demonstrated payoff. | Do not build around it. At most, tolerate the occasional stale observation and zero age delta. |
| 4 | A skipped non-positive agent can persist and start a one-tick death cascade | In `[dying, already-negative, live]`, the first removal skips the negative agent. It is returned alive at `-0.5`; its removal next tick skips the live follower. | Negative returned energy is ordinary status. The cause and list mechanics require source knowledge. | Compliant in the fixture. Initial negative energy is artificial. | Treat as an engine quirk, not a strategy. A controller must handle an alive status with negative energy without assuming it can be saved unless fruit was already reached. |
| 5 | Predation of post-aging negative energy reverses the score sign | The reproduced overlap case adds `0.00601` score instead of subtracting it; a young positive-energy control subtracts `0.004`. | Predator rest/energy and exact post-aging order are hidden. The overlap is dangerous. | Compliant action, but fixture-only geometry and assigned hidden state. | Reject as an optimization target. The reward is tiny and requires losing an agent in an unreliable state. |
| 6 | The extinction step still awards its normal time increment | A sole agent dying to passive drain yields `time=0.1`, `score=0.1`, and zero survivors. | Direct response fields. | Normal behavior. | Neutral accounting detail; no policy advantage. |

## Natural-map check

The script ran untouched default generated worlds for seeds 1, 2, 3, and 42. It did not move entities, alter energy/age, remove spawning, or inspect hidden state for decisions. After a bootstrap response, it issued exactly one idle action for every returned agent. All 20 initial agents died naturally between 104.2 and 122.9 simulated seconds.

| Seed | End time | Score | Skipped lifecycle updates | Inferred old-age onsets |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 122.9 | 122.9708 | 3 | 5 |
| 2 | 120.5 | 120.5200 | 3 | 5 |
| 3 | 104.2 | 104.2660 | 3 | 5 |
| 42 | 119.7 | 119.7000 | 1 | 5 |

All four runs exposed at least one zero-age-delta skip in the ordinary response. This establishes natural incidence, not a stable probability estimate: four idle seeds are small, deaths occur at different list positions, and predation removals occur after the lifecycle loop and therefore do not cause this skip.

Old-age onset was also inferable without privileged state. With no movement, the normal energy change is `-0.1` per tick; the first larger negative delta reveals that the hidden age threshold has been crossed. Fruit collection can obscure an individual delta, so a controller should infer onset over several food-free ticks rather than trust one noisy transition.

## Update order and why it matters

The relevant upstream order is:

1. All submitted actions execute (`src/utils/simulation.py`, action loop).
2. `non_agent_step` iterates the live `self.agents` list directly.
3. Age increments and passive energy is deducted.
4. `energy <= 0` removes the agent immediately.
5. Old-age energy is deducted, with no second death check.
6. Observation is refreshed and overlapping fruit is collected.
7. Predators update and can kill agents.
8. Time and base score increment even if the final agent died.

Removing an agent at step 4 mutates the list being iterated. Python then advances its index, skipping the object shifted into the removed slot. Predator kills at step 7 do not have this effect because the agent lifecycle loop is already complete.

## Controls and rejected hypotheses

- A skipped follower's walk still moved it 10 units and charged 0.5 energy. The bug does **not** cancel its submitted action; it only skips lifecycle, observation refresh, and fruit handling.
- Predator removal of the first agent did not skip the second agent's lifecycle. The follower advanced to age 0.1 and energy 149.9 before predation.
- A fruit cannot save an agent whose passive drain reaches exactly zero, because the death gate is strict `<= 0` and precedes collection.
- The final-death tick is not omitted from time or base score.
- Dead-agent observation entries remain in the internal cache. They are not returned for dead IDs and offer no controller advantage; this is only an internal stale-cache/memory issue.

## Practical controller implications

The useful implementation rule is to reserve the full action cost plus the next passive tick before committing to fruit. For a young agent on the current upstream biomes:

`required pre-action energy > movement_cost + turning_cost + 0.1`

For an old agent, use a larger observed drain estimate when judging whether the destination fruit will be reached, even though the engine technically permits post-gate rescue. Do not assume an agent with negative returned energy has already been removed: the old-age ordering and list-skip bug can both return such an agent for one tick.

No tested finding justifies intentional death. The list skip saves only one lifecycle tick per affected removal and can be harmful because it also prevents fruit collection and leaves observations stale. The negative-energy predation bonus is three orders of magnitude smaller than the survival score from one second and requires losing an agent.

## Reproduction

From the repository root:

```sh
survival/.venv/bin/python survival/research/mechanics_hunt/07_death_aging.py
```

The command asserts every controlled result and writes [07_death_aging.json](../../results/mechanics_hunt/07_death_aging.json). To repeat only a shorter generated-map subset:

```sh
survival/.venv/bin/python survival/research/mechanics_hunt/07_death_aging.py --seeds 1 --horizon 140 --output /tmp/07-death-aging-seed1.json
```

## Evidence boundaries

- **Proven with ordinary generated worlds:** in-loop death skips naturally appear and are detectable from returned age/energy; old-age onset is inferable from energy deltas; no duplicate IDs or extra actions are needed.
- **Proven only in controlled fixtures:** exact passive threshold at a colocated fruit, post-old-age fruit rescue, a pre-existing negative-energy cascade, and score-positive predation. The rescue was also transferred to an otherwise default seed-42 map with its real spawned fruit and actual max-age, but energy, age, and overlap were assigned.
- **Privileged diagnostic inputs:** assigned max-age/age/energy, exact overlap, entity handles, and internal cache inspection. None were used to choose actions in the generated-map runs.
- **Not claimed:** hosted behavior, a broad incidence probability, a score-improving death tactic, or exhaustive proof over all death combinations.
