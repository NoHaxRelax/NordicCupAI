# Speed selection, fruit timing, and young orchard colonies

Source inspected: public `amboltio/Nordic-AI-Cup-2026`, commit `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`, downloaded 2026-09-17. Evidence below is source analysis and controlled mechanics tests. Genetic results are an optimistic mutation model, **not** full-game survival results. Remote evaluation code equivalence is unverified.

## Speed breeding is possible, but a slow investment

- Founders walk 10 and sprint 20; spawned predators walk 11 and sprint 15. Movement is a distance **per tick**, with 10 ticks per simulated second. Source: `agent.py:14`, `environment.py:478`.
- Each offspring independently mutates each trait with probability 10%; the multiplier is uniform from 0.5 to 1.5. Walking is capped at 20; sprinting at 40. Improving walking beyond 15 from a founder needs at least two positive walking mutations. Source: `environment.py:297-341`.
- Select on **`min(speed, sprint_speed)`**. Sprint speed is always an initial hard movement cap, even if mutation has made the walking trait larger. Source: `environment.py:508-517`.
- `min(speed,sprint_speed) > 15` can outwalk a predator's full sprint on matching unobstructed terrain. It cannot guarantee escape through obstacles, from multiple predators, or when the agent is on slower terrain.
- All founders are identical in walking speed. A rule requiring an already superior agent to reproduce prevents the first mutation from ever appearing. Maintain a population floor and apply soft preference after useful variation appears.

The script models 10,000 independent breeding searches, always retaining the current best parent. It grants unlimited food, an immortal parent, and unrestricted births, and ignores all other traits. These assumptions make its birth budget optimistic relative to actual play.

| Cheap movement target | Median births | 10th–90th percentile | Chance within 20 births | Chance within 40 births |
| --- | ---: | ---: | ---: | ---: |
| >15 | 45 | 15–97 | 17.3% | 44.9% |
| >16 | 50 | 17–106 | 14.1% | 39.0% |
| 20 cap | 71 | 28–137 | 4.8% | 20.1% |

Each birth costs the parent 100 energy and creates a child with 75, so 45 births require 4,500 energy paid by parents, with 1,125 energy lost across the species before movement/living costs. Supporting rejected offspring is additional expense. This is a useful optional improvement to a working foraging policy, not a safe opening plan that starves every slow founder.

Birth safety matters: a default child starts below the default sprint threshold of 100 energy (20% of max energy 500). A larger max-energy mutation raises that threshold. Children with max energy 1,000 need 200 energy to sprint. Spawn in a food patch, away from predators, with enough parental reserve after the 100-energy charge. Source: `agent.py:16-17`, `environment.py:512`, `environment.py:621-623`.

## Walking is cheaper, not free

Walking costs 0.05 per requested unit; the part above walking speed costs 0.5 per unit. Energy is charged before terrain reduces displacement. All listed biomes retain base passive drain 1 energy per simulated second. Source: `environment.py:500-533`; `biome.py:19`.

At ten decisions per second, without turns or aging:

| Action on normal terrain | Energy / simulated second |
| --- | ---: |
| Remain still | 1 |
| Founder walks 10 every tick | 6 |
| Founder requests 15.1 every tick | 31.5 |
| Founder sprints 20 every tick | 56 |
| Selected walk-16 agent requests 15.1 | 8.55 |

The last row gives the real value of breeding: the same slightly-faster-than-predator movement becomes about 3.7 times cheaper. Even then, continuous kiting needs roughly 8.55 mature fruits per minute before turns, reproduction, or aging. Move only as much as separation requires; do not sprint constantly merely because sprint is available.

## Fruit should sometimes be allowed to mature, but age is hidden

Fruit starts at 20 energy and gains 2 energy per simulated second until roughly 60, so a known newborn fruit takes about 20 seconds to reach full value. Energy remains at that value afterward; its visual color changes but does not reduce nutrition. The fruit's internal age also increments at 2 per simulated second and removal checks `age > 100`, so rot occurs at about **50 simulated seconds**, not 100. Source: `fruit.py:10-25`; `environment.py:730-735`.

An actual observation includes only type, distance, and angle for fruit. It omits ID, energy, radius, age, and color. The controlled probe compares fresh and ripe fruit at the same location and verifies identical observation payloads. Source: `creature.py:134-164`.

A waiting rule therefore needs a tracked fruit known to have appeared in a continuously observed patch. First seeing a fruit does **not** establish its birth time. Waiting twenty seconds for an old fruit may let it rot. Prefer an orchard circuit that returns to recently spawned fruit, with immediate collection when energy is low or a predator approaches. There is no advantage to waiting beyond maturity. This is a proposed policy extension, not a tested scoring improvement.

Eating occurs automatically when distance is less than agent size plus fruit radius. Normal fruit starts with radius 5 and grows to about 9, so waiting at distance 10 can accidentally collect it before maturity; stay outside about 14 plus a margin. Source: `environment.py:405`, `environment.py:669`; `fruit.py:21`.

Forest and grassland trees attempt 0.1 fruits/second, swamp 0.08, desert 0.05; these are **per mature tree in the implementation**, despite the biome comment describing an area rate. Trees start producing at age 20. Their age-dependent death check begins after age 50, and new-tree spawn attempts halve every 300 simulated seconds. Thus a safe food patch needs rediscovery over a long run. Source: `biome.py:22-51`; `environment.py:738-753`.

## Third simple strategy: a small colony that renews its foragers

Use a few dispersed young foragers near fruit/tree clusters, preserve a reserve, and replace aging members. Keep faster descendants as preferred breeders, but keep ordinary members alive and useful until food supply or age makes them costly. Food scarcity is a reason to cap population, not a reason to refuse all food to slow agents before a replacement exists.

The main score grows by one per simulated second while the species survives, independently of population size. A fully ripe fruit adds only 0.06 score. Population is therefore survival redundancy and collection capacity, not a direct multiplier on the main survival reward. Source: `environment.py:670-671`, `environment.py:756-757`.

Each agent has a hidden senescence threshold uniformly between 60 and 120 simulated seconds. Afterward the source charges `0.01 * age` **per tick**, in addition to passive drain. At age 100 this means 11 energy/second standing still, compared with 1 for a young agent. At age 120 the cost is 13/second, so a mature fruit covers only about 4.6 seconds of idling. Source: `agent.py:27`; `environment.py:637-647`.

An old-to-young replacement has a 25-energy net birth loss across parent and child. At age 100 the 10-energy/second aging surcharge consumes that in 2.5 seconds. However, the old parent remains alive after birth, the child cannot immediately sprint at default traits, and the threshold itself is hidden. Practical policy: begin securing offspring before the age window, infer extra drain from successive energy observations when no fruit was eaten, and stop prioritizing expensive elders once replacements are established. Do not describe a birth as automatically deleting the old parent or transferring its full energy.

This approach is simpler to establish than specialized predator trapping because it exploits known energy economics and needs only reported age, traits, energy, and local observations. It still needs full-seed benchmarking with the actual food and predator dynamics.

## Reproduce the measurements

```sh
survival/.venv/bin/python survival/research/speed_and_fruit.py
```

Results are written to `survival/results/speed_and_fruit.json`. The script uses upstream `Fruit.grow`, observation, and movement methods for its mechanics probes; it does not modify vendor files or run remote submissions.
