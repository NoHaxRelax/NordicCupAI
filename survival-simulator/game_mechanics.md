# Survival simulator: agent vs. predator mechanics

These notes come from reading the simulator source (`src/elements/`). All movement values are per tick, and one tick is 0.1 s.

## Stats

| Stat | Agent (start) | Agent max via mutation | Predator |
|---|---|---|---|
| Size (radius) | 5 | doesn't mutate | **10** |
| Walk `speed` | 10 | 20 | **11** |
| `sprint_speed` | **20** | 40 | 15 |
| Starting energy | 150 for the 5 starting agents, 75 for spawned agents | doesn't mutate | 0 (spawns asleep) |
| `max_energy` | **500** | 1000 | 200 |
| Hearing radius (all around) | 50 | 100 | **60** |
| Vision range | 200 | 400 | **250** |
| Vision cone (full width) | π/3 (60°) | π/2 (90°) | π/3 (60°) |
| Walls block vision | yes | game rule | yes |

- **Mutation:** only `speed`, `sprint_speed`, `max_energy`, hearing radius, vision range and vision cone can mutate. When an agent spawns, each of these traits has a 10% chance to change by up to ±50%. A mutation can make a trait worse as well as better, and the max column is the cap.
- **Spawning:** the parent must have more than 100 energy and pays 100. None of that goes to the new agent, which always starts with 75.
- **Touch range:** an agent is eaten when the distance to a predator is below 5 + 10 = 15.
- **Chase range:** a predator only considers agents in its own and the neighbouring map chunks (400×400 each), which barely limits it with 250 vision.

## Energy

| | Agent | Predator |
|---|---|---|
| Walking cost | 0.05 per unit moved | same |
| Sprinting cost | 0.05 × `speed` + 0.5 per unit above `speed` | same |
| At < 20% of `max_energy` | can only walk (below 100 energy at default max) | can only walk (below 40) |
| Passive cost | 0.01/tick × biome modifier, higher once older than 60–120 s | none |
| Turning cost | \|angle\| / 2 × π | none |
| Getting energy | eating fruit | eating an agent (takes all its energy, up to 200) |
| At 0 energy | dies | sleeps, regains 3/tick, wakes above 100 |

A predator's sprint costs about 2.55 energy/tick. After it wakes (just above 100 energy), it can sprint for only about 24 ticks (about 2.4 s) before it drops below 40 and is limited to walking speed 11.

## How predators chase (`predator.py`)

A predator only reacts to the **closest** agent it can sense.

Whether that agent is "looking at" the predator is decided by `rel_dir`: the angle between the direction the agent faces and the direction from the agent to the predator.
- It is a **half-plane** test: the predator counts as seen whenever `|rel_dir| ≤ π/2`, i.e. anywhere in front of the agent.
- The agent's real vision cone and walls blocking the view are **not** taken into account.

| Situation | Predator behaviour |
|---|---|
| Predator is behind the agent (`\|rel_dir\| > π/2`) | Chases directly at sprint speed |
| Distance < 90 (`hearing_radius 60 × 1.5`) | Chases directly, **even if the agent is looking at it** |
| Predator is in front of the agent and farther than 90 | Sprints at ±45° off the line to the agent, then turns to face the agent again |

In the last case the predator does **not** back off. Moving at 45° still brings it about 0.7 × its speed closer each tick, so it circles in toward the agent's side and back.

With no agent in range, the predator steers away from walls or wanders at walking speed.

## Predator spawning

Each tick, a new predator spawns with probability `time × 0.0001 × 0.1 / number_of_predators`. The chance grows over time, and there is no hard cap on the number of predators.

## Biomes (`biome.py`)

| Biome | Movement multiplier | Tree spawn acceptance | Fruit rate per mature tree | Energy drain multiplier |
|---|---|---|---|---|
| Forest | 1.0 | 1.0 | 0.1/s (one every ~10 s) | 1.0 |
| Grassland | 1.0 | 0.5 | 0.1/s (one every ~10 s) | 1.0 |
| Swamp | **0.5** | 0.9 | 0.08/s (one every ~12.5 s) | 1.0 |
| Desert | 0.8 | **0.1** | 0.05/s (one every ~20 s) | 1.0 |
| River | **0.3** | 0 (no trees) | 0 | 1.0 |

- **Movement multiplier:** every move, walking or sprinting, is scaled by the multiplier of the biome the entity is standing in. This applies to predators too. A sprinting agent in a river covers 20 × 0.3 = 6 per tick.
- **Tree spawn acceptance:** a new tree picks a random spot and is kept with this probability, with no retry. Trees are therefore about 10× rarer in desert than in forest.
- **Fruit rate:** each tick, a tree aged 20 s or older drops a fruit with probability `fruit rate × 0.1`. Trees die between 50 and 100 s old. The code comment says "per 100x100 area", but the rate is really per tree.
- **Energy drain:** every biome has the default of 1.0, so the biome doesn't change the passive energy cost yet.
- **River flow:** rivers define `stream_flow_speed = 5.0`, but nothing uses it, so rivers don't push entities.
- **Map layout:** the map is split into 10 Voronoi regions. Each region is randomly forest, swamp, desert or grassland, and the same type can appear more than once. One river with a radius of 20–100 units (40–200 wide) is then drawn between two random map edges, on top of the other biomes.
- **What agents see:** the agent only gets the biome name (`forest`, `grassland`, `swamp`, `desert`, `river`) for its own position. It can't see the biome of anything around it.

## Fruit and trees (`fruit.py`, `tree.py`)

Each tick, every fruit runs `grow(0.2)`:

| Time since spawn | Energy | Radius | Colour |
|---|---|---|---|
| 0 s | 20 | 5 | green |
| 0–20 s | +2 per second | +0.2 per second | green |
| 20 s | 60 (the maximum) | 9 | starts changing |
| 20–50 s | stays at 60 | stays at 9 | turns reddish-brown (only affects how it's drawn) |
| 50 s | fruit is removed | | |

- **Eating:** touching a fruit adds its current `energy` to the agent, capped at `max_energy`. It also adds `energy / 1000` to the score. A fruit eaten at 20 s or later is worth 3× a fresh one.
- **Rotting:** a fruit's `age` goes up by 2 per second, so the `age > 100` check removes it after 50 s. Over-ripe fruit loses no energy. Only its colour changes.
- **Trees:** trees drop fruit from age 20 s, at the biome's fruit rate (see Biomes). Each fruit lands 1–3× the tree's radius from its centre. A tree's radius starts at 10 and grows by 1 per second up to 20, so fruit lands 10–60 units away. Trees die between 50 and 100 s old.

## What agents observe (`creature.py` `observe()`)

Objects the agent senses (within hearing radius all around, or inside its vision cone) show up in `observations` with only these fields:

| Type | Fields |
|---|---|
| `Fruit` | `type`, `distance`, `angle` |
| `Tree` | `type`, `distance`, `angle` |
| `Agent` | `type`, `distance`, `angle`, `rel_dir`, `id` |
| `Predator` | `type`, `distance`, `angle`, `rel_dir` |
| `Edge` | `type`, `coords` (wall ends, relative to the agent) |

- **Fruit:** agents can't see a fruit's energy, radius, age or ID. A fresh 20-energy fruit looks the same as a full 60-energy one.
- **Trees:** agents can't see a tree's age or size. They can't tell whether a tree is old enough to drop fruit or about to die.
- **Other creatures:** agents can't see the energy, age or stats of other agents or predators.
- **The agent itself:** it gets its own energy, age, biome and stats.
- **Working around it:** an agent can remember roughly where fruits are and when it first saw them. A fruit seen about 20 s ago is at full value. Energy before and after eating shows what a fruit was worth. Trees that have already dropped fruit are old enough to keep producing.

## Takeaways

- **Speed:** a healthy agent sprinting (20) outruns any predator (15), and the predator soon tires down to 11. A walking agent (10) loses to a walking predator (11).
- **Low energy is deadly:** an agent below 20% of its `max_energy` can't sprint and so can't escape.
- **Starting agents vs. spawned agents:** the 5 starting agents (150 energy) can sprint from the start. Spawned agents (75 energy) can't sprint until they eat at least 25 energy of fruit, unless they mutated to a `max_energy` below 375.
- **Facing a predator** only slows its approach while it is farther than 90 units.
- **Biomes:** forest and grassland are the best places to be: full speed and the most fruit. In a river, a sprinting agent (6 per tick) is slower than a predator walking on land (11). Stay out of swamps and rivers when a predator is near.
- **Fruit timing:** a fruit is worth the most (60) between 20 and 50 s after it spawns, but agents can't see its value. Remember where fruits are and when you first saw them.
- **Detection:** predators can see slightly farther (250 vs 200) and hear slightly farther (60 vs 50) than a default agent. Mutation can push agents past both.
