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

## Takeaways

- **Speed:** a healthy agent sprinting (20) outruns any predator (15), and the predator soon tires down to 11. A walking agent (10) loses to a walking predator (11).
- **Low energy is deadly:** an agent below 20% of its `max_energy` can't sprint and so can't escape.
- **Starting agents vs. spawned agents:** the 5 starting agents (150 energy) can sprint from the start. Spawned agents (75 energy) can't sprint until they eat at least 25 energy of fruit, unless they mutated to a `max_energy` below 375.
- **Facing a predator** only slows its approach while it is farther than 90 units.
- **Detection:** predators can see slightly farther (250 vs 200) and hear slightly farther (60 vs 50) than a default agent. Mutation can push agents past both.
