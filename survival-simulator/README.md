# Survival simulator

Improvise, adapt, overcome!

You are the hivemind of an entire species of herbivores. Ensure their survival by eating fruits and conserving energy, but beware of the predators roaming the territory.

## About the game
You are in control of all members of your species (agents) simultaneously. At every tick you will recieve a list of each agent's observations and status. The agents have a hearing/smelling radius and a vision cone. Any object within the vision cone is added to observations with an object type and data depending on object type. Note that creatures cannot hear/smell walls, only see them and vision can be blocked by walls.

![Entities](/images/Entity_overview.png)

After receiving an observation for each agent you will have to respond with the action for each agent. This includes moving, turning and even spawning new agents.
Agents can move in any direction but are limited by their speed/sprint_speed with sprinting having a higher energy cost per unit traveled. They can also turn at a low energy cost and even spawn new agents at a very high cost.

A newly spawned agent will inherit the parent's traits with a small chance of mutations happening in each trait. These mutations can affect the agent both positively and negatively.

During the simulation predators will spawn in to hunt the agents.

The simulation will run until all agents have died or the environment has simulated 3000 seconds corresponding to 30000 ticks.


## Environment
Running the simulation creates a random environment with different biomes which affect fruit spawn rates and creature movement. The environment will also have a number of obstacles, obstructing vision and movement.
5 agents will spawn in at the start of the simulation as well as some fruits and fruit trees.
During the simulation fruit will spawn around trees as a source of energy.
Both trees and fruit have a life cycle growing/rotting over time. 

Predators will spawn in during the simulation with increased odds over time. The predators will hunt down agents, killing any agent they touch and stealing their remaining energy. This will decrease your score based on the agent's remaining energy!


## Your goal
Your main goal is to keep your species alive for as long as possible.
If your species can consistently survive the entire simulation time, your score can be increased further by eating fruit and avoiding getting eaten by predators.

## Status and Observations

At every environment step you will receive game_status, score and a list of agent_status objects.
The game_status will be "ok" as long as the simulation is running.
The score is the current accumulated score since simulation start.
The content of each entry in the agent_status list can be seen in the table below:
| Name              | Explanation                                                   |
|-------------------|---------------------------------------------------------------|
| agent_id          | ID to keep track of agents                                    |
| observations      | List of observations for the agent                            |
| energy            | Agent's remaining energy                                      |
| biome             | The biome type that the agent is currently in                 |
| age               | How many simulated seconds the agent has been alive           |
| speed             | The maximum speed the agent can move at no additional cost    |
| sprint_speed      | The maximum speed the agent can move (higher energy cost)     |
| hearing_radius    | How far the agent can hear/smell entities                     |
| vision_angle      | The angle of the vision cone (radians)                        |
| vision_range      | How far the agent can see                                     |
| max_energy        | How much energy the agent can store                           |

The speed, sprint_speed, hearing_radius, vision_angle, vision_range, and max_energy describes static agent traits/attributes that can mutate when spawning new agents.

The sense traits/attributes are shown in the following figure:
![Traits](/images/agent_trait_ref.png)

The observations have the following format based on what is being observed:

| Observation type  | Data                                                                  |
|-------------------|-----------------------------------------------------------------------|
| Fruit             | Type, Distance, Angle (radians)                                       |
| Agent             | Type, Distance, Angle (radians), Relative looking direction           |
| Predator          | Type, Distance, Angle (radians), Relative looking direction           |
| Tree              | Type, Distance, Angle (radians)                                       |
| Edge              | Type, Coordinates (start, end)                                        |


Edges are only observed if within the vision cone. Other observations are also observed in the hearing/smell range:
![Sensing](/images/Agent_senses.png)

## Controls
After receiving the list of agent statuses and observations from the environment, your controller must decide what each agent should do during the next simulation step.
This decision should be returned as a list of ActionRequests, one for each agent (See [DTOs.py](src/utils/DTOs.py)).

Each ActionRequest must include the following fields:

|Field	            | Type	| Description                                                               |
|-------------------|-------|---------------------------------------------------------------------------|
|agent_id	        | int	| The ID of the agent this action applies to.                                   |
|move_distance	    | float | Distance to move (capped by speed or sprint_speed).                       |
|move_direction     | float | Movement direction relative to the agent's current facing (radians).      |
|turn_angle	        | float | Rotation applied this step (radians).                                     |
|spawn_agent	    | bool  | Whether the agent should attempt to spawn a new agent (high energy cost).   |

For a full example of how actions are used in practice, see [dummy_agent_policy.py](src/utils/controllers/dummy_agent_policy.py) and [agent_server.py](agent_server.py).

## Energy costs
|Action                                               | Energy cost                                     |
|-----------------------------------------------------|-------------------------------------------------|
| Walking (move_distance <= speed                     | move_distance * 0.05                            |
| Sprinting (speed <= move_distance <= sprint_speed)  | speed * 0.05 + (move_distance - speed) * 0.5    |
| Turning                                             | abs(turn_angle) / 2 * pi                        |
| Spawning                                            | 100                                             |
| Living (passive cost over time)                     | 1 / 10 * biome_energy_modifier                  |

When agents are older than a randomly chosen age between 60 and 120, their living cost will increase by 0.01 * agent.age.


## Scoring
Your score is mainly determined by how long your species survive. However, eating a fruit will increase your score by a small amount and getting eaten by predators will decrease your score based on how much energy the agent had remaining.

## Validation and Evaluation
To test your model and server connection, start a validation attempt. You can only have one attempt going at once, but attempts are unlimited. Your attempt will be put into a queue, and run when it's your turn. The validation attempts will use random seeds. 

Once you are ready to evaluate your final model, start your evaluation attempt. You only have **ONE** try, so make sure the model is ready for the final test. Your score from the evaluation is the one you will be judged on. 

Note that the evaluation attempt will run three attempts in a row and your score will be the average result, so you should ensure your agent server keeps running through all three simulations.

The evaluation will have preset seeds.

## Quickstart

Clone the repository and enter the use case folder:

```cmd
git clone https://github.com/amboltio/Nordic-AI-Cup-2026
cd Nordic-AI-Cup-2026/survival-simulator
```

### Install
The simulator requires **Python 3.10 or newer**. We recommend installing the dependencies in a virtual environment so they do not interfere with your other projects.

Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Remember to activate the environment (`source .venv/bin/activate` or `.venv\Scripts\activate`) in every new terminal before running any of the scripts below.

To verify that the installation works, run a quick headless simulation:

```cmd
python -c "from local_playground import local_simulation; local_simulation(verbose=False)"
```

You should see a stream of `Score | Agents alive | Time` lines, ending with a `Game over!` message and the seed that was used.

# Testing locally
To test the simulation locally you can run [local_playground.py](local_playground.py). This can be used to get an idea of how the simulation works. It is recommended to use this file for any potential training with "verbose" set to False to run simulations faster.

# Run on server
You can serve your endpoint locally and test that everything starts without errors by running [agent_server.py](agent_server.py). Then open a browser and navigate to [http://localhost:9052](http://localhost:9052). You should see a message stating that the agent server is running. 
Feel free to change the `HOST` and `PORT` settings in [agent_server.py](agent_server.py).

To run a simulation on the server, you can run [simulation_server.py](simulation_server.py) while the endpoint is running.

The local playground and `/predict` endpoint now use the
[expert policy](src/utils/controllers/expert_policy.py). The original
[dummy policy](src/utils/controllers/dummy_agent_policy.py) remains available for comparison.

## Expert controller

All expert parameters live in [config/expert_policy.json](config/expert_policy.json).
The controller loads and validates the file once at startup; restart the server
or local run after editing it. Unknown settings and out-of-range values fail early.

Each agent applies these local rules, with short-term crowd memory for spacing:

1. If a predator is within `perception.predator_danger_radius`, move directly
   away from the nearest one at the configured fraction of sprint speed. The
   simulator's low-energy walking limit still applies. When no nearby predator
   is observed, continue along the last escape heading for
   `memory.predator_escape_seconds` (default 2 simulated seconds). Do not
   reproduce while fleeing, including during this memory period.
2. Otherwise, consider the `perception.nearest_fruits` closest observed fruits.
   Walk toward the ripest, using distance to break ties. If any candidate lacks
   ripeness, choose the nearest fruit. Stop the requested movement at the target's
   distance to avoid overshooting it.
3. With no fruit, walk forward while gently scanning left and right. Exploration
   speed, turn amplitude, and period are configurable.
4. While foraging or exploring, gently steer toward directions with fewer recent
   agent sightings. The bias fades as sightings age and is weaker when foraging.
   It never overrides visible or remembered predator escape.
5. Reproduce when energy is **above** `reproduction.energy_threshold`, the minimum
   age is met, and movement, turning, and spawning leave at least
   `reproduction.minimum_energy_reserve`. Set `enabled` to `false` to disable it.

[Input preparation](src/utils/controllers/policy_inputs.py) exposes all own stats,
up to n fruit vectors with optional ripeness, and an optional nearby predator.
Vectors use the agent's current facing: `(distance * cos(angle), distance * sin(angle))`.
Angles are radians and movement distances are per tick, not per second. Movement
angles are **relative**, as implemented by the simulator.
Movement happens before turning; facing changes are limited by `max_turn_angle`
and ignored below `turn_deadzone`.

The competition observations currently supply no fruit ripeness. An optional
numeric `ripeness` field is supported (larger means riper); no hidden simulator
information is used. For fruit targeting, "known" means observed in the current
tick. Crowd memory uses recent agent sightings. The optional section planner is
disabled by default. Neither behavior plans routes around obstacles; the
simulator handles collisions.
The `mechanics` config section mirrors engine energy costs and the
sprint cutoff for cost estimates; editing it does not change simulation physics.
The reserve is calculated before the subsequent passive and aging energy drain.

Escape memory is independent for each agent. Its timer starts on the first tick
without an observed predator inside the danger radius. A new nearby sighting
updates the escape heading and resets the timer; after timeout, ordinary food,
exploration, and reproduction rules resume. Set `memory.predator_escape_seconds`
to `0` to disable memory. Durations use simulation time, not rendering/wall time.
The stored heading is adjusted for commanded turns, so looking around does not
rotate the intended escape path. It does not predict the unseen predator's motion.

### Short-term crowd memory

Each agent remembers the ID, direction, distance, and last-seen time of nearby
agents. Repeated sightings refresh one entry instead of increasing the count.
Directions are adjusted for the observer's commanded turns. The memory records
where neighbors were seen; it does not estimate their unseen movement or correct
old bearings for the observer's translation.

Nearby, recent sightings carry more weight. Their weight fades linearly to zero
over **4 simulated seconds**. The controller scores candidate directions by
these weighted sightings, chooses a less crowded direction, and gently turns
its normal movement toward it. More remembered neighbors give a stronger bias,
up to a cap. Roughly uniform crowding produces no preference; near-ties favor
the food/exploration direction, with a stable per-agent left/right tie-break.
This is a spacing heuristic, not a measured probability of encountering agents.

All settings are in `crowd` in [config/expert_policy.json](config/expert_policy.json):

| Setting | Purpose |
| --- | --- |
| `enabled`, `memory_seconds` | Enable crowd steering and set history duration; zero duration disables it. |
| `neighbor_radius`, `max_remembered_neighbors` | Nearby-agent radius (default 120 units) and bounded distinct-agent memory. |
| `direction_bins`, `angular_spread` | Number of candidate directions and the angular spread of each sighting, in radians. |
| `full_strength_neighbors` | Weighted neighbor count at which crowd strength reaches its cap. |
| `minimum_contrast`, `tie_score_fraction` | Ignore nearly uniform crowding and tolerate near-equal direction scores. |
| `exploration_strength`, `foraging_strength` | Maximum fraction of the angular difference to steer: 0.30 exploring, 0.12 foraging. |

Steering preserves requested movement distance, including stopping at fruit.
Turning cost is included in reproduction reserves. During predator escape,
crowd memory continues updating but cannot change the escape action. Memories
expire independently and clear on agent death or episode reset. Set
`crowd.enabled` to `false` to compare with the local expert without spacing.

Both entry points process the complete population through
`ExpertPolicy.actions_for_step(..., sim_time=...)`. This removes memory for dead
agents and resets it on an empty population, game over, or a simulation-clock
restart. Use a separate policy instance for each active simulation; the API is
intended for one active simulation in one server worker, with sequential games
supported. Custom runners should use the same batch method or call `reset()` at
episode boundaries. Single-agent `action_decision` calls can supply `sim_time`,
or otherwise use the agent's age as the clock. Prepared rule inputs also expose
the remembered escape direction, remaining timeout, weighted neighbor history, and optional section hint
for future policy reuse. Global planning requires the batch method; standalone
`action_decision` calls apply local rules unless passed an explicit section hint.

Run with the default config:

```cmd
python agent_server.py
python local_playground.py
```

For a reproducible headless local run with another config:

```cmd
python -c "from local_playground import local_simulation; local_simulation(verbose=False, seed=1, config_path='config/expert_policy.json')"
```

Both entry points also accept the `EXPERT_POLICY_CONFIG` environment variable as
the config path. An explicit `config_path` takes precedence. Relative custom paths
are resolved from the working directory; the default path works from any directory.

Run the controller and endpoint contract tests from `survival-simulator`:

```cmd
python -m unittest discover -s tests -v
```

## Shared map, clusters, and exploration

Mapping and resident/scout duties are enabled in both the local playground and
`/predict` through [config/global_planner.json](config/global_planner.json).
They are independent of the older section planner's `enabled` switch.

- Each agent begins in its own local coordinate frame. Identified agent sightings
  align positions and headings, merge map groups, and retain timestamped links.
  A group stays connected after agents lose sight of one another or a connecting
  agent dies. No continuous chain of visible agents is required.
  Overlapping newborns are recorded as contacts, but their undefined bearing is
  not used to align headings until they separate.
- Observed stone and wall segments are stored as fixed landmarks. Reobserving
  their full endpoints corrects movement estimates after collisions. Tree memory
  expires because trees can disappear; static edge memory is bounded but does not
  expire. Similar isolated stones are not enough to merge unrelated groups.
  `detect_stale_observations` uses unchanged agent age to ignore cached sightings
  on skipped simulator updates, while still integrating the action that ran.
- Biome names are sampled only at visited positions. Unknown terrain remains
  unknown; cells show sparse observations, not a surveyed biome boundary.
- Two compatible perpendicular long boundary segments can establish the world's
  origin, axes, and dimensions. Every pose and remembered feature is transformed
  together. Independently anchored groups with compatible dimensions also merge.
  This uses the supplied simulator's full endpoint observations and consistent
  positive-x/positive-y endpoint ordering, its 30-unit walls, and the fact that
  ordinary stones have sides below the default 200-unit boundary threshold. Map
  dimensions are inferred, never read from the environment. This original
  two-wall method remains the fallback; inconsistent evidence prevents anchoring.
  After anchoring, full wall endpoints also constrain position after large
  collisions. Observations that contradict the established boundary mark the
  pose uncertain and suspend new map evidence instead of creating duplicate walls.
- `estimator.shared_landmark_merges` also connects clusters that recognize the
  same stones before their agents meet. It requires two distinct nonparallel
  segments with matching lengths and agreeing endpoints, rejects ambiguous
  placements, and excludes repetitive boundary walls. Geometry matching is
  cached and bounded; it never invents an agent-to-agent sighting.
- `estimator.anchor_single_boundary` uses directed stone edges to identify the
  axes, then anchors from one suitable full boundary observation. Top and left
  walls can establish an absolute origin before both dimensions are known.
  Observed wall lengths are shared across clusters, allowing bottom and right
  walls to anchor when the missing dimension is known. Conflicting dimensions,
  ambiguous axes, and views too close to distinguish inner/outer walls are
  rejected. The display identifies incomplete bounds separately from the
  absolute coordinate frame.
- The `estimator.relocalize_stones` guard is enabled by default.
  It requires exact directed shapes for ordinary matches and two nonparallel
  segments for larger corrections. Along-edge contradictions pause new map
  samples; a previously unseen opposite face remains a valid possibility.
- The priority is a **shared absolute coordinate system, then population
  growth**. Initial groups assign up to 85% of healthy agents to scouting,
  leaving a resident when possible. A scout that sees a full wall can head
  toward its nearer corner to locate a perpendicular wall. Once all living
  agents share an absolute frame for one second, the population phase keeps
  about 20% scouting and sends residents toward remembered food trees.
  Healthy means at least 18% of energy capacity; hunger still overrides duties.
  Roles are reconsidered every 12 simulated seconds or when eligibility changes. Healthy existing
  scouts keep their routes when newborns join their frame.
- A shared frontier plan refreshes every two seconds; known obstacle edges
  exclude blocked direct routes and scouts reserve different destinations.
  A target that cannot be reached yields to another after 20 seconds. With
  `exploration.rapid_mapping.enabled`, unanchored scouts hold a travel heading
  toward new boundaries. `only_until_anchored` ends rapid behavior as soon as
  their cluster gains absolute coordinates; otherwise longer frontier routes
  remain available afterward.
  They sweep their view while moving to discover landmarks and nearby clusters.
  Visible or remembered predator escape takes priority over duties.
- Healthy lower-ranked agents preferentially scout and move at least 250 units
  from their recorded birth/first-observed position. That origin follows map
  transforms and survives later role changes. Hunger interrupts dispersal.
  High-ranked residents receive priority for patrol areas near remembered trees.
  Newborns joining the map do not restart the entire colony's alignment phase.
  Losing every absolute anchor or restarting the game does restart alignment.

`exploration.balanced_resident_homes_enabled` optionally assigns new residents
to less crowded nearby tree patches after alignment. It bounds relocation to
300 units, avoids known blocked routes, and waits until young agents recover
150 energy before giving them a lasting assignment. It is disabled by default:
in the same five-seed comparison it increased births but reduced mean living
population versus ordinary food homes (22.0 to 20.8 at 60 seconds and 31.4 to
31.0 at 120 seconds).

`exploration.directed_foraging_enabled` keeps healthy scouts on their mapping
routes instead of continually chasing distant fruit. Above the configured
energy threshold they still collect fruit within the pickup distance. Low
energy, uncertain position, resident duty, or at least 90% planner coverage
restores normal food priority.

Rapid scouting can request short sprint bursts through
`exploration.rapid_mapping.sprint_enabled`. Requests require recent progress,
enough energy, a reliable position, and a clear route inside the current vision
cone. The expert rechecks the final heading after steering and budgets both
movement and turning against the configured energy reserve. Bursts pause
between attempts and stop once the planner's coarse coverage reaches its
configured threshold. `movement.food_sprint_enabled` independently enables
short dashes toward nearby visible food until the population phase. Optional
`movement.scan_nearby_food` lets scouts survey while collecting fruit that
remains within hearing range. Predator escape keeps its existing
priority and energy rules.
Cached observations from an unchanged-age frame cannot authorize a new scouting
or food sprint. Reproduction independently preserves its post-birth reserve.
The default alignment settings allow 0.2-second bursts separated by at least
3 seconds, above 130 energy and 25% of capacity. Scouting bursts stop when the
agent's group anchors; food dashes stop when the colony enters its growth phase.

### Trait scores and reproduction

The six inherited traits are scored equally: `speed`, `sprint_speed`,
`hearing_radius`, `vision_angle`, `vision_range`, and `max_energy`:

```text
trait_ratio = current_trait / initial_population_mean_for_that_trait
total_score = mean(all six trait_ratios)
```

The baseline is frozen from the first complete population and resets between
games. A speed increase of 20% gives a speed ratio of 1.2; if all other traits
stay unchanged, the total is 1.033. A total of 1.2 means a 20% average increase
across the six ratios. Current energy and age are not inherited trait scores.
The score is a breeding heuristic, not measured survival fitness.

The top `ceil(20% * living_agents)` receive breeding priority; the bottom 20%
are preferred for dispersal. Identical populations have no elite or inferior
designation. Once scores differ, IDs deterministically break ties at the cutoff
without promoting an entire tied population. The display percentile still treats
tied scores equally. Map labels show the score (for example `1.03x`); `*` marks
the elite group.

Default reproduction settings in `reproduction.selection` inside
[config/expert_policy.json](config/expert_policy.json):

| Priority | Before alignment: energy / cooldown | Population phase: energy / cooldown |
| --- | ---: | ---: |
| Top 20% | 220 / 8 seconds | 220 / 5 seconds |
| Middle | 260 / 12 seconds | 260 / 8 seconds |
| Bottom 20% | 300 / 12 seconds | 300 / 8 seconds |

After alignment, shorter cooldowns let well-fed agents reproduce sooner while
preserving the energy thresholds that protect parents from starvation.

All still require the existing 100-energy birth cost **and** at least 75 energy
remaining after the proposed movement and turn. No breeding is requested during
visible or remembered predator escape. At 40 living agents before alignment,
or 60 during the population phase, thresholds rise to
at least the original 350 to temper growth. This is a soft target, not a hard
population cap. The later settings live in `reproduction.selection.post_alignment`;
the phase transition is controlled by `population_after_alignment` and
`alignment_hold_seconds` in the planner config. Thresholds are capped at 85% of each agent's energy capacity so
low-capacity mutations can still reproduce if the independent reserve is met.
Retries at the same simulation time return the same breeding decision.

Set `reproduction.selection.enabled` to `false` to restore the original breeding
threshold. Set `exploration.trait_roles_enabled` to `false` in the planner config
to disable trait-based duties independently. `faster_mapping_enabled` controls
the cached frontier strategy and early scouting allocation. Initial populations
are treated equally, so early growth benefits all founders.

### Centralized harvesting without predators

Run the predator-free experiment with coordinated fruit harvesting and breeding:

```powershell
.\.venv\Scripts\python.exe local_playground.py --seed 42 --no-predators --central-harvest
```

`--no-predators` disables initial, random later, and explicit predator spawns,
including after a simulation reset. Ordinary runs retain predators. The
`harvest` section in `config/expert_policy.json` controls the coordinator;
`--central-harvest` enables it for a local run and `--no-central-harvest`
provides the previous policy for comparison. Setting `harvest.enabled` to
`true` also enables the same coordinator in the agent server. The server
cannot disable predators in a remotely hosted simulation.

Mapping still runs first. Once the population shares absolute coordinates,
the controller associates fruit sightings by estimated position and records
when each fruit was first seen. Observations have **no fruit IDs or ages**.
A preceding empty view brackets a newly spawned fruit's birth; otherwise
the default estimate assumes it was already five seconds old when found,
bounded by episode start. Repeated sightings keep the original age estimate.
Reliable missing observations, expiry, and frame changes retire old tracks.

Fruit starts at 20 energy, grows by two energy per simulated second, reaches
about 60 after 20 seconds, and disappears after approximately 50 seconds.
After the absolute map dimensions are known, one territory owner chooses
each area's food and survey tasks. Every second the coordinator checks for
emergency food, fruit ripe by arrival, and overdue or unexplored patrol goals,
in that order. A viable fruit target stays assigned until collected or blocked.
Developing fruit is left to ripen while there is useful patrol work; otherwise
the agent can explicitly wait outside pickup range (56 energy by default).
Hungry agents may borrow food from another territory, and colonies below 12
agents can collect their own fruit early to fund growth. Fruit reservations
remain unique; there is no colony-wide assignment optimizer or genetic food
priority. Eating is automatic on contact, so incidental pickup while passing
through another territory remains possible.

All harvest and patrol destinations use one cached route planner over observed
rock edges. It plans around rocks with bounded A*, while local edge avoidance
checks the actual movement step. After five seconds without progress along
the route it replans once; another five seconds without progress releases the
target for a cooldown. Intentional ripening waits do not count as stuck.
The old scout/resident and section-steering policies stop supplying goals to
mapped territories; they remain available before alignment and for disconnected
groups. Walking remains the normal harvest speed; predator escape has priority.

Breeding uses the existing mean of six founder-normalized trait ratios.
Higher overall scores receive priority, while low-ranked agents can still
reproduce to recover a small population. Once established, ordinary breeding
requires at least the founder-average trait score. Trait thresholds and normal
cooldowns come from the existing population tracker, without a second set of
harvest-specific breeding thresholds or individual-trait champion exceptions.
Observed fruit arrivals and standing food stocks set a population
target between 12 and 60. Births are staggered, and parents retain at least
100 energy, in addition to the normal action-cost checks.
Valuable agents aged 40 or older can produce a replacement even above the
food target when fewer than two young agents retain at least 95% of their
overall trait score. Such births are limited to one per parent per 35 seconds and
one colony-wide per five seconds, and still obey the 60-agent hard cap.
This is a food-aware selection heuristic: mutations remain random and good
genes are not guaranteed to survive indefinitely.

The internal map shows green squares for developing fruit, gold squares for
estimated ripe fruit, and gold dashed routes. Agent labels distinguish
collecting, patrolling, scouting, ripening waits, and blocked destinations. The `harvest` field
in `planner_final.json` records tracks, estimated values, reservations, and
the current population target. These estimates use only public observations;
the benchmark's true fruit energies and inherited traits are diagnostics.

Territories balance estimated food potential rather than area alone. Observed
trees and recent fruit locations outweigh the weaker inferred-biome prior;
unknown terrain retains exploration value. Territory cells connect through
observed open corridors, and ownership persists when an agent becomes hungry.
Births and deaths trigger reassignment, while food-balance checks run on a
slower 30-second schedule. Faster biome fits do not continually shuffle agents.
Each owner patrols its own remaining gaps and stale areas, without rotating
scouting quotas. Orchard visits become due after 20 seconds; other terrain
after 60 seconds. Previously observed areas still need visits for new fruit.

A finer, bounded grid finds remaining **tree-sized unexplored patches**.
The default minimum width is 25 world units, based on the simulator's tree
placement footprint. Both sufficient area and usable width are required:
single small raster specks, narrow slivers between rocks, and spaces enclosed
by observed rock edges are excluded. Nearby healthy agents within 220 units
get worthwhile patches as territory scouting tasks. The
check is conservative and approximate at the configured 10-unit resolution;
it never consults hidden obstacles or trees. Known wall interiors are also
excluded from coarse patrol targets. Until map dimensions are known, the
existing boundary/frontier exploration continues.

Coverage is credited only from actual hearing or unobstructed vision. The
display's percentage measures survey samples, not proof that every pixel or
fruit has been seen. Cyan crosses show survey-area centers, cyan dashes show
patrol routes, and cyan circles mark worthwhile unexplored patches. Settings
are under `harvest.coverage`; setting `enabled` to `false` disables territory
ownership and patrols while retaining shared fruit tracking and navigation.
`benchmark_harvest.py --coverage` and `--no-coverage` compare these modes,
reporting fruit rot and visited free-area cells as well
as score. The `harvest.coverage` snapshot includes freshness and assignments.

After simplifying to stable food territories, matched 180-second checks on
seeds 1 and 42 raised mean score from 205.949 to 208.411 (+1.20%) and mean final
population from 17.5 to 32.5. Both seeds collected riper fruit. Coverage was
slightly lower on average, fruit rot was mixed, and living trait averages were
lower than before (still above founder baselines). More frequent biome fits
and larger colonies increased controller time. These short runs do not
establish a long-term genetic advantage; results and map images are in
`runs/simplify-optimized/report.md`.

Before the territory simplification, a matched 180-second predator-free run
on seed 42 increased visited free-area
cells from 89.56% to 95.50%, score from 199.079 to 202.038, and reduced rotted
fruit from 63 to 30. Final population was 16 in both runs. This compares the
centralized harvester with coverage disabled versus enabled; it is one short
run, not evidence that every seed will improve. See `runs/coverage-final/report.md`.

Compare matched predator-free seeds with:

```powershell
.\.venv\Scripts\python.exe benchmark_harvest.py --seeds 1 7 42 --seconds 180
```

The JSON reports separate survival time from fruit score, mean energy per
fruit, ripe-fruit percentage, population, offspring traits, and controller
time. Matching seeds gives the same initial world; later random draws can
diverge as policies choose different births and collect different fruit.
When both policies have results, `report.md` summarizes the comparison.

Before distributed coverage was added, 180-second checks on seeds 1 and 42
showed mean total score increasing from
202.064 to 203.990 (+0.95%), and mean collected fruit energy rose from
32.00 to 40.83. Results varied: seed 1 improved from 202.768 to 208.900,
while seed 42 fell from 201.360 to 199.079. The living trait averages were
above their founder baselines on both seeds, but below the old policy's
averages; these short runs do not establish an evolutionary advantage.
The local comparison is `runs/harvest-scouting/report.md`. Longer runs and
more seeds are needed to assess performance over the full 3000-second game.

### Estimated biome borders

After all living agents share an absolute frame, the controller fits an
**inverse Voronoi model** to their labelled biome samples. The simulator's map
format uses ten land generators whose biome labels can repeat. The estimator
therefore fits up to ten generator positions, allowing several forest or desert
regions, and draws the straight bisectors between differently labelled cells.
It suppresses internal seams between cells of the same biome.

Local sample connectivity initializes separate patches. A smooth classification
objective then moves the generators so observations are nearer a generator of
their own biome than one of a different biome. The fit reuses previous estimates
and adds sites when needed. These are inferred generators: sparse labels do not
uniquely determine the original generator positions or reveal unseen biomes.

Rivers are painted over the land Voronoi map by the generator. They are fitted
as a separate local overlay using river and nearby land observations, with
coarse estimated bank segments. Their observations are excluded from the land
site fit. Consequently, a river can cross several underlying land regions.

Defaults live under `biome_inference` in
[config/global_planner.json](config/global_planner.json). The first estimate is
made as soon as a shared absolute frame has enough reliable samples. Checks
start every **two simulated seconds for 30 seconds**, then each stable check
adds **two seconds** to the interval: 4, 6, 8, and so on, capped at **30
seconds**. These are intervals between checks, not times since alignment.

New biomes, contradictory observations, changing borders, or a poor sample
fit restart the fast learning period. Cheap evidence checks continue every
two seconds even when full fits are far apart, allowing new discoveries to
bring the next fit forward. Unchanged evidence reuses a well-fitting estimate.
Stability requires at least 95% land-sample agreement, no more than 5% changed
grid labels, and at least 90% agreement on new observations when 12 or more
are available. These checks are heuristics, not a guarantee of map accuracy.

The scheduler also measures fitting cost and increases the minimum interval
to target at most 0.10 seconds of fitting work per simulated second, using a
smoothed cost rounded to two-second interval increments. For example, a
two-second fit calls for at least 20 seconds between fits. The 30-second cap
still applies. A new coordinate frame restarts the schedule.

Fits require 12 samples, use at most 768 samples with position
uncertainty at most eight units, and cap the output grid at 6,000 cells.
The nominal grid spacing is 24 units. Land extrapolation is limited to 240
units from evidence and river interpolation to 100 units. Darker shading means
less support; black areas remain unknown. This support score is a heuristic,
not a calibrated probability. Partial map dimensions use a bounded observed
domain until the remaining world edges are found.

The internal view shows shaded predictions, tan dashed land borders, blue
estimated river banks, and the original observed dots. Press **B** to toggle
the estimated layer. The footer's **sample fit** measures agreement with the
training observations, not accuracy against the hidden world. Fitted layers
appear in each group's `biome_estimate` snapshot, including sites, borders,
grid labels, support, fitting time, and whether the bounds are complete.
The footer also counts down to the next scheduled check. The snapshot's
`refit_schedule` reports the current interval, next check, fit count, measured
compute cost, new-observation agreement, and scheduling reason; `updated_at`
remains the time of the last actual fit when a check reuses cached results.
The fit is retained across a newborn's brief disconnection, and invalidated
when its coordinate frame changes. Population behaviour continues as before;
inferred biome labels remain separate from observed biome memory.

The mapping benchmark also measures inferred grid labels against simulator
terrain, including river predictions. That diagnostic uses true terrain only
after the controller has produced its estimate. Unknown cells are excluded
from accuracy and reported separately through the supported-grid percentage.
In the initial fixed-cadence 60-second checks on seeds 1 and 42,
supported-grid biome-label accuracy was
86.9% and 87.1%, with predictions on 95.3% and 97.3% of model-grid centers.
Shared-frame times and population counts matched the policy without biome
inference. These measurements include extrapolated terrain and river banks;
they are separate from the higher training-sample fit shown in the viewer.
The local report is `runs/biome-inference-final-60/report.md` and the comparison
image is `runs/biome-map-final/map-comparison.png`.

An earlier 120-second adaptive-schedule check on seed 1 produced 89.7% supported-grid
accuracy with predictions at all model-grid centers and a final population
of 26. It kept five-second intervals while new evidence and model changes
continued to restart learning. Its schedule trace, summary, and comparison
image are in `runs/biome-adaptive-schedule/`. Regression tests separately
exercise the current two-second cadence and 30-second cap, explicit legacy
custom schedules, early refits on contradictory evidence, and compute-driven spacing.

The default rendered view shows the actual simulation on the left and the
controller's internal map on the right. Each disconnected group has its own
panel and coordinate frame. The right-hand view receives only the serialized
estimate: it is never aligned using true agent coordinates. It shows IDs and
roles, headings, uncertainty circles, tree and edge landmarks, sampled biomes,
and dashed historical sighting links (not present-day measured distances).
Pan an internal panel by dragging, zoom with the wheel, and press **F** or middle
click to fit it again. Matching ID colours help compare the two views.

From `survival-simulator`:

```powershell
.\.venv\Scripts\python.exe local_playground.py --seed 42
.\.venv\Scripts\python.exe local_playground.py --headless --seed 42 --max-seconds 60 --map-screenshot runs/map-comparison.png
```

Use `--no-map-view` for the original simulation view. `--map-screenshot PATH`
saves the final comparison with or without a window. Diagnostics save the full
estimated map, roles, and sighting history in `planner_final.json`.

Set `exploration.enabled` to `false` for passive mapping with the original local
movement policy. To disable mapping too, set `mapping_enabled` and the old
section-planner `enabled` flag to `false`. Estimator memory limits, boundary
assumptions, role energy thresholds, patrol radius, and exploration speed are
configurable in the same JSON file. Exploration is a heuristic, not a complete
pathfinder or a guarantee of eventual full coverage. Anchoring establishes the
coordinate frame; individual pose estimates can still drift, and high-uncertainty
poses stop adding map evidence or following absolute targets.

### Mapping benchmark

[benchmark_mapping.py](benchmark_mapping.py) compares repeatable seeds, recording
first and sustained shared absolute coordinates, the population-phase start,
population at 60/120 seconds, births/deaths before and after alignment, score,
time to 25/50/75/90/95% coarse coverage, inherited traits, sprint energy estimates,
anchored position and biome accuracy, and controller latency:

```powershell
.\.venv\Scripts\python.exe benchmark_mapping.py --seeds 1 7 42 --seconds 120 --output runs/mapping-check
```

Outputs include `results.json`, `summary.csv`, `lineage.csv`, and a
Markdown report. `--compare PATH/results.json` adds a matched-seed comparison;
`--source-root PATH` runs a frozen `src` and `config` pair using the same harness.
`--expert-config PATH` and `--planner-config PATH` override individual config
files for controlled comparisons. Each result records the effective configs
and source/config hashes.
Coverage credits the free area of an 80-unit cell once an agent visits it. It is
a coarse exploration measure, not complete reconstruction of every biome pixel.
Simulator truth is used only for diagnostic measurements and never fed to the
controller. Policy timing excludes the simulator and rendering. Short runs
measure mapping progress; they do not establish full-horizon survival fitness.

Compared with the previous walking policy on seeds 1, 7, 42, 11, and 23 over
120 simulated seconds, the alignment-then-growth defaults produced these means:

| Measure | Previous | Current |
| --- | ---: | ---: |
| Shared absolute frame, sustained interval start | 33.30 seconds | 10.94 seconds |
| Living population at 60 seconds | 18.2 | 22.0 |
| Living population at 120 seconds | 30.2 | 31.4 |
| Final mean position error, in anchored coordinates | 0.20 units | 0.08 units |

All five runs established a shared absolute frame, with no deaths before
alignment in the new policy. The benchmark confirms each frame remained shared
for five seconds; the policy's growth phase begins after its one-second hold.
Shared-frame membership and positional accuracy are measured separately.
Lowering reproduction energy thresholds increased births but reduced surviving
population, so defaults retain the protective thresholds and shorten cooldowns.
Tradeoffs: mean time to 90% coarse coverage increased from 84.40 to 94.44 seconds,
and mean score fell from 135.68 to 132.45. Coverage is secondary to coordination
and population. These short paired runs do not establish full-game survival or
genetic improvement. The detailed local report is
`runs/mapping-guarded-growth-120/report.md`.

The mapper also prepares observations once per tick, reuses geometry arrays,
and caches unchanged boundary inference. On a frozen 650-step observation/action
replay its mean update time fell from 6.09 to 2.29 milliseconds. All retained
geometry matched exactly; new origin fields and the intentional frame-revision
bookkeeping change were excluded from that comparison.

## Global section planner (disabled by default)

The [global planner](src/utils/controllers/global_planner.py) is currently
disabled in both the local playground and `/predict`. Its settings live in
[config/global_planner.json](config/global_planner.json). Restart a run after
editing them. Set `enabled` to `true` only to try section planning again.
The description below applies when it is enabled; ordinary runs use the shared
map and exploration duties above, the local expert, and crowd memory, with no
section overlay.

The competition API provides relative sightings and own stats, but no absolute
positions, facing angles, map dimensions, or full biome map. The planner therefore
starts with a partial estimated map and expands it as agents explore:

1. Track each agent's commanded movement and turns, including the configured
   movement factor for its observed biome. Repeated observations of at least two
   static trees can correct position drift.
2. Use identified agent sightings and their relative directions to align maps.
   Agents that have never been connected by sightings keep separate map groups.
   A shared group keeps its map even when its agents leave one another's sight.
3. Accumulate observed tree locations and visited biome cells. Trees supply most
   of the food weight; biome food potential supplies a small prior. Unseen space
   has no invented food value. These are **food-generation proxies**, not measured
   fruit-per-second rates: tree ages and the biomes under distant trees are hidden.
4. Divide each group's discovered extent into rectangular sections with roughly
   equal total food weight, using recursive cuts. Food-dense sections can be
   smaller. Balance is limited by grid resolution and concentrated sources.
   With no estimated food, divide known area instead.
5. Assign agents to sections, preserving ownership between replans and matching
   old and new centers when boundaries change. Aim for one section per living
   agent, capped by known cells and `max_sections`; share sections when needed.
   New agents get assignments immediately, and deaths release their assignments.
6. When an agent is closer to another occupied section's center than its own,
   blend a small movement vector toward its assigned center into foraging or
   exploration. A margin accounts for position uncertainty and boundary jitter.
   Visible **and remembered predator escape always override this bias**.

Defaults are a **15% velocity blend**, a **10-simulated-second** replan interval,
and **40-unit** grid cells. Connected map merges also trigger replanning. The
blend can slow or deflect movement; it is a soft preference, not a command that
guarantees return to a section. Speed limits and reproduction reserves are
calculated after blending.

| Settings | Purpose |
| --- | --- |
| `push_strength`, `center_distance_margin` | Strength and activation margin of the center bias; zero strength disables steering. |
| `replan_interval_seconds`, `max_sections` | Update cadence and section cap per connected group. |
| `grid_cell_size`, `max_grid_cells` | Spatial resolution and memory bound; large discovered extents automatically use coarser cells. |
| `tree_food_weight`, `biome_prior_weight` | Relative influence of observed trees and visited biome potential. |
| `estimator.biome_movement_factors`, `biome_food_potential` | Configurable movement estimates and food priors for observed biome names. |
| `estimator.odometry_error_per_unit`, `max_position_uncertainty` | Drift accumulation and the cutoff for using map guidance or adding new map evidence. |
| `estimator.landmark_match_radius`, `landmark_consensus_tolerance`, `minimum_landmark_matches`, `sighting_uncertainty` | Tree matching, agreement requirements, and observation confidence. |
| `estimator.tree_memory_seconds`, `visited_cell_size`, `max_trees_per_group`, `max_visited_cells_per_group` | Landmark expiry and bounded map history. |
| `estimator.unknown_biome_movement_factor`, `unknown_biome_food_potential` | Fallbacks for unrecognized biome names. |
| `draw_overlay` | Show estimated section boundaries, centers, and assignment lines in the local renderer. |

The position uncertainty is a heuristic, not a calibrated error guarantee.
Obstacle deflections, boundary clamping, and ambiguous tree matches can make the
map drift. Uncertain agents fall back to the local expert until sightings improve
their estimate. Section centers may be obstructed: this first planner does not
provide pathfinding. Different disconnected groups cannot coordinate overlapping
territories until their maps are aligned.

After re-enabling the planner, start the rendered simulation from `survival-simulator`:

```powershell
.\.venv\Scripts\python.exe local_playground.py --seed 7 --open-report
```

Colored outlines and circles show estimated sections and centers. Labels use
`group:section` and an approximate food weight; lines connect estimated agent
positions to their assigned centers. Thicker lines indicate a requested bias
(escape may override it); faint lines indicate uncertain positions. The renderer
uses one real agent per group **only to place the overlay on screen**. That display
transform never enters the planner or policy. Drifting and overlapping outlines
reflect the estimated maps rather than the simulator's true map.

Use `--planner-config PATH`, `local_simulation(planner_config_path=...)`, or the
`GLOBAL_PLANNER_CONFIG` environment variable for another configuration. An explicit
path takes precedence. As with escape memory, use one policy instance per active
simulation, call the batch method each tick, and reset it at episode boundaries.
The run recap folder saves planner settings in `run_config.json` and the final
estimated map, sections, assignments, and uncertainties in `planner_final.json`.
If the population becomes extinct, the planner resets and that final snapshot is
empty.

## Run diagnostics and recaps

Every local playground run now saves a separate folder under `runs/`, including
when you close the simulation window or press Ctrl+C. Reports are local files;
the recorder does not supply extra information to the expert or consume the
simulation's random stream. The generated run folders are ignored by Git.

Run the rendered simulation and open its recap when it finishes:

```powershell
.\.venv\Scripts\python.exe local_playground.py --seed 1 --open-report
```

For a faster run without rendering, or a short diagnostic run:

```powershell
.\.venv\Scripts\python.exe local_playground.py --headless --seed 1
.\.venv\Scripts\python.exe local_playground.py --headless --seed 1 --max-seconds 120 --open-report
```

The terminal prints the path to `report.html`. Open that file to see:

- Cumulative score, score gained per simulated second, fruit gains, predation
  losses, and population over time. A partial final second is normalized by its
  observed duration; raw gain is also exported.
- Survival time, births, peak population, deaths by cause, and food collected.
  Predation diagnostics show sprint eligibility and predator visibility at the
  last decision; overlapping categories describe context, not proven causes.
- One combined plot of all six trait averages among living agents, each as a
  percentage of its starting-population average (100% = unchanged). The baseline
  stays fixed throughout the run; an extinct population has no trait average.
- A table comparing founder traits, the last generation's traits, and the final
  living population, plus generation survival/reproduction/food outcomes.
- Trait–offspring associations and a cautious verdict on whether later agents
  performed better in that run.

The recap emphasizes plots and key numbers. Trait tables, comparison details,
selection associations, benchmark explanations, and downloads are collapsed.
Raw trait means and 10th–90th percentiles remain available in the CSV data.

Each folder also contains `summary.json`, `per_second.csv`,
`traits_over_time.csv`, `agents.csv`, `generations.csv`, `events.jsonl`,
`run_config.json`, and four PNG plots. The configuration snapshot includes the
seed, policy and planner settings, analysis settings, simulation settings, OS, and code
revision/status. Agent records include parent IDs, birth/death times, inherited
traits, offspring, and whether their lifetime is still incomplete at run end.

### Score percentages

Actual score is `elapsed_seconds + fruit_energy_eaten / 1000 - predation_energy / 100`.
There is no fixed, known theoretical maximum: food generation depends on the
random trajectory and future food after an early stop is unobserved. The recap
therefore distinguishes three measurements:

1. **Survival target reached:** elapsed time / 3,000 seconds.
2. **Score / optimistic benchmark:** actual score divided by 3,000 survival
   points plus full-ripeness credit for every fruit spawned during the observed
   run. This is a reference percentage, **not a percentage of the true optimum**.
3. **Score / elapsed-run upper bound:** the same fruit allowance, using actual
   elapsed time instead of 3,000. It assumes all spawned fruit could be collected
   fully ripe with no predation loss, regardless of reachability or ripening time.

The allowance per fruit is `60 + 2 * dt` energy, a conservative bound allowing for
one growth-step overshoot. Terminal tick overrun and the engine's unusual
negative-energy predation bonus are accounted for. Signed score losses and
negative score percentages are preserved. Short runs remain labelled partial and
are still compared to the 3,000-second survival target.

### Interpreting evolution

Generation 0 is the founders; each child gets `parent_generation + 1`. Generations
overlap. The "last generation" means the deepest lineage ever born, including
members that died; it is not necessarily the final living population.

Fitness comparisons use equal observation opportunities: by default, survival,
offspring, and food collected within the first **30 simulated seconds** of life.
An agent must have been born at least 30 seconds before the run ended to enter
that comparison, even if it died earlier. This prevents recent early deaths
from being compared against older cohorts while their living peers are excluded.
The verdict requires at least five eligible agents in both cohorts. If the last
generation lacks data, the report also compares the latest assessable generation.
It never ranks generations using the incomplete mean lifespan of survivors.

The verdict considers changes in survival and reproduction, with food shown as
a separate diagnostic. A larger trait alone is not evidence of greater fitness.
Trait–offspring correlations are descriptive; shared ancestry, food, predators,
location, and the founders' higher starting energy confound the comparison.
The report can find results consistent with selection, but cannot prove a causal
or repeatable advantage from one run. Compare multiple seeds before drawing that
conclusion.

Tune analysis settings in [config/diagnostics.json](config/diagnostics.json):
sampling interval, comparison age, minimum cohort size, practical-change
threshold, progress logging interval, plot DPI, output directory, and whether
to open the report automatically. Use `--diagnostics-config PATH` for another
file, `--output-dir PATH` to override the destination, or `--no-diagnostics` to
disable recording. Relative output paths are resolved from `survival-simulator`.


### Overnight policy tuning on DTU HPC

The [DTU tuning guide](hpc/README.md) includes a resumable Optuna TPE optimizer,
installation instructions, and an LSF submission script requesting 24 CPU cores,
96 GB RAM and 12 hours. It searches the predator-free centralized policy and
compares finalists against the baseline over full-length runs on separate seeds.

### OBS
The simulation is deterministic as long as it runs on the same OS. If you want to test how a specific seed runs on the validation/evaluation server, you should test on a Linux machine.

To avoid bottlenecking the system, the server will wait for responses for up to 10 seconds. If no responses are received from the agent server within that time or if the accumulated wait time reaches 600 seconds, the run will end.
