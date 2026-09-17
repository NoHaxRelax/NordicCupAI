# Survival Simulator: simple strategies and game-mechanics findings

17 September 2026. Initial investigations plus continuing strategy experiments. **Small, food-efficient colonies remain the measured full-game baseline. Follow-up testing now supports water containment in suitable controlled conditions and wall holding on generated-map sites. Neither has yet established a full-game score advantage.**

## Follow-up results and active work

The initial water failures below have been superseded by substantially better controllers. The original results remain as a record of those simpler policies.

- [Water tuning](../survival/research/water_tuning.md): cached-observation control held one predator for 63.5 seconds before starvation; sufficiently rich constructed shores supported several 120-second holds. Land-to-river acquisition worked in 11/12 new straight fixtures, but all three usable native-map site trials failed after 9.2–17 seconds. Natural bank geometry and replacement remain open problems.
- [Wall acquisition and replacement](../survival/research/wall_acquisition.md): 52/53 valid arranged placements on ten generated maps held for 60 seconds. Resting-window acquisition works in suitable arranged cases. A depleted guide can alternatively establish a trap by being captured, and one legal birth can extend a holder's useful lifetime.
- [Dedicated sprinters, relays and banishment](../survival/research/sprinter_decoys.md): 142 initial controlled runs establish useful bursts and temporary relocation, but not sustained occupation or robust colony protection. Separate active tasks now develop each strategy further.
- [Wall orchard rotation](../survival/research/wall_orchard.md): the final controller maintained acquired traps for 300 seconds in 3/4 trials with native tree spawning, including both held-out geometries. These use privileged coordinates and ripeness, arranged forest sites, and one fixed predator. Full-game validation remains outstanding.
- [Fruit waiting](../survival/research/fruit_waiting.md): verified recent spawns can be worth waiting for, but unknown-age first sightings should not be assumed fresh. The tested reproduction-enabled patch gained 0.35% mean score; waiting reduced survival in the non-reproducing comparison. A separate task coordinates investigations into gameplay mechanics.

The reusable [replay debugger](../survival/debugger/README.md) includes native simulator rendering, playback speed, timeline seeking, clickable agent statistics and ten demonstrations. Recordings distinguish arranged conditions, privileged geometry and prepared food from ordinary gameplay.

The complete public source is available. This work downloaded the [official simulator](https://github.com/amboltio/Nordic-AI-Cup-2026/tree/acfc31a4003a5f91bf11032a02cd98c178ddbd7e/survival-simulator), pinned it at commit `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`, and verified every vendored file against Git blob hashes. Tests use its original movement, sensing, energy, collision and scoring methods. There were no hosted attempts, deployment, remote repository changes or final evaluation submissions.

## The three main strategy directions

| Strategy | What we learned | Recommendation |
| --- | --- | --- |
| Opposite-bank water ping-pong | Water slows predators, but detection and target switches are restrictive. All 18 simple pair controllers lost a bait in 0.3–2.6 seconds. | Keep as a secondary hypothesis; these results reject the tested controllers, not every water-trapping policy. |
| Breed agents that outwalk predators | Mechanically possible once `min(speed, sprint_speed) > 15` on matching terrain. Mutation and food costs are substantial. Stronger selection reduced mean score in this small benchmark. | Prefer useful fast descendants without starving ordinary foragers or preventing emergency births. |
| Stationary decoy behind a wall | Thin, long walls can make a predator repeatedly chase through an obstacle without navigating around it. Thickness 31/32/35 and height 70 held in 27/27 arranged tests. | Best new interaction-based strategy. Next step is observation-only trap acquisition and safe replacement on generated maps. |

### 1. Water helps escape but did not produce a reliable simple trap

Predators choose the closest **observed** agent on every active tick. Their hearing radius is 60 and works through walls; farther targets require their forward vision cone and unobstructed sight. A predator looking toward one bank usually cannot detect a decoy on the opposite bank until it comes within 60.

River movement is 30% of normal for both species. A predator's sprint moves 4.5 instead of 15 units per tick, while a dry-land agent walks 10. This makes a shore-side escape promising. But if both are in water, the agent is slowed too. Wide rivers also put the opposite decoy out of hearing range.

Movement energy is charged before the water multiplier. Predator sprint still costs 2.55 energy per tick: water makes travel expensive per distance, but does not inherently increase its rest frequency per time. A controlled straight-chase test produced identical 95 active ticks and 34 rest ticks on land and water, starting at 102 energy. Low-energy speed reduction explains why activity lasts longer than a constant full-sprint calculation predicts.

We tested 18 pair controllers over widths 40/60/80/120/160/200 and three switch positions. They knew exact geometry and predator position, had generous 500-energy baits, and still lost an agent quickly. They did not include optimized sprint escapes, food collection or replacement. Target-switch delay, sensing and acceleration on leaving water make this harder than the initial intuition suggests.

### 2. Speed selection works mechanically; strict selection was worse here

Founders walk 10/sprint 20; predators walk 11/sprint 15. Offspring independently mutate each trait 10% of the time by a multiplier between 0.5 and 1.5; walking caps at 20 and sprinting at 40. Select on `min(speed, sprint_speed)`, because a low sprint trait can cap actual motion even when the walking trait is high.

In a 10,000-trial optimistic genetic search, exceeding 15 cheap movement required a median 45 births, with a 10th–90th percentile range of 15–97. This assumes an immortal best parent, unlimited food and no predators. Actual gameplay must also pay for those births and keep the lineage alive.

The energy benefit is real: requesting 15.1 units each tick costs a founder 31.5 energy/second including young passive drain, versus 8.55 for an agent with walking speed 16. Neither cost includes turning or aging. Outwalking does not guarantee escape from obstacles, a predator on faster terrain, or several predators.

Do not deny slow founders food. They all start with the same traits, and their births are how mutations appear. Preserve a viable population and food collection, then bias spare births toward promising descendants. Newborns start at 75 energy, below the normal 100-energy sprint cutoff, so they need protected food access. Increasing max energy also raises that cutoff.

### 3. Wall baiting is the best new mechanics lead

A predator hears a decoy on the far side of a wall, continually targets it, and can oscillate against the near face instead of finding a route around it. One stationary decoy is cheaper than two moving river baits.

![Controlled wall trap](/Users/bumblebee/Github_repos/Projects/nordic-ai-cup-2026/survival/results/wall-trap.png)

For realistic obstacle dimensions 31/32/35 thick and 70 long, all 27 combinations of three initial lateral offsets and three near-forward headings retained the predator for 60 seconds. The bait did not move and spent only 60 passive energy, from 150 to 90. Separate 30-thick, 70/100-long cases also worked. Short walls fail; wider walls often allow the predator to lose the decoy's scent and wander away. Surviving because the predator disengages is not successful containment.

A broader heading test held only 5 of 12 cases, although all baits survived. This distinction matters: **we demonstrated holding an acquired trap, not robustly acquiring it from ordinary gameplay**. The tests arranged the geometry and initial positions, disabled new predators and omitted food.

Unfed stationary baits lasted 71.9, 96.0 and 122.4 seconds for assigned old-age thresholds 60, 90 and 120. The trap persisted until starvation. Reinforcements or nearby fruit are necessary; random offspring placement does not guarantee safe replacement. Ordinary agents must avoid becoming closer targets, and another predator approaching from the decoy's side remains a risk.

A practical next controller should recognize a suitable obstacle from observation history, position an expendable forager on its far side, and hold only while repeated observations show that the predator remains constrained. Test acquisition rate, food access, replacement safety and total game score. The policy cannot use the true map, predator energy or resting state used for diagnostic measurements here.

## Useful bugs and surprising rules

Detailed reproduction cases are in [edge_cases.md](../survival/research/edge_cases.md).

- **Repeated actions are executed before one world update.** Ten walk-10 requests for one ID move 100 units in one tick for 5 movement energy. Predators only receive their normal step afterward. This is a substantial action-count validation gap, but it is outside the documented one-action-per-agent contract. Hosted enforcement and competition permission are unverified; baseline policies exclude it.
- **Collision checks endpoints, not swept paths.** A synthetic evolved sprint 40 agent crossed a 30-wide wall. A related clamp-order case put an agent inside the outer boundary. Neither is a proven sustainable refuge, and reaching that mutation/geometry remains necessary.
- **Movement direction is relative, not absolute.** Despite README wording, agents can strafe or retreat while still facing a predator. Changing movement direction itself does not charge turning energy.
- **Predator observations are one move behind.** In a probe the returned distance was 40 while the predator had already moved to 25. Safety margins must account for this.
- **Sleeping predators do not kill on contact**, but sleep/energy are hidden and waking makes overlap dangerous.
- Minor list-removal and old-age ordering bugs can skip one agent's update or reward a tiny amount for eating a negative-energy agent. Neither currently merits strategy investment.
- Negative moves give no energy; oversized single moves are capped; omitting actions does not stop aging. Those proposed loopholes were falsified.

## Fruit timing and the stronger immediate baseline

Fruit starts at 20 energy, matures to about 60 after 20 simulated seconds, and rots at roughly 50 seconds. Its internal age increments twice as fast as simulation time. The game observations expose no fruit ID, age, energy, size or ripeness; fresh and ripe fruit at the same location produced identical payloads in the test.

Waiting can help only when a controller tracks a fruit known to have appeared recently under continuous observation. First sighting does not reveal its age. Do not wait blindly 20 seconds for every newly noticed fruit. Eat when energy is low or danger approaches. Waiting too close also causes automatic collection as the fruit radius grows.

Small colonies near productive trees were the stronger immediate baseline. Young stationary agents cost 1 energy/second. After a hidden age threshold between 60 and 120, an extra `0.01 × age` is charged every tick: an age 100 agent costs 11 energy/second while idle. Reproduction and limiting population demand matter. Trees die and new-tree generation halves every 300 seconds, so an orchard is not a permanent food source.

## Forty generated-map runs

Each policy ran seeds 1–9 and 42 with default map dimensions, obstacles, trees, fruit, predators and a 3000-second horizon. The engine was unmodified. Controllers receive only ordinary agent observations and simulation time, and issue exactly one action per living observed agent. Full results are in [benchmark-summary.json](../survival/results/benchmark-summary.json), with one JSON file per run.

| Policy | Mean score | Mean survival | Survival range | Reached 3000 s |
| --- | ---: | ---: | ---: | ---: |
| Upstream local random controller |32.3 |32.1 s |14.6–74.1 s |0/10 |
| Greedy fruit collection, escape and reproduction |396.8 |371.8 s |47.8–703.4 s |0/10 |
| Tree-oriented foraging, cap 6, earlier births for older agents |**631.2** |**641.2 s** |93.6–1034.5 s |0/10 |
| Same policy plus stronger speed-based breeding restriction |469.8 |479.2 s |93.6–847.5 s |0/10 |

The capped colony policy beat greedy on 7/10 seeds. It already allocates scarce birth slots fastest-first; the final row specifically tests restricting normal births to agents within 95% of the best effective walking speed, with a population-floor fallback. This is not a comparison of all possible speed selection against none. The stricter rule beat the capped policy on 2 seeds, lost on 4 and tied on 4.

These are small preliminary samples, not a winning solution. None completed the game. The policies do not yet perform wall trapping, deliberate fruit maturation, explicit elder retirement or guaranteed replacement. The age rule only lowers the birth reserve after 65 seconds and population capacity can still block births. The random comparator matches the local playground's persistent policy RNG; the example HTTP server resets its RNG each request and is a different baseline.

Runs used macOS ARM 64/Python 3.12.7 and exact upstream dependencies. The organizer recommends Linux for comparison with official evaluation. Same seed does not establish identical hosted-server behavior; different policies also alter consumption of the engine's shared RNG. No claims here depend on privileged state for policy actions. Diagnostic metrics use engine state;50-second trait sampling can miss transient mutants.

Current benchmark policy SHA 256: `ed52fe5fd816329f771909642c0230f696434a7f4e12bc69f8bc4dedbf673ad2`. Exploratory results before the population/exploration correction are preserved separately and excluded from these statistics.

## Files and reproduction

Start at [survival/README.md](../survival/README.md). It links setup, scripts, source provenance, full reports and commands. The three specialist investigations are [predator control](../survival/research/predator_control.md), [speed/fruit](../survival/research/speed_and_fruit.md), and [edge cases](../survival/research/edge_cases.md).
