# Findings: local-food successors

All 640 BO trials (128,000 checkpoint continuations) and all 23,000 fresh full games completed. Each configuration has exactly 1000 paired seeds, 15001–16000. Training and test panels are distinct. No official validation submission was made.

The highest test mean is **expanded_local_food: 1706.0**, 95% CI 1682.1–1729.6. Its paired improvement over unchanged local_food is **61.1 points**, CI 34.7–86.8, or 3.7%. Compared with scheduled breeding it gains 155.3 points, CI 129.0–182.2. The old local_food baseline scores 1644.9 on this fresh panel; its earlier 1598.2 result used different maps.

The expanded winner uses fruit reach 210.2, tree reach 623.8 and post radius 116.8. Its breeding weights are vision 0.983, hearing 0.692, energy 1.169 and speed 1.902. These are tuned preferences, not normalized probabilities. Changes to the four breeding weights are part of every family, so the experiment does not isolate the effect of any single new feature.

One-tick risk prediction is the strongest avoidance candidate: score 1673.5, paired gain 28.6 [2.5, 54.2]. It records 139.2 predator deaths per game versus 219.0 for local_food (36% fewer), and 372.8 energy deaths versus 326.1. Death counts are not exposure-adjusted rates: duration and population differ. The predictor uses observations and conservative estimated predator movement; it does not peek at future engine states.

Two-tick prediction, periodic turns during travel, wall-risk steering and the explicit behind-predator preference all underperformed. Faster normal turning and late food-radius changes had positive point estimates but their paired intervals include zero. Turning only while idle was roughly neutral. These findings concern the specific tested implementations and search ranges, not every possible implementation of the ideas.

Tail-training gains often fail to transfer. Trait selection has the highest continuation score (406.5 versus the baseline's 244.0), yet full-game gain is only 5.9 [-20.9, 32.6]. Expanded local food reaches 406.1 in training and transfers better. All intervals are pointwise; selecting the best of 20 test candidates introduces selection uncertainty.

The winner costs 522 microseconds of policy time and 964 microseconds of whole-loop CPU time per population tick; local_food costs 564 and 1019 respectively. Measurements include the workload's population size and concurrent pod scheduling, rather than isolated policy complexity.

The policy still uses orchard's shared learned map within connected groups, odometry, boundary anchoring, food memory and exploration memory. It does **not** use Nikolaj's separate Python global-map module, nor does this implementation share predator sightings between agents. Normal travel previously limited turns to 0.25 radians per tick in policy code; the game allows faster turns at an energy cost. Walking speed above 15 does not guarantee escape across biome transitions, walls, multiple predators or close initial separations.

The previous local_food file contains 71 explicit settings, of which four food parameters were varied in its original family. The new experimental configuration contains additional optional scan, phase and avoidance controls. Each new family tunes 6–7 dimensions, including the four breeding weights. Inputs are scaled to [0,1] and GP objectives standardized; reported scores remain game-score units.

Active attributed compute was approximately $3.73 training plus $3.03 evaluation, $6.76 total. This excludes setup, storage and idle pod uptime. The ten CPU pods were left running as requested.

- [All 20 training histories, paired gains and death counts](TRAINING.md)
- [Full-game means, confidence intervals and timings](final/RESULTS.md)
- [Protocol and family definitions](PROTOCOL.md)
- Raw per-map measurements, frozen configurations and source hashes: `pod-*` and `final/shard-*`.
