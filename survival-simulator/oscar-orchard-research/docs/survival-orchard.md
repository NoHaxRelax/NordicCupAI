# Survival Simulator without predators: the orchard population policy

18 September 2026. Assumes teammates neutralise predators; the question here is how to keep the
species alive for the full 3000 s and eat as much ripe fruit as possible. Local research only; the
vendored engine is unmodified and no online attempt was made. Code: `survival/research/orchard/`.

## What the engine rewards and charges (source facts the design uses)

- Score is +1 per simulated second while anyone lives (max 3000) plus fruit energy/1000 per fruit
  eaten (0.02 unripe, 0.06 ripe). Fruit is 20 energy at spawn, 60 after 20 s, and rots at 50 s.
- Trees spawn uniformly per biome (forest 1.0, swamp 0.9, grassland 0.5, desert 0.1, river 0), fruit
  only from age 20, die around 55–62 s, and the spawn rate halves every 300 s. The live tree count
  therefore halves roughly every 600 s: about 80 at the start, 20 at 1200 s, 5 at 2400 s.
- A tree fruits for only ~65% of its life, so one tree yields ~0.065 fruit/s, about 3–4 energy/s if
  eaten ripe. An agent costs 1/s standing still, 0.05 per unit walked, and 100 per child (who starts
  with 75). Past a hidden senescence age of 60–120 s an agent loses a further 0.1 × age per second.
- Observations carry no coordinates. Heading is exact (turns are applied exactly); position drifts
  only when the engine deflects a blocked step by 10° (1.74 units per 10-unit step). Boundary
  edges are over 1000 units long and reveal the absolute pose. A child spawned on top of its
  parent reports distance 0 and its heading is `-rel_dir`.

## Measured energy budget (seed 2, first 1200 s, `debug_ledger.py`)

| Where the species' energy went | Share |
| --- | ---: |
| Births (100 each; the child receives 75) | 32.6% |
| Passive living | 29.4% |
| Walking | 22.6% |
| Old-age drain | 13.1% |
| Turning | 2.3% |

Turning is negligible; walking and births are the levers. Births are unavoidable turnover (one per
lineage per ~90 s), so the population must be small relative to the trees and must bank energy.

## How the policy works

1. **Poses and a shared map.** Each agent dead-reckons its pose. Children are placed in the parent's
   frame from the parent's observation of them, so a lineage shares one map (trees, fruit with age
   bounds, coverage cells with biome). Lineages merge when members see each other; a boundary wall
   anchors a lineage to absolute coordinates. Per-tick visual odometry matches static landmarks
   between consecutive ticks and removes unpredicted deflections; remembered trees and walls correct
   the rest. Measured: heading always exact, position within 5 units in 96% of agent-ticks.
2. **Posts.** Every young agent is assigned to one live tree (one agent per tree), sits within 30
   units of it, sweeps slowly (one rotation per 20 s costs 1 energy), and steps away from fruit that
   would be collected before ripeness. Fruit age bounds come from exact hearing history; fruit is
   eaten at the earlier of "surely ripe" (20 s after the latest possible spawn) and "about to rot"
   (47 s after the earliest). Old agents never take unripe fruit and eat last. Ripe meals rose from
   41% to 80% of all meals (`debug_eatwhy.py`).
3. **Population.** Young population is capped at 30% of the estimated tree count with a floor of
   3–5. Fruit goes to the hungriest agent within reach, ties to the fitter one. Births fill open
   slots from the fittest parents (vision range and cone, hearing, energy capacity, speed); every
   senescent agent leaves one heir, and rich senescent agents standing at a fruiting site may add
   more children since their energy is otherwise lost. Selection drives late-game agents to vision
   400, cone 1.57, hearing 100 and max energy 900–1000, all engine caps.
4. **Watching instead of walking.** An agent with no tree goes to a lookout in tree-rich interior
   terrain, spread away from other members, and sweeps: with 400-range vision a stationary sweep
   covers a quarter of the map for 1 energy/s instead of 5 energy/s walking.

## How the colonies die (rounds 1–2 of the hyperparameter search, 1019 runs)

| Death class at the moment the last agent died | Runs |
| --- | ---: |
| Policy failure: 5 or more trees still alive (median 7 trees, 15 fruit on the map) | 801 |
| Thin map: 3–4 trees left | 197 |
| True wall: 2 trees or fewer and at most 5 fruit | 19 |
| Reached 3000 s | 2 |

A run's own tree trajectory is censored by its death, so "time until the map ran out" cannot be read
from the run; instead the search objective credits a run with the full 3000 s only when it died at a
true wall. End-game post-mortems (`debug_endgame.py`) show map knowledge is not the problem: the nearest
known live tree equals the true nearest tree in every snapshot. Colonies die of energy: with 4–8 trees
left each agent has about one tree, a tree fruits only two thirds of the time, income equals cost, and
the fixed population floor keeps producing heirs that have no tree to inherit.

## Results (no predators, generated maps, 3000 s horizon)

Runs are not reproducible per seed (the engine iterates Python sets), so every comparison uses seed
means. "Full" is the number of runs that reached 3000 s.

| Policy | Seeds | Mean survival (s) | Min | Full | Mean fruit score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nursery baseline (no predators) | 4 | 1254 | 1022 | 0 | 51 |
| Society v11 baseline (no predators) | 4 | 1654 | 1541 | 0 | 68 |
| Orchard v4 (first shared-map version) | 6 | 1801 | 1346 | 0 | 104 |
| Orchard v8 (population at 30% of trees, one agent per tree) | 8 | 2309 | 2050 | 0 | 112 |
| Orchard v10, floor 4 (ripeness fixes, watch instead of walk) | 12 | 2488 | 2092 | 0 | 151 |
| Orchard v13 base (all hyperparameters exposed) | 12 | 2398 | 1909 | 0 | 147 |
| Search winner r3s2c6 (floor 8, 23% cap, breed-first, 11.5 s wait) | 20 | 2549 | 2001 | 2 | 149 |
| Same winner, recorded validation on fresh seeds 25–36 (replays in `survival/results/orchard/final/replays`) | 12 | 2500 | 1926 | 3 | 145 |
| Search runner-up r3s0c3 | 20 | 2515 | 1696 | 1 | 159 |
| Same runner-up, recorded validation on seeds 25–36 | 12 | 2324 | 1424 | 0 | 160 |

The hyperparameter search (`opt.py`, 37 parameters, 192 configurations, 1603 runs on eight Runpod CPU
pods) improved the mean by roughly 150 s over the hand-tuned base; the landscape is flat, and the
remaining failure is the 3–6 tree end-game described above.

## Late-game experiments from checkpoints

Almost all compute goes into the first 1500 s, which never fails, while the outcome is decided after
1800 s. `snapshot.py` runs a seed to 1500 s under the winning configuration and pickles engine plus
controller (rendering surfaces dropped and rebuilt); `late_sweep.py` resumes many late-game variants
from those checkpoints, each costing only the cheap last half. Same engine, same rules, 3–5× more
late-game experiments per hour.

A 35-configuration single-parameter grid on 12 checkpoints (paired against the winner, which reached
2481 s from the checkpoint) found nothing beyond noise: no extra children from old parents +58 s (6/12
checkpoints better), wider lookout reach +26 s, lower explore threshold +25 s; most changes hurt
clearly (tree reach 250: −404 s; slowest sweep: −307 s; fruit reach 320: −271 s; heir reserve 400:
−263 s; population floors 2–4: −120 to −185 s, so more late agents beat fewer). A local replicate of
seven of these on the same checkpoints reordered them, so the resume-to-resume noise is about ±150 s
on 12 checkpoints; later behaviour tests use 44 checkpoints. The winner's own endgame traces show
the mechanism: at 2000–2150 s the map has 6–8 trees but only 4–8 fruit in total (a third of trees are
under 20 s old), agents sit at live trees with under 50 energy, and heirs born at fruitless spots die
as watchers hundreds of units from the nearest tree.

Behaviour tests on 44 checkpoints (12 configurations, 528 runs, paired against the winner at 2432 s
from the checkpoint): a small "nursery" bonus that steers a parent about to spawn an heir towards a
fruiting site +14 s (22/44 checkpoints better, standard error ±56 s); heir-at-food −86 s; every
combination negative. The winner's late game is therefore at a local optimum for all parameters and
behaviours tested; further gains need a different mechanism (for example a controller that keeps a
larger energy reserve per lineage before 1800 s, or exploits the C++ engine port for far larger
searches).

## What was tried and rejected

- Culling the lowest-fitness surplus agents (feeding them last): 1590 s vs 1749 s without.
- Population at 100% of the tree estimate (v5): 1766 s on 12 seeds; one agent per tree is break-even.
- Extreme rules requested for comparison (v8 base 2309 s on 8 seeds): mandatory 10 s wait after any
  first sighting 2422 s (one seed reached 3000), 20 s wait 2034 s, no eating after age 60/90 2034/2108 s,
  minimal 6-agent settlements 1931 s, standing still after 1800 s 1941 s, all combined 1731 s.

## Reproduce

```sh
survival/.venv/bin/python survival/research/orchard/sweep.py --configs '{"v12":{}}' --seeds 1 2 3 4 5 6 --horizon 3000 --workers 6 --out survival/results/orchard/repro
survival/.venv/bin/python survival/research/orchard/analyze.py survival/results/orchard/repro --traj
```

Diagnostics (read engine state, never fed to the policy): `debug_pose.py`, `debug_ledger.py`,
`debug_harvest.py`, `debug_eatwhy.py`. Remote sweeps ran on Runpod CPU pods (see
`docs/runpod-agent-workflow.md`); results are under `artifacts/cpu-results/`.
