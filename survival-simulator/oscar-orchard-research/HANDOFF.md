# Survival Simulator handoff: keeping the species alive without predators ("orchard" policy)

Prepared 18 September 2026, about 16:15 Europe/Copenhagen, for the NordicCupAI team. It is a snapshot of Oscar's Claude Code session "Survival simulator population optimization" (started 11:26 the same day), against upstream simulator commit `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`. The engine is included unchanged under [survival/vendor/survival-simulator](survival/vendor/survival-simulator). Start with [README.md](README.md) for setup and the file map. Replay files are excluded from this branch; result JSONs, sweep summaries and world logs are included.

## The question and the honest answer

**Question.** The trap work (branches `survival-simulator/oscar-trapper` and `codex/survival-trap-handoff-2026-09-17`) aims to hold every predator. Assume it succeeds and no predator ever touches the colony: how long can the species survive out of the 3000 s game, and how much fruit can it eat ripe? Score is survival seconds plus gross fruit energy / 1000, so survival dominates and one ripe fruit (60 energy) is worth 0.06 points.

**Answer so far.** The orchard policy roughly doubles the fruit score and adds 600 to 900 s of survival over the earlier policies, but no configuration reaches 3000 s reliably. One run in about 200 did.

| Policy | Seeds | Mean survival (s) | Min to max (s) | Fruit score | Where |
| --- | ---: | ---: | --- | ---: | --- |
| nursery baseline, no predators | 4 | 1230 | 1022 to 1487 | 50.6 | `results/orchard/baseline-nopred` |
| society v11 baseline, no predators | 4 | 1654 | 1541 to 1730 | 67.6 | `results/orchard/baseline-nopred` |
| orchard v4 | 6 | 1801 | 1346 to 2249 | 103.7 | `results/orchard/sweep-v4` |
| orchard v5 (strict lineage slots) | 12 | 1766 | 1185 to 2282 | 109.6 | `results/orchard/pod/orchard-v5-s12c` |
| orchard v8 (small population) | 8 | 2309 | 2050 to 2609 | 111.6 | `results/orchard/pod/orchard-v8-x` |
| orchard v8 + `fruit_min_wait` 10 | 8 | 2422 | 1955 to 3000 | 111.1 | `results/orchard/pod/orchard-v8-x` |
| orchard v9b (ripeness fixes) | 3 | 2527 | 2368 to 2761 | 160.7 | `results/orchard/dev`, label `v9b` |
| orchard v10 (v9b on the pod, partial) | 9 | 2224 | 1548 to 2861 | 136.6 | pod job `orchard-v10-a`, still running |
| orchard v11 = the file in this branch | 3 | 2370 | 1902 to 2659 | 163.3 | `results/orchard/dev`, label `v11` |

Read the last three rows together: the current policy family survives about 2200 to 2500 s on average, with seed-to-seed spread from about 1500 to 2900 s, and eats 2500 to 3300 fruit per game at 70 to 80 percent ripeness. Three-seed local comparisons are inside the noise. The pod's 12-seed batch is the number to trust once it finishes (section 7).

**Decisions to carry forward.**

- **Keep the population small.** The single biggest gain (v7 to v8, roughly +500 s) came from capping the young population at about 30 percent of the estimated live-tree count instead of one agent per tree. A tree only fruits about 65 percent of its life, so one agent per tree is break-even and nobody accumulates energy.
- **Keep fixing ripeness, not rules.** The second gain (v8 to v9b, fruit score +45 percent) came from correcting how the policy knows a fruit's age, not from new behaviour. Extreme rules (long forced waits, refusing food to elders, standing still late, tiny colonies) all lost survival or fruit or both.
- **Do not cull.** Starving surplus agents on purpose hurt in every test. Let the birth plan control population instead.
- **One agent per tree post.** Two slots per tree was never better.
- **The late game is the open problem.** After about 1800 s fewer than ten trees remain, the population sits at the floor of three, and lineages end when the last agents cannot find a fruiting tree in time. Every full game so far ended this way, not through early collapse.

## 1. Engine facts the design relies on

Verified by the diagnostic scripts in `survival/research/orchard/debug_*.py` and by reading the vendored source. Observations are the only legal input; the harness reads engine state for diagnostics only.

| Mechanic | Verified behaviour | Consequence for the policy |
| --- | --- | --- |
| Heading | Turn requests are applied exactly. | Dead-reckoned heading never drifts. |
| Position | Drifts only on unpredicted collision deflections (10 degree steps, about 1.74 units per 10 unit move) and boundary clamps. | Visual odometry against static landmarks (fruit, trees, edge endpoints) corrects the median shift each tick. |
| Newborn placement | A child spawned exactly on its parent reports distance 0 and its heading is `-rel_dir`. | The child gets the parent's frame immediately, so a lineage shares one map. |
| Boundary edges | Edges longer than 1000 units are the map boundary (1600 by 1200). | Seeing one gives the group an absolute pose; groups that see each other merge frames. |
| Fruit | 20 energy at spawn, 60 after 20 s, rots at 50 s. | Eat 20 s after the latest possible birth time, or just before 50 s after the earliest. |
| Trees | Live about 55 to 62 s; the live count halves roughly every 600 s from about 80. | Population cap follows `0.3 * 80 * 0.5^(t/600)`, clamped to [3, 20]. |
| Old age | Hidden threshold at 60 to 120 s; afterwards drain of 0.1 times age per second. | Detected from an energy drop that exceeds the known action costs; old agents stop travelling and turn energy into one heir. |
| Births | Parent pays 100, child starts at 75. | About a third of all energy is spent on births; each birth must be justified by an open slot. |
| Costs | Passive 1 per second; walking 0.05 per unit; sprint 0.5 per unit beyond walk speed; turning up to 0.5 per half turn. | Standing still with a wide sweep is the cheapest way to find trees. |
| Determinism | The engine iterates Python sets, so the same seed does not reproduce. | Judge changes on 6 or more seeds by the mean, never by one seed. |
| Platform | Official evaluation is Linux; local runs are macOS. | Pod (Linux) results are the reference. |

## 2. Where the energy goes

Measured on 18 September with `debug_ledger.py`, `debug_harvest.py` and `debug_eatwhy.py` on the v7 to v9 policies (seed 2 and 4, 900 to 1200 s):

- Births take about 33 percent of all energy, walking 23, passive drain 29, old-age drain 13, turning 2.
- A tree yields about 0.065 fruit per second while it lives, and only fruits about 65 percent of its life. One agent per tree is break-even.
- Before v9, 60 percent of meals were unripe: old agents ate anything, the fruit age bounds derived from coverage cells were wrong, and remembered fruit was never checked for occlusion. After v9b about 69 to 80 percent of meals are ripe (`claimed_ripe_rule` 69 percent, uncertain-age claims 11.5 percent, the rest emergency eating by hungry agents).
- At seed 4, t about 800, v7 had 19 agents spread over about 30 mostly fruitless trees. With the v8 cap most agents hold 150 to 460 energy at t = 100 against 50 to 120 before.

## 3. What the policy does (`survival/research/orchard/orchard.py`, class `OrchardPolicy`)

Observation-only. Input per tick: the list of per-agent observation dictionaries and the simulation time. Output: one `ActionRequest` per living agent. It keeps a `Mind` per agent and a `Group` per shared coordinate frame.

1. **Poses and maps.** Every agent dead-reckons its pose from its own last action. Newborns are placed in the parent's frame from the parent's observation of the child. Groups merge when members see each other (agent observations carry ids) and anchor to absolute coordinates when a boundary edge is seen. Each group remembers trees (with first and last sighting and whether it was seen appear), fruit (with a birth-time interval `born_lo` to `born_hi`), obstacle edges, and 100 unit coverage cells with biome. Remembered items that should be visible but are not are deleted, taking occlusion by remembered edges into account. Hearing works through walls; vision does not.
2. **Fruit age.** A fruit first seen at a spot that was inside the agent's exact previous view (cone minus a margin, or hearing disc) is born now. Otherwise the latest time any group member's hearing disc covered the spot bounds its age. Ripeness rule in `_ready`: eat at `min(born_hi + 20, born_lo + 47)`, earlier only when the agent cannot afford the remaining wait. `fruit_min_wait` forces an extra wait after the first sighting (0 by default, 10 was the best variant).
3. **Posts.** `_assign_posts` gives each young agent one live tree within `tree_reach` (420) with a free slot (`tree_slots` 1), valued by expected future fruit plus fruit already there, minus travel, wait and a distance penalty. An agent keeps its post for `min_stay` 15 s and re-evaluates every `repost_every` 10 s, switching only for a gain above `switch_gain` 100. At the post it sits inside `post_radius` 30, sweeps slowly (`sweep_rate` 0.03 rad per tick) and steps back from unripe fruit within 16 units so it is not auto-collected.
4. **Fruit claims.** `_assign_fruits` hands each ready fruit within `fruit_reach` 200 to one agent: agents owing an heir first, then by energy bucket, then fitness, then distance. Old agents only claim within 60 units and only surely ripe fruit.
5. **Without a post.** Below `explore_energy` 200, or within `watch_patience` 30 s of last having a site, the agent watches: `_watch_post` picks the interior spot within `watch_reach` 500 whose vision disc covers the most tree-friendly cells (forest 1.0, swamp 0.9, grassland 0.5, desert 0.1, river 0) not already watched by another member, walks there once and sweeps at triple rate. Richer agents explore the stalest cell within `explore_radius` 450, weighted by biome tree rate, and sweep on arrival. Stuck detection after 15 ticks without progress triggers a random 90 degree detour and blocks the target for 40 s.
6. **Old age.** `_observe` compares the energy drop with the known action costs; a deviation of 0.45 to 2.5 while below maximum energy, or age above 120.5, marks the agent old. Old agents idle at their post, eat last, and are asked for one heir as soon as they hold more than 101 energy.
7. **Reproduction plan (global, per tick).** Cap = `cap_mult` 0.3 times the tree estimate (`n0` 80 halving every `tree_half` 600 s), clamped to [`cap_min` 3, `cap_max` 20]. Then, in order: every senescent or 55 s old agent owes exactly one heir (`heir_reserve` 250 energy needed while young, 101 when old; weak lineages below the young median fitness minus `heir_slack` are skipped); emergency births when two or fewer agents live; up to `births_per_tick` 3 open slots go to the fittest parents above the breeding reserve (200 energy) that have food nearby. Fitness is swept vision area, hearing disc, energy capacity and speed from public traits. No one is culled (`cull` False).

All parameters are keyword arguments of the constructor and can be passed with `--kw '{"cap_min": 4}'` to the harness or per config to the sweep.

## 4. Versions and what each one taught

Only the final file is in this branch; intermediate versions were not kept, so the deltas below are reconstructed from the session and from the sweep configurations. Survival numbers are means unless stated.

| Version | Change | Result | Lesson |
| --- | --- | --- | --- |
| v2 | first version: posts, dead reckoning, group merging | 600 s smoke runs, 60 plus agents | population exploded |
| v3 | spatial index for trees and fruit | 6-seed sweep crashed with `KeyError` in the grid | in-place tree moves left stale grid cells; fixed by re-binning (`Group.move_tree`) |
| v4 | first full sweep | 1801 s (6 seeds), 407 agents created per game | colonies died from churn: fruit went straight into children, the late breeding reserve was never reached |
| v5 | strict lineage slots: one heir per senescent agent, extra births only into open slots by fitness, old agents eat last | 1766 s (12 pod seeds); cap times 0.7: 1571, cap times 1.4: 1722, 2 slots: 1834, `cap_min` 6: 1697 (5 seeds each) | no variant separable from noise; still 420 agents created per game |
| v6 | culling of surplus agents, 2 tree slots as default | 1590 s; one slot: 1846; no cull: 1749; `heir_reserve` 130: 1800; `dist_pen` 0.2: 1747; `tree_reach` 250: 1762 (8 seeds each) | culling and double slots both hurt; reverted |
| v7 | one slot, no cull, ledger diagnostics | 2051 and 1461 s (2 local seeds) | agents spread thin over fruitless trees; nobody accumulates energy |
| small03 | v7 with `cap_mult` 0.3 and wider fruit reach | 2261 s, peak 33 agents (seed 4) | agents hold 150 to 460 energy at t 100 instead of 50 to 120 |
| v8 | small population as default, plus "extreme" variants on the pod | v8 2309; `fruit_min_wait` 10: 2422 (one run reached 3000); wait 20: 2034; `no_eat_age` 60: 2034; 90: 2108; minimal colony: 1931 with fruit 72.7; `late_still_t` 1800: 1941; combination: 1731 (8 seeds each) | only a short forced wait helps; the population cannot be shrunk further; standing still late loses fruit and survival |
| v9, v9b | ripeness fixes: exact previous-view test for "seen born", hearing-disc history as the age bound, occlusion in memory verification, old agents wait for ripe | 2527 s, fruit 160.7 (seeds 1, 3, 5 local) | ripe share 60 to 80 percent; fruit score up 45 percent |
| v10 | v9b pushed to the pod; adds the watch-first rule (`explore_energy`, `watch_patience`) | partial, 31 of 72 runs at 16:12: v10 2224 (9 seeds), plus `fruit_min_wait` 10: 2299 (10 seeds), `cap_min` 4: 2362 (5 seeds) | see section 7 for the final numbers |
| v11 | chosen lookout positions (`_watch_post`, `watch_reach`, `watch_refresh`) | 2659, 2547, 1902 s; fruit 163.3 (seeds 1, 3, 5 local) | not separable from v9b with 3 seeds |
| v11cm6, v11cm8 | v11 with `cap_min` 6 and 8 | running locally at snapshot time (seeds 1, 3, 5 each) | tests whether a higher late-game floor helps |

## 5. How a colony dies now

Trajectory of v11 seed 1 (2659 s), from `results/orchard/dev/world-v11-seed1.log`, sampled every 50 s:

| t (s) | Alive | Trees | Fruit | Mean energy |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 18 | 50 | 83 | 211 |
| 1000 | 12 | 31 | 45 | 238 |
| 1500 | 6 | 17 | 38 | 161 |
| 2000 | 5 | 11 | 18 | 71 |
| 2500 | 2 | 6 | 6 | 68 |

The last agent died at 2659 s in watch mode with 10 energy at age 57, with 7 trees and 14 fruit on the map. Of the 13 deaths in the last 300 s, 9 were old agents. The same shape appears in every long run: the cap floor of three is reached around 1800 s, trees keep halving, and the remaining agents find trees too late or spend their energy walking to them. Watch mode (v10, v11) was meant to make that search cheap; it did not change survival measurably at 3 seeds.

Things that might fix it, in the order I would try them:

1. **Confirm the population floor.** `cap_min` 4 leads the partial v10 batch; the local `cap_min` 6 and 8 runs answer whether more is better. A floor of 4 to 6 gives more eyes on the map late without exceeding the fruit supply.
2. **Let late lineages breed on sight.** Late births currently need 200 energy and food nearby. With 5 trees on the map, a 250 energy elder should split as soon as it sees any fruiting tree, so two agents can watch two trees.
3. **Spread late watchers.** Groups merge into one frame, which is right for mapping, but `_watch_post` only discounts spots within 0.8 vision radius of another member. Push late watchers to opposite halves of the map explicitly.
4. **Keep the 10 s forced wait.** It was the only rule to reach 3000 s once; it is neutral in the partial v10 batch. Decide on the 12-seed numbers.
5. **Fold the policy into a real game.** This branch never spawns predators. The trapper policy and this one have to be merged into one controller (posts and births here, guides and traps there), and the merged policy needs the recording rules in the team `AGENTS.md`.

## 6. How to run things

From `survival-simulator/oscar-orchard-research` (see README for the virtual environment):

```sh
# one seed, full length, with a world log every 50 s
survival/.venv/bin/python survival/research/orchard/harness_np.py --seeds 1 --horizon 3000 --label mine --out survival/results/orchard/dev --world-log survival/results/orchard/dev/world-mine-seed1.log --world-every 50

# parameter override
survival/.venv/bin/python survival/research/orchard/harness_np.py --seeds 1 3 5 --label cm6 --out survival/results/orchard/dev --kw '{"cap_min": 6}'

# parallel sweep: configs x seeds, summary table at the end
survival/.venv/bin/python survival/research/orchard/sweep.py --configs '{"base":{},"wait10":{"fruit_min_wait":10}}' --seeds 1 2 3 4 5 6 7 8 --horizon 3000 --workers 8 --out survival/results/orchard/sweep-mine

# per-seed table and trajectories of any results directory
survival/.venv/bin/python survival/research/orchard/analyze.py survival/results/orchard/dev --label v11 --traj --every 300

# baselines without predators
survival/.venv/bin/python survival/research/orchard/harness_np.py --policy nursery --seeds 1 2 3 4 --label np --out survival/results/orchard/baseline-nopred
```

`harness_np.py` imports the society harness and replaces `Environment.spawn_predator` with a no-op before the world is built. Everything else is the unchanged engine. `--record` writes a state-only replay through `survival/debugger/recorder.py` (labelled as a diagnostic fallback, not organizer rendering). Each full-length run takes 7 to 12 minutes of one CPU core.

Diagnostics (`survival/research/orchard/debug_*.py`, argument order seed then horizon, policy overrides through `ORCHARD_KW='{"cap_min":4}'`):

- `debug_ledger.py`: energy spent per category (births, walking, passive, old age, turning) against the engine's own accounting.
- `debug_harvest.py`: fruit spawned, eaten, ripe and rotted per 300 s window, and how far rotted fruit was from agents and from known map.
- `debug_eatwhy.py`: for each meal, why the policy ate it at that energy (claimed ripe, uncertain age, hunger, old).
- `debug_pose.py`, `debug_residual.py`, `debug_odometry.py`, `debug_heading.py`: absolute pose error of anchored groups, when it jumps, and dead-reckoning versus engine truth.

## 7. Jobs that were still running at the snapshot, and the CPU pod

The Runpod CPU pod `cpu-a` (pod `1wyg2o2e00a22g`, 32 vCPU, about 0.96 dollars per hour, 30 dollar budget) belongs to Oscar and is managed by Codex; Claude only runs jobs on it. Its files live under `/workspace/users/oscar/cpu-a` (the code bundle under `code/survival`, runs under `runs/<job>/results`). From the laptop repository the helper is `python3 compute/gpu.py {push,exec,start,status,pull} cpu-a`; `build_cpu_bundle.py` produces the explicit-allowlist tarball to push. Never run `setup` or `check` on it, and never stop it without Oscar's say.

- **Pod job `orchard-v10-a`**, started 13:54 UTC: 6 configurations times 12 seeds (`v10`, `v10w10` = `fruit_min_wait` 10, `capmin4`, `capmin5`, `capmin4w10`, `oldexplore` = `explore_energy` 60 and `watch_patience` 0). 31 of 72 rows were done at 14:12 UTC; expect the rest by about 14:45 UTC. Pull with `python3 compute/gpu.py pull cpu-a runs/orchard-v10-a/results survival/results/orchard/pod/orchard-v10-a` and read `summary.json` or run `analyze.py` on it. This is the batch that decides `cap_min` and the forced wait.
- **Local runs** `v11cm6` and `v11cm8` (seeds 1, 3, 5, started 16:05 local) write to `survival/results/orchard/dev` on Oscar's laptop.

Finished pod batches are already in this branch under `survival/results/orchard/pod/` (v5 12 seeds, v5 variants, v6 variants, v8 extremes), each with `summary.json` and the per-run JSON.

## 8. Boundaries

No competition validation or evaluation endpoint was called and no score was submitted. The vendored engine is unmodified; the harness wraps `kill_agent`, `remove_fruit` and `non_agent_step` read-only for diagnostics. The policy reads only observation fields and simulation time. Runs are local research and, per the team rules, every completed run should stay inspectable: this branch keeps the JSON results and world logs but not the replay files, which remain on Oscar's laptop.

## Update 18 September 2026, late evening (Claude, orchard session)

- Read `docs/survival-orchard.md` first: energy budget, ripeness fixes, tree-limited population, the
  1600-run hyperparameter search, death-by-tree-count analysis, checkpoint methodology, rejected ideas.
- Best configuration: `survival/results/orchard/best-config.json` (pass as `--kw` to `harness_np.py`,
  `sweep.py` or `opt.py`). On 32 fresh seeds without predators: mean survival ~2530 s, fruit score ~148,
  score ~2680, 5/32 runs reach 3000 s. Recorded validation runs: `survival/results/orchard/final/`.
- Tools: `sweep.py` (parallel sweeps, `--record`), `opt.py` (random + hill-climbing search with successive
  halving, tree-wall-credited objective), `opt_report.py`, `analyze.py` (`--pair`, death classes),
  `snapshot.py` + `late_sweep.py` (late-game experiments from 1500 s checkpoints), `debug_*.py`
  diagnostics (engine truth, never fed to the policy).
- Search journals (every configuration and run): `artifacts/cpu-results/*/`; `artifacts/late-grid.json`
  and `confirm-configs.json` are the late-game grid and the confirmed leaders.
- Where it stands: the tuned late game is at a local optimum (single-parameter and behaviour changes on
  44 checkpoints are all within +-60 s). Colonies die at 5-8 trees with 4-8 fruit on the map because heirs
  restart at 75 energy; a different mechanism is needed to reach 3000 s reliably.
- `survival/fastsim/` is the C++ engine port from the parallel Claude session (bit-identical to the
  Python engine up to its set-ordering nondeterminism); build once with `fastsim/build.py`, then
  `SURVIVAL_ENGINE=fast` on any harness command. `research/rustsim/` holds only a bit-exact port of
  CPython's random.Random (parked).
