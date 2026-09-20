# Why harvest bursts miss predator contact

20 September 2026. Local mypc Ubuntu WSL diagnosis. Eight generated games, seeds 101–108, native C++ simulation and native `pred_best.json` policy, unchanged Python `contact_nearest` harvester. Budget 200,000, trigger 30, closing 6, cooldown 50 seconds, at most 100 bursts; horizon 3000 seconds or extinction. No API requests, deployment changes, or policy fixes.

The interaction fails because the burst trigger does not establish that an awake predator will touch the drained agent during the next predator phase. In this sample the action application and list-mutation skip worked every time.

## Measured outcomes

| Result | Bursts |
|---|---:|
| Negative-energy predator meal | 49 |
| Sleeping predator; eating phase skipped | 14 |
| Predator pursued the drained agent but steering left it outside contact | 12 |
| Predator pursued the drained agent but its movement range could not reach contact | 5 |
| Predator selected different prey | 1 |
| Total | 81 |

Transfer rate: 49/81 = 60.49%. All 49 meals occurred in the burst tick. All 32 failed farms survived that tick through the intended list-mutation skip, then died from negative energy exactly one tick (0.1 simulated seconds) later. The exact burst times, selected farms, sacrifices and outcomes match the earlier uninstrumented results for these eight seeds; final scores match their stored three-decimal values.

Fourteen sleeping targets account for 43.75% of misses. Eleven were already deeply negative from earlier successful transfers, so the policy wasted more agents on predators it had already disabled. Three were naturally resting at positive energy below their wake threshold. Resting predators skip movement **and eating**, even if the agent is overlapping them.

The seventeen pursuit misses comprise twelve cases where the available step length could geometrically reach the contact circle, but the actual heading/steering did not, and five where even a straight step of the available length could not reach it. Predator rotation is capped at 0.3 radians per tick; low energy and terrain can reduce a nominal 15-unit step to 11, 7.5, 5.5, or 3.3. Collision redirection also occurred in three of these seventeen cases, but all three would already have missed along their requested, unredirected paths. Therefore these are not counted as three independent obstacle-caused failures.

## Concrete reproductions

| Seed / time | Observation and exact trace |
|---|---|
| 104 / 311.9 s | Farm 139 only 4.38 units from predator, but predator resting at energy 3. No eating occurs; farm starves at 312.0 s. |
| 102 / 314.1 s | Farm 139 at 27.02 units from a previously disabled predator, energy -97,455.6. Another burst cannot transfer while it sleeps. |
| 101 / 363.2 s | Farm 183 reported at 26.3 units; actual distance before this predator phase was 17.69. Predator targeted farm 183 but started 1.281 radians off heading. After its 15-unit move, separation was 15.5806, outside the strict `< 15` contact test. Farm starved at 363.3 s. |
| 106 / 266.1 s | Farm 118 at actual distance 22.53; swamp reduced the predator step to 7.5. Final distance 15.0920, just outside contact. |
| 104 / 682.7 s | Burst drained farm 309, but predator 5 selected agent 311 after the action/death phase. Distance to farm 309 increased from 20.87 to 26.47. |

All eight complete reproductions have state replays under `results/replays/`. Every simulation step is passed to `ReplayRecorder`; ordinary frames are sampled every five seconds, births/deaths/rest changes retain exact frames, and burst windows are forced at 0.1-second resolution. These are explicitly labelled C++ state-only fallback recordings, not original Python-renderer images. Full per-burst inputs, pre-action world snapshots and C++ phase traces are in the eight adjacent `results/contact_nearest-*.json` files. Duplicate action multiplicity is recorded in burst metadata; the replay action map contains only the final action per agent. The local viewer was not running; the offline `ReplayCatalog` discovered all eight files with zero errors (1,185–1,403 frames per game).

## Source-level cause

`survival/nightsim/serve/harvest.py:41` takes the nearest reported predator distance, then `:69` checks whether that distance fell by six. This does not distinguish predator motion from agent motion or a change in which predator is nearest. The first-contact branch at `:66` also permits a burst with no motion history. No resting state, predator identity, heading forecast, terrain speed or post-action prey selection is checked.

The harvester removes the farm's and sacrifice's original actions at `:104`, leaving both stationary while draining them. Other agents still execute their original moves and births. The predator chooses prey after those changes, so the farm's earlier distance is not enough to establish that it will remain the selected prey.

In the reference source, `environment.py:660` records observations before the predator movement phase at `:674`. Thus a reported predator pose is usually already one predator move old when used for a decision. At `:676–681`, a resting predator returns before contact handling at `:709–724`. `predator.py:32` chooses closest observed prey, and `:38` clamps steering to ±0.3 radians. Contact is strictly less than the sum of radii (15).

## Implication for the next change

Prioritize tracking predators already disabled by confirmed transfers and excluding those locations until their conservatively estimated wake time. A shorter distance alone cannot solve the sleeping-target case: the 4.38-unit example was already well inside contact.

For remaining targets, replace the distance-drop trigger with a contact prediction that accounts for our own movement, the unobserved prior predator move, its pending move, steering, possible low-energy/terrain speed, obstacles and other agents after their actions. The public observations omit predator IDs and resting state, so tracking/association uncertainty must remain explicit. These changes are recommendations from the diagnosis, not a tested replacement policy or a claim of 100% transfers.

Raising action count increases the negative energy of a successful meal. It does not extend the farm's one-tick survival window or make a sleeping/poorly aligned predator eat it. This investigation did not test HTTP capacity; that is being handled separately.

## Reproduction artifacts

`survival/research/action_bug_debug/diagnose.py` runs the unchanged policy and collects read-only traces/replays. `analyze.py` classifies outcomes from those traces. `instrumentation.patch` applies only to an isolated copy of `nightsim/_nengine.cpp`; it exposes `dbg_trace_burst` and `dbg_pop_trace` and logs existing calculations without changing physics, RNG, or policy inputs.

The recorded build is on mypc at `C:/Users/oscar/NordicCupAI/burst-debug-3Q8ISQ`. Example:

```sh
python3 diagnose.py --runtime /mnt/c/Users/oscar/NordicCupAI/burst-debug-3Q8ISQ \
  --out /mnt/c/Users/oscar/NordicCupAI/burst-debug-3Q8ISQ/results --seed 101
```

The policy name is the existing `pred_best.json` orchard/evasion baseline. It is not independently established here as Lucas's separate approximately 1700-score policy. Hidden state is used for diagnosis only. Native-port evidence does not establish identical hosted execution.
