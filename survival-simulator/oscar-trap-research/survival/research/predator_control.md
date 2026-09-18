# Predator control: controlled source-based probes

Date: 17 September 2026. Source: public Nordic AI Cup simulator, commit `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`. Personal/student competition research. No hosted validation, evaluation or submission was used.

**Best lead: a stationary bait behind a thin, sufficiently long obstacle can hold a predator at almost no running cost. The opposite-bank water idea did not work in the simple variants tested.** These are controlled mechanics results, not evidence of better full-game scores.

Reproduce with:

```sh
survival/.venv/bin/python survival/research/predator_control.py
```

Results and sampled trajectories: `survival/results/predator-control.json`. The harness imports upstream classes and uses their original movement, sensing, collision and tick methods. It arranges initial positions, substitutes a flat forest with a straight river where relevant, disables new trees/predators, and omits food. River and wall setup use privileged geometry, which is not a deployable controller. The orbit/rear-follow action logic itself uses only actual cached observation fields.

## Actual predator rules

- A predator selects the nearest **currently observed** agent anew every active tick, with no target commitment. It hears/smells agents within 60 units, in all directions and through walls. Farther agents require its 60-degree vision cone, range 250, and unblocked line of sight. `predator.py:10–44`, `creature.py:25`, `creature.py:131–159`.
- Within distance 90 it directly chases. At greater distance it also directly chases when the agent faces away; otherwise it approaches obliquely. Direct chase turning is capped at 0.3 radians per tick. Movement and vision orientation are separate for player agents. `predator.py:37–62`, `environment.py:606–618`.
- Baseline movement per tick: agent walk 10 / sprint 20; predator walk 11 / sprint 15. These are distances per 0.1-second tick, not units per second. Contact kills below center separation 15 for default sizes.
- Sprinting is disabled below 20% of maximum energy. Thus the normal agent cutoff is 100; an unfed newborn starts at 75 and cannot sprint. Predator cutoff is 40. `environment.py:508–518`.
- Predator energy has no passive depletion. It rests when depleted and restores 3 energy per tick, waking only above 100. It can replenish by eating agents. The resting branch skips contact kills. `environment.py:675–727`.
- Observations expose predator bearing, distance and relative orientation, but neither energy nor resting status nor a predator ID. Player state exposes only the agent's current biome, not a map of nearby river borders. Observations are cached before the predator moves that tick. `environment.py:575–598, 649–660, 675–704`.

## Water: a useful speed difference, but no simple reliable trap yet

River movement is 0.3 times normal for **both** species; movement energy is charged before that multiplier. Swamp is 0.5 and desert 0.8. All currently defined biomes have the same baseline passive energy drain. The river's declared flow-speed field is unused.

| One action | Forest distance | River distance | Energy either terrain |
|---|---:|---:|---:|
| Agent walk | 10 | 3 | 0.5 |
| Predator sprint | 15 | 4.5 | 2.55 |

So an agent on dry land can comfortably outwalk a predator still in water. Water costs the predator 3.33 times as much energy **per actual distance**; it does not increase expenditure or force extra rest **per tick** for the same requested motion. In the straight-chase probe, a predator starting at 102 energy acted for 95 ticks and rested for 34, on both terrains. It covered 1,145 units on land versus 343.5 in water. Real turns change these figures a little.

The main obstacle to opposite-bank switching is sensing. When a predator faces one bank, the agent on the other bank must normally get within 60 units to be considered at all. For example, in a 120-unit-wide river with agents 5 units onto each bank, a centered predator cannot hear the agent behind it: that agent is 65 units away. A 40-unit river with the predator 15 units off center leaves the opposite bait at 40 units, making switching possible in principle. River generation chooses radius 20–100, so nominal widths span 40–200 and local geometry can vary further.

The 18 tested pair variants used widths 40, 60, 80, 120, 160 and 200; switches at 0, 5 or 10 units past center; each bait moved to its bank or retreated 50 units; both stayed on land and walked. Each started with an unusually generous 500 energy, and the controller knew exact geometry and current predator position. **Every variant lost at least one bait within 0.3–2.6 seconds.** The turning delay, loss of awareness of the opposite bait, and the predator accelerating on exiting water defeated these simple rules. Sample trajectories are saved.

This rejects this particular simple controller, not every imaginable two-agent water strategy. Further water work needs reliable acquisition, early target switches, escape actions that continue after a failed switch, and sensing-aware geometry. That complexity currently makes it a poorer first investment than the wall trap.

## Third simple strategy: a stationary wall bait

Place one agent at the middle of the far face of a long, thin rectangular obstacle, while a predator approaches the opposite face. Keep their separation inside the predator's hearing radius. The predator keeps selecting the bait through the wall, but its local collision response oscillates along the near face instead of planning a path around the obstacle.

In the strongest tested realistic geometry, the obstacle was **30 units thick and 70 or 100 units long**. The bait stood 5 units beyond its far face; the predator started 15 units before its near face, facing towards the bait. For each height and initial lateral offsets 0, 10 and 20:

- All 6 cases retained the predator for the full 60-second probe, with continuous sensing and a minimum bait/predator separation of about 45 units.
- The bait took zero movement actions and spent exactly 60 passive energy: 150 → 90.
- The predator continued tiring and resting against the wall.

Geometry matters. A 30×30 obstacle failed within 0.5–0.6 seconds. A 30×50 obstacle held initially but failed after 38.8–51.7 seconds. At widths 40 or 50, the bait often survived because the predator lost its scent and wandered away; that is **not** successful predator containment.

The effect is not restricted to the exact 30-unit minimum obstacle width. A further 45-case tolerance sweep used height 70, widths 31/32/35/37/38, offsets 0/10/20, and headings −0.3/0/+0.3 radians:

| Wall thickness | Full 60-second holds | Baits alive at 60 seconds |
|---:|---:|---:|
| 31 | 9/9 | 9/9 |
| 32 | 9/9 | 9/9 |
| 35 | 9/9 | 9/9 |
| 37 | 6/9 | 8/9 |
| 38 | 3/9 | 9/9 |

Thus **roughly 30–35 thick and at least 70 long** is the supported starting geometry, rather than an exact equality exploit. This is a tested range, not a proven universal threshold. It still needs safe acquisition on generated maps.

Acquisition also matters. Across 12 further heading/offset perturbations around a 30×70 wall, only 5 retained the predator for the whole minute. All baits survived, but the other 7 predators disengaged. Thus do not confuse an already acquired, stable wall trap with a robust automatic trapping policy.

### Food and reinforcement cost

A young stationary bait costs 1 energy per simulated second. This is much cheaper than continuously walking at baseline speed, which adds another 5 per second. Once the hidden maximum age (randomly 60–120 seconds) is exceeded, old-age drain adds `0.01 × age` **per tick**, about 6–12 energy per second at the threshold.

Without food, a 150-energy stationary wall bait died of energy depletion at:

| Assigned maximum age | Bait lifetime | Predator contained throughout |
|---:|---:|---|
| 60 | 71.9 seconds | Yes |
| 90 | 96.0 seconds | Yes |
| 120 | 122.4 seconds | Yes |

Replace baits or arrange nearby fruit before this. A newborn starts at 75 energy, so its young stationary budget is at most 75 seconds before food; offspring placement is random 10–30 units from its parent, so spawning does not itself guarantee a safe handoff. A bait need not have a speed mutation once positioned. A thin-wall trap next to food is more sustainable, but its availability and safe foraging radius were not established in generated worlds.

Other agents must avoid becoming a closer sensed target. Additional predators can approach the bait from its own side. Both are untested failure modes for full-game integration, not reasons to discard the mechanics lead.

## Single-agent orbit and rear following

Tested ordinary walking controllers using only cached predator `distance`, `angle` and `rel_dir`: tangential orbit with radial correction, and following a point behind the predator. Each had initial separations 25/40/60, initial bearings front/side/rear, and target radii 25/40; total 18 cases per controller.

- Orbit: all 18 were eaten within 0.1–9.7 seconds (mean 1.31 seconds). A turn-rate limit alone does not guarantee safe circling; closing speed and initial placement matter.
- Rear-follow: frontal acquisition generally failed quickly. Successful escape from side/rear avoided contact until energy ran out around 26–27 seconds, but the predator sensed the bait in only roughly 10–39% of ticks in the longest cases. This spends energy following a predator that mostly is not chasing the bait, and is not reliable containment.

These are bounded implementations, not proofs against all strafing or kiting controllers. They provide no reason to prioritize these particular controllers over low-motion wall baiting.

## Recommended next experiment

Integrate **opportunistic thin-wall baiting** with the ordinary foraging baseline: recognize a suitable observed wall, approach its far-side midpoint when a predator is already pursuing, then hold only while observation history confirms the predator remains nearby. Prefer low-value/non-breeding agents as baits; keep a reserve that can take over before old age. Evaluate the full-map survival/score tradeoff and successful acquisition rate before treating this as a competitive strategy.

Do not feed true map coordinates, true predator rest state, or true energy to this policy. Those fields were diagnostic conveniences in this investigation and are unavailable at the public action interface.
