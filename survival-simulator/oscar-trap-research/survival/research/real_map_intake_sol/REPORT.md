# Real-map intake evidence

This harness tests generated native maps with random valid initial bait,
parent, and predator starts. The disclosed setup oracle controls only the
parent before its first ordinary Predator DTO. At that observation the policy's
`oracle_handoff()` clears parent navigation memory, the oracle is retired, and
all later actions depend only on native DTOs, public time, and the immutable
static map. Agent energy is refilled; predator energy and rest remain native.

## Scoring contract

A guided delivery requires all of the following for the same tracked predator:

1. the parent receives an ordinary native Predator DTO;
2. native reproduction produces a child after that encounter;
3. the bait remains within 6 units of the depth-5 goal for one second;
4. after every lineage guide-role change, the predator freshly pursues the
   declared active guide under the native chase gate;
5. the predator remains within 75 units of the mouth, no more than 10 units
   outside it, and resting or targeting the bait for two seconds;
6. no physical or target loss occurs, and the predator remains contained for
   the final 30 seconds.

The evaluator never returns tracked predator identity, target choice, position,
energy, or rest state to the policy. Native target reconstruction is scoring
only.

## Frozen v13 results

Policy SHA-256:
`4eaa7ad3fc2be8ac38fe556e00b9d32debd135e103721a54179a278f512cf33f`

Harness SHA-256:
`bf52f369d60d09cee01c05f68aa0d322fff5b3cc2fe9a8ccd9b40e0c075dfd64`

| Map / fixture | Bait | Encounter / birth | Active guide | Fresh pursuit | Agents at 60 s | Entry / delivery | Final mouth distance |
|---|---:|---:|---|---:|---|---|---:|
| 5101 / 9203 | 1.2 s | 4.4 s | child from 4.8 s | 4.9 s | all alive | 0 / 0 | 747.3 |
| 10000 / 9204 | 5.3 s | 6.1 s | parent | 6.2 s | all alive | 0 / 0 | 1060.4 |

Both runs have 601 contiguous native-rendered frames from 0.0 through 60.0
seconds. On map 5101 pursuit was revoked at 15.9 seconds. On held-out map 10000
it was revoked at 15.8 seconds. In both cases the tracked predator then spent
long intervals with no target and never entered the bait site. The candidate
therefore fails at post-rest or lost-visibility reacquisition. These tests do
not establish real-map delivery.

Receipts and replays:

- `results/real_map_intake_sol/real-map-RealMapGuidePolicy-m5101-f9203-p1-oracle_approach_child_guide-sitebait0-52e09db5.json`
- `results/real_map_intake_sol/replays/real-map-RealMapGuidePolicy-m5101-f9203-p1-oracle_approach_child_guide-sitebait0-52e09db5.json.gz`
- `results/real_map_intake_sol/real-map-RealMapGuidePolicy-m10000-f9204-p1-oracle_approach_child_guide-sitebait0-92b2e1dd.json`
- `results/real_map_intake_sol/replays/real-map-RealMapGuidePolicy-m10000-f9204-p1-oracle_approach_child_guide-sitebait0-92b2e1dd.json.gz`

Reproduction command, changing the seeds for the held-out case:

```bash
/home/Ucals/projects/NordicCupAI/.venv/bin/python -u \
  survival/research/real_map_intake_sol/run.py \
  --policy real_map_guide_sol.policy_v13_safe_moving_gaze_frozen:RealMapGuidePolicy \
  --map-seed 5101 --fixture-seed 9203 --seconds 60 --predators 1 \
  --native-width 480 --mode oracle_approach_child_guide
```

The harness writes the receipt only after saving its replay, then performs a
best-effort refresh of `research/real_map_visualization/manifest.json`.

## Frozen v14 follow-up

Policy SHA-256:
`9c19c8be29677b9a54b7bc302145d95ff82cdb17a2f4554da8e4d78ed9aebee3`

The same map 5101 / fixture 9203 pilot again deployed the bait at 1.2 seconds
and produced the native child at 4.4 seconds. Parent pursuit was revoked at
14.4 seconds. The policy reassigned the guide role to the child at 20.0
seconds, but the evaluator found no fresh pursuit of that child afterward.
All agents survived through 60 seconds; physical entry and delivery were both
zero, and the predator ended 1520.9 units from the mouth. The moving rest
orbit did not restore pursuit, and the later role handoff was not valid for
delivery.

- `results/real_map_intake_sol/real-map-RealMapGuidePolicy-m5101-f9203-p1-oracle_approach_child_guide-sitebait0-3155d613.json`
- `results/real_map_intake_sol/replays/real-map-RealMapGuidePolicy-m5101-f9203-p1-oracle_approach_child_guide-sitebait0-3155d613.json.gz`

## Frozen v15 final follow-up

Policy SHA-256:
`4455972445d877117433907a9fb855a9112520fc0ee86c3d9afd0bbc7a72e8ab`

The v15 same-seed run kept the parent as guide and attempted a 54-58 unit
hearing-range orbit. The physical replay shows that local geometry prevented
the requested orbit: guide-predator distance was 69.2 at the first wake (16.2
seconds), 73.6 at the second wake (36.6), and 74.1 at the third wake (52.7).
The guide remained pinned near `[1490,1125]` while the predator remained near
`[1557,1156]`, outside the native hearing radius of 60. Pursuit was not
restored. All agents survived to 60 seconds, but physical entry and delivery
were zero and final mouth distance was 1520.9.

- `results/real_map_intake_sol/real-map-RealMapGuidePolicy-m5101-f9203-p1-oracle_approach_child_guide-sitebait0-922a9679.json`
- `results/real_map_intake_sol/replays/real-map-RealMapGuidePolicy-m5101-f9203-p1-oracle_approach_child_guide-sitebait0-922a9679.json.gz`
