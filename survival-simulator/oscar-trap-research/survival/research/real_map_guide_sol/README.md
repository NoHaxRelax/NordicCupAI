# Real-map encounter guide policy

`RealMapGuidePolicy` uses an immutable obstacle map to choose an 11--19 unit
refuge gap and route a randomly spawned bait agent to a fixed depth of 5. At
runtime it receives only native per-agent DTOs and public simulation time. It
does not receive predator IDs, positions, targets, rest state, or energy.

The encounter fixture may route the random parent toward the tracked predator
only until the first native `Predator` DTO. At that instant it calls
`oracle_handoff()`, which clears parent navigation memory without importing the
fixture position. All later localization, pursuit verification, parent/child
guide assignment, and movement use ordinary DTOs. The parent requests one
native child at the encounter. Both lineage agents remain in the world.

## Current protocol

The frozen current candidate is
`policy_v15_hearing_reacquisition_frozen.py`, SHA-256
`4455972445d877117433907a9fb855a9112520fc0ee86c3d9afd0bbc7a72e8ab`.

- The bait follows a radius-5 visibility-graph route to `mouth + inward * 5`.
  Arrival requires fresh native edge consistency with its prior
  observation-derived pose.
- The active lineage guide follows radius-11 static-map routes, preserving
  predator clearance. A route corner is discarded only when reached or when a
  direct map-clear shortcut exists. Every issued movement segment is checked
  against the static map before updating odometry.
- Following is confirmed from repeated native observations: predator facing
  plus reconstructed predator motion toward the guide. Confirmation is
  revoked when the evidence disappears. While unconfirmed, the guide stays in
  a moving 54--58 unit orbit around the last native predator point and does not
  advance toward the refuge until pursuit is reconfirmed.
- Every fourth engaged tick turns toward the last observed predator while
  continuing to move. Intervening visible ticks look beyond perpendicular.
- Predator DTO positions precede one native predator move. Spacing allows for
  that hidden move and the next possible move. The evasion selector uses an
  achievable absolute safety floor; reserve lineage agents also flee
  immediately below 65 observed units.
- Guide role changes require four consecutive facing/closest observations and
  a one-second cooldown. The switch decision never uses the evaluator's hidden
  target.
- At the refuge, the guide stages 125 units outside, waits for axial alignment,
  leads to 75 units outside, and holds for sacrifice.

Maps without a qualifying site raise an explicit unsupported-map result. A
separate census found eligible conservative sites on 53/128 generated maps
with overlap at least 55; overlap at least 20 yielded candidates on 127/128.
The smaller-overlap geometry has not been validated for guided delivery.

## Recorded evidence

All listed qualifying runs use native rendering, one replay frame per tick,
random bait deployment, and a first-native-observation oracle cutoff. Replay
and receipt names are under `survival/results/real_map_intake_sol/`.

| Frozen policy | SHA-256 prefix | Result on map 5101 / fixture 9203, 60 s |
|---|---:|---|
| v6 | `9c7961d2` | Bait deployed at 1.2 s; parent and child died; no entry. |
| v7 | `267db0fc` | Parent advanced about 900 units, then died at 29.1 s; no entry. |
| v8 | `95e65f2e` | Parent died at 7.0 s from stale-position spacing; no entry. |
| v9 | `b7e14e35` | Parent survived; child diverted predator and died at 6.8 s; no entry. |
| v10 | `784b23c1` | Fixed child guide survived, but pursuit returned to parent; all alive, final predator 963.8 from mouth; no entry. |
| v12 | `9a25bb27` | Dynamic guide roles kept all agents alive and improved final distance to 651.8, but thrashed roles early and lost pursuit; no entry. |
| v13 | `4eaa7ad3` | Stable child role; all agents alive; predator rested and the guide abandoned reacquisition, final distance 747.3; no entry. Held-out map 10000 reproduced the same failure (1060.4). |
| v14 | `9c19c8be` | Parent survived but its 68--76 orbit lay outside predator hearing 60; predator woke and wandered away. A later role switch was false; final distance 1520.9; no entry. |
| v15 | `44559724` | Parent remained guide and all agents survived, but an obstacle pinned the physical standoff at 69--74 instead of the commanded 54--58. Predator woke outside hearing and wandered away; final distance 1520.9; no entry. Replay suffix `922a9679`. |

The v9 no-birth control replay (`...-7d5ea2d7.json.gz`) isolated a routing
failure: a corner was popped while still six units away, collision deflection
left dead reckoning inside an expanded wall, and all later evasion candidates
were rejected. V13 directly fixes that route/odometry defect as well as the
previously impossible requirement that a single evade step increase spacing by
25 units. A v13 no-birth replay then isolated a second corner oscillation;
v14 lowers the absolute endpoint floor to the achievable two-move latency
bound with margin and retains the prior escape heading as a tie-breaker.

The early v1 120-second result `...-ec135f67.json.gz` is state-only and has no
native images. Native-render reproductions were recorded separately. It must
not be cited as native footage.

No completed run in this directory has yet demonstrated real-map delivery.
The frozen versions and negative replays are retained to make that limitation
and each behavioral change reproducible.

## Recommended next experiment

Keep v15's hearing-range standoff and role filter, but replace the greedy local
step toward the remembered predator with a radius-5 static-map visibility-graph
path to a 54--58 unit standoff point. The v15 replay shows the guide's command
intent was correct while physical movement remained on the opposite side of a
corner: at the first wake (16.2 s) separation was 69.2, and at later wakes it
remained about 73.6--74.1. Replan after each scheduled native sighting and do
not resume the refuge route until native predator motion toward the guide is
reconfirmed. One same-seed 60-second native-render run on map 5101 / fixture
9203 should establish whether obstacle-aware reacquisition fixes this specific
failure before any broader sweep.
