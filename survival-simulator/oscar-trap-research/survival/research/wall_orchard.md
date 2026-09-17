# Feeding and replacing a wall-trap crew

The local feasibility test can maintain an already acquired wall trap for five minutes with food collection, legal births, and repeated handoffs. The final controller reached the 300-second horizon in three of four runs with normal tree spawning, including both held-out wall/orchard geometries. This is a promising component, not a full-game score result or a proof of indefinite sustainability.

## Setup and privileges

All 24 runs call the vendored engine's ordinary movement, sensing, energy, reproduction, predator, tree, and fruit methods. There is one action per living agent per tick. Every replacement is a native birth costing its parent 100 energy; children start with 75 energy and receive normal random mutations and age limits. There are no fruit or energy injections, free replacement agents, or duplicate actions.

Each run begins with an **already acquired** trap: a predator on the left face of a thin wall, a 150-energy holder seven units beyond the right face, and a second 150-energy agent near the orchard. The two founders' old-age thresholds are fixed to 90 seconds; descendants use native thresholds. The map is a homogeneous forest with physical arena boundaries. Initial native trees start at age zero. New predator spawning is disabled so the experiment measures one held predator's support cost.

This is a **privileged feasibility controller**. It reads true coordinates and fruit energy, which ordinary observations do not expose. It does not solve wall discovery, localization, acquisition, hidden fruit maturity estimation, competing predators, or travel between colonies. Trees are arranged near the protected wall face. Tests marked “new trees on” restore the exact native `spawn_tree` method and global placement rules; they do not place replacement trees in the orchard. Homogeneous forest is more favorable for tree recruitment than many generated mixed-biome maps.

## Controller

The holder stands still at the wall anchor. Foragers walk to fruit on the protected side within a 260-by-360-unit rectangular patch. They normally wait until fruit energy reaches 58, but accept young fruit below 65 personal energy. Incoming holders reach the anchor before the outgoing holder leaves, preserving the closer detectable target. Foragers never deliberately cross the wall or approach closer to the predator than its anchor.

The initial controller counted everyone against a total population cap of three. That was an implementation bottleneck: retired elders occupied birth slots even when only one useful holder remained and fruit was available. A six-run refinement tested caps of three, four, and five with retirement ages of 65 or 85 seconds. Larger total caps prolonged some trials but did not remove the underlying shortage of replacement workers.

The final **active-cohort** variant permits at most three working agents and six total living agents, including retired elders. It allows the holder to reproduce when the working reserve is short, permits breeding from age 25 with more than 170 energy, and gives younger agents more weight when choosing an incoming holder. Old outgoing holders retire from the orchard and die through normal age and energy costs. Well-fed foragers stop deliberately consuming fruit near their energy capacity. This controller still has rough edges: an old forager can remain active too long, a retired agent can survive while a replacement is short of energy, and the bounded cohort can shrink despite remaining food.

## Final controller results

“Hold” means that the predator remains on the original wall face, within 45 units horizontally and within ten units of the wall's vertical extent. “Attention” is the fraction of pre-world-tick sensing checks whose nearest sensed agent is the assigned holder or incoming holder. Newborns temporarily overlapping the anchor can lower this metric while containment remains intact. Survival is the time until extinction or the 300-second cutoff; it is distinct from containment time.

| Set / seed | Wall; initial orchard | New trees | Survival | First hold loss | Hold | Attention | Births / maximum descendant generation | Handoffs | Captures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Training 1 | 32×80; 3 trees, x offset 90 | Off | 169.6 s | 139.9 s | 82.4% | 81.2% | 3 / 2 | 3 | 0 |
| Training 1 | Same | On | 244.4 s | 210.3 s | 86.0% | 85.9% | 4 / 1 | 5 | 1 |
| Training 2 | 32×80; 3 trees, x offset 90 | Off | 154.7 s | 139.9 s | 90.4% | 89.3% | 2 / 1 | 3 | 0 |
| Training 2 | Same | On | **300 s** | None | **100%** | 98.2% | 7 / 3 | 9 | 0 |
| Held out 3 | 30×70; 2 trees, x offset 60 | Off | 159.6 s | 140.8 s | 88.2% | 82.9% | 2 / 1 | 3 | 0 |
| Held out 3 | Same | On | **300 s** | None | **100%** | 98.8% | 11 / 6 | 14 | 0 |
| Held out 4 | 35×100; 3 trees, x offset 120 | Off | 139.1 s | None before extinction | 100% | 100% | 2 / 1 | 2 | 0 |
| Held out 4 | Same | On | **300 s** | None | **100%** | 100% | 10 / 5 | 10 | 0 |

The successful 300-second runs finished with one, three, and five living agents respectively. The training success had only one holder remaining, so its horizon result is especially weak evidence of longer sustainability. Across those three runs, the crews ate 86, 89, and 94 fruits, worth 4,282.8, 4,197.6, and 4,575.6 gross energy. Gross energy includes energy lost to the creature's capacity limit. The only final-controller capture occurred after containment had already failed.

The two held-out sites change geometry and seed together; four trials are insufficient to estimate a success probability or isolate which environmental factor matters most. The final controller was fixed before these held-out cases ran.

## Controls and constraints

In the initial six fixed-orchard runs, using one, two, or three trees and seeds 1–2, the total-cap-three relay maintained the trap until extinction at 101.0–155.9 seconds, with zero captures. Five runs made one replacement child. The seed-1 three-tree static control lasted 99.8 seconds and first lost the wall at 96.4 seconds; the walking relay without births lasted 129.3 seconds with continuous containment. These comparisons support testing food and rotation, but native RNG draws change when births and later population histories differ, so these are not identical realized-food counterfactuals.

Native trees begin producing fruit at age 20 seconds and can die after age 50. The finite initial patches in these runs disappeared around 57–63 seconds. Native fruit matures over about 20 seconds and rots after roughly 50 seconds. A nearby initial orchard therefore provides a temporary stock. Recruitment of new trees, wider foraging, or outside support is needed for longer tests. The observed fixed-patch failures are failures of the tested bounded controllers, not a universal impossibility result.

The next useful engineering work is an observation-based version of the role/handoff controller on generated maps, followed by measured food-patch selection and recovery when a replacement is missing. More synthetic parameter sweeps are not necessary before that step.

## Reproduce

Run from the repository root:

```sh
survival/.venv/bin/python survival/research/wall_orchard.py
survival/.venv/bin/python survival/research/wall_orchard.py --suite refinement --output wall-orchard-refinement.json
survival/.venv/bin/python survival/research/wall_orchard.py --suite cohort --output wall-orchard-cohort.json
survival/.venv/bin/python survival/research/wall_orchard.py --suite heldout --output wall-orchard-heldout.json
```

Results live in `survival/results/wall-orchard*.json` with five-second traces, legal-birth records, deaths, handoffs, meals, food totals, and containment measurements. These are local macOS engine tests, not competition API evaluations. Upstream uses object sets in some interactions, so a fixed seed is not a promise of cross-process or cross-platform bit-for-bit determinism.
