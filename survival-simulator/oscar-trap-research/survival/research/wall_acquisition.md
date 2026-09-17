# Wall traps: acquisition, map availability and replacement

17 September 2026. These follow-up experiments use the unchanged upstream engine at `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`. All tests are local; no competition API calls or scores were submitted.

**Holding an acquired wall trap transfers well to actual map geometry. Safe acquisition is possible during suitable resting windows, but remains unresolved for arbitrary awake encounters. A depleted guide can establish a trap by being captured, and a legal newborn can replace its stationary holder.** The [orchard follow-up](wall_orchard.md) also sustained several acquired traps for 300 seconds with repeated legal generations, under explicitly privileged and arranged conditions.

## What the tests do and do not establish

Controllers in this report know the relevant wall and creature coordinates. This measures physical and resource feasibility, not discovery/localization from ordinary observations. The synthetic fixtures have physical boundary obstacles, one predator starting awake with 102 energy, young 150 energy bait, and no food or additional predator/tree spawns unless explicitly stated. A success requires 60 seconds alive, with the predator both contained and still detecting bait for over 90% of the final 20 seconds. Survival after the predator wanders away is a failure.

The retained files contain 613 simulated cases: 128 solo-acquisition cases, 180 two-agent handoff cases, 48 sacrifice cases, 96 replacement cases, 108 resting-window acquisition cases and 53 valid generated-map placements. Parameter-grid cases share much of their setup and are not independent statistical trials. Exploratory controller revisions were used for tuning; this is not a pre-registered evaluation.

## 1. Running around a wall does not reliably acquire a trap

The solo controller routes around a rectangle with 7 units clearance to the opposite face, then stops. The 96 training cases vary four approach families, two lateral offsets, two starting gaps, energy 150/500, and walking/minimal-sprint/full-sprint escape. These are one legal action per agent per tick.

Only 12/96 held the predator for 60 seconds. Every success started with the bait already on the far side. None of the tested approaches that had to run around the wall produced a lasting trap. Holding still out of sight often made the predator wander past the site. Six policy/energy variants repeat the same two successful geometries, so 12 successes should not be interpreted as 12 distinct acquisition solutions.

A separate 32-case geometry/heading test of the sprint controller produced 0 holds. More sprint energy did not fix losing the predator's attention. These results reject this routing controller, not all possible wall acquisition.

Evidence: `results/wall-acquisition.json`; script `research/wall_acquisition.py`.

## 2. A guide can transfer pursuit, but the tested successful transfers sacrifice it

A second controller starts a holder behind the wall and a guide on the predator's side. The guide approaches the near face, waits until the predator is close, then tries to sprint along the wall. We tested trigger distances, starting gaps, energies, escape directions, and either a stationary holder or one tracking the predator along the far face.

- 28/144 training runs established 60-second containment.
- 7/36 new width/heading/offset combinations did so with selected 150-energy guides.
- **Every successful transfer lost the guide. No safe, sustained handoff was demonstrated.**

Moving the hidden holder continuously to match the predator did not help in this tested implementation. Peeling the guide away earlier tended to take the predator around the wall, outside the hidden bait's hearing range. Peeling late often allowed capture, after which the predator selected the bait across the wall.

The representative replay captures the guide at 0.9 seconds and retains the stationary holder through 60 seconds. The guide still had 132.25 energy when captured, feeding the predator; this is an expensive acquisition, not a safe escape.

Evidence: `results/wall-handoff.json`, native-rendered `results/wall-handoff-replay.json.gz`; script `research/wall_handoff.py`.

### Use a depleted guide rather than spend a healthy worker

We then explicitly tested a guide that waits to be captured at the near face. This is a different objective from attempting safe escape. Across widths 30/32/35/38, headings±0.3 and lateral offsets±15:

| Guide starting energy |60-second holds | Guide losses |
|---:|---:|---:|
|20 |12/16 |16/16 |
|75 |12/16 |16/16 |
|150 |12/16 |16/16 |

All failures were at width 38; all 12 tested combinations with widths 30–35 held at each energy.20 energy is enough for the short walking approach and avoids feeding a large reserve to the predator. This is conditional evidence for using an expendable, depleted worker. It still assumes an already-positioned hidden holder and a suitably aligned pursuit. It does not show that sacrificing a worker improves total game score.

Evidence: `results/wall-sacrifice.json`; script `research/wall_sacrifice.py`.

## 3. Suitable walls exist on generated maps

Seeds 1–9 and 42 produced 46 candidate walls among 800 generated obstacles. The filter was thickness at most 35 and length at least 70, in either orientation. Every sampled map had 2–7 candidates.

We tried both sides of each candidate.39 of 92 placements were rejected because another obstacle or the physical boundary occupied the required creature position. The remaining 53 placements used the original generated terrain, obstacles, initial trees and fruit. Agents and one awake predator were deliberately positioned across the wall. Native fruit growth, collection, tree fruit production and tree death continued; new predator/tree spawning was disabled to isolate the hold.

- 52/53 valid placements held for 60 seconds with continuing predator attention.
- All 53 baits survived.
- The one failed hold lost the predator's attention rather than losing the bait.
- 28/53 placements had a tree within 150 units of the bait.
- 30/53 had initial fruit within 100 units.

These included grassland, forest, swamp, desert, river and mixed terrain. The result is **an acquired-trap test on extracted real-map sites**, not 52 successful traps discovered during full games. Nearby food is not automatically reachable while holding a predator. Most stationary baits still finished with 90 energy; only two collected a fresh fruit in place.

Evidence: `results/wall-map-sites.json`; script `research/wall_map_sites.py`.

## 4. A legal newborn can replace a bait

One 150-energy stationary holder was permitted one actual native birth at 40,45 or 48 seconds. The parent paid 100 energy; the child received its normal 75 and native random traits/position. The child started acting on the following tick and walked to the holding position. There were no injected agents, energy or fruits. The parent stayed in place until depletion.

Eight birth-placement seeds and three assigned founder senescence thresholds were compared with no birth, for 96 runs. All replacements preserved the trap; no parent or child was captured. Attention and containment remained above 99% until final depletion.

| Founder senescence threshold | No replacement | Birth at 48 seconds, mean final survival |
|---:|---:|---:|
|60 seconds |71.9 s |120.9 s |
|90 seconds |96.0 s |120.9 s |
|120 seconds |122.4 s |120.9 s |

The 48-second replacement runs ranged 114.6–123.1 seconds because children's own random senescence thresholds differ. Earlier births gave shorter mean final survival:114.9 seconds at 40,118.6 at 45. Reproduction avoids some senescence cost but converts 100 parent energy into 75 child energy and briefly supports two agents. It slightly reduces lifetime for a founder blessed with the longest old-age threshold. That threshold is not observed, so this is a policy trade-off.

This is **one affordable replacement**, not indefinite reproduction. The unfed child never reaches the 100 energy birth requirement. Sustained generations need food and safe rotation, which the orchard follow-up investigates.

Evidence: `results/wall-replacement.json`; script `research/wall_replacement.py`.

## 5. A resting predator offers a non-sacrificial acquisition window

The actual engine starts new predators resting at zero energy. They recover toward their wake threshold before chasing. If a resting predator is already beside a suitable wall, the bait can walk around it and hide before it wakes. Contact with a sleeping predator does not kill the bait.

We tested two bait approach positions, eight predator headings, three points in the resting cycle, and walking versus sprinting. The predator started 12 units outside the near face; the bait had 150 energy. Its geometry and resting opportunity were arranged, not found by an observation-based detector.

| Predator energy at start of rest test | Walking: lasting holds | Sprinting: lasting holds |
|---:|---:|---:|
| 0, fresh rest, about 3.4 s available | 10/16 | 10/16 |
| 45, about 1.9 s available | 10/16 | 10/16 |
| 75, about 0.9 s available | 5/16 | 10/16 |

Every bait survived the fresh and middle-rest cases, including failures where the predator wandered away. Two late-rest baits were captured in each movement group. Near-forward predator headings produced the useful holds; predators initially facing far away could leave hearing range before turning back. Sprinting improved the short-window acquisition rate but did not cure heading sensitivity.

A further 12 walking tests with new wall dimensions and headings held 9/12. All eight at widths 30/35 succeeded; only one of four at width 38 did. These successful acquisitions required no sacrificed guide. They remain opportunistic: the predator must rest close enough to an appropriate wall, and the actual remaining rest time and energy are hidden. The test controller does not yet detect or arrange that opportunity during a full game.

Evidence: `results/wall-rest-acquisition.json`; script `research/wall_rest_acquisition.py`.

## Reproduction commands

```sh
survival/.venv/bin/python survival/research/wall_acquisition.py
survival/.venv/bin/python survival/research/wall_handoff.py
survival/.venv/bin/python survival/research/wall_sacrifice.py
survival/.venv/bin/python survival/research/wall_map_sites.py
survival/.venv/bin/python survival/research/wall_replacement.py
survival/.venv/bin/python survival/research/wall_rest_acquisition.py
```

The next practical requirements are finding and coordinating around a wall through observations, acquiring it without an expensive sacrifice, supplying repeated replacements, and measuring worker protection and score with additional predators still spawning. The high acquired-hold rate should not conceal those gaps.
