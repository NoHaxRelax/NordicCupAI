# Prepared33 retention and bait replacement on a native map

Run `93d0e1dc` retains all33 original predators for the full3000-second horizon on native map10144. Independent auditing verifies all30001 original native frames, with no physical loss and no gap in bait-goal coverage. The replacement bait survives to the end. This is a prepared near-mouth crowd, not33 randomly encountered predators transported across the map.

The site lies between obstacle11 and native boundary wall1. Gap10.1038, overlap56.4771, bait depth5. Mouth(476.5663,1164.9481), bait goal(481.5663,1164.9481), rear entry(553.0433,1164.9481). The radius5.01 rear-access path is clear in static geometry. A lateral approach offset of−6 permits the predator-sized approach beside the boundary.

The replacement reaches the bait goal by13.1s and the old bait reaches the rear exterior by29.1s (these times are from the one-second receipt trace). The departed old bait later dies at1530.4s; the protected replacement remains alive through3000s. After30s, every awake original predator has a goal occupant as its closest agent inside native hearing range. This is sufficient for native target selection, including through walls. Native predator energy/rest remains enabled; only agent energy is replenished.

The original harness accidentally tracked the mutable `env.predators` list. Native spawning later enlarged it, and the test `held_count == 33` became false when34 predators were held at2067.1s, rising to38 at2999.1s. This was not an escape by the original cohort. The separate audit fixes the tracked IDs from frame0 and checks every frame. Preserve the original receipt and its explicit correction. The future harness now freezes a tuple of the original33; the previous source is preserved in commit56f1c77.

Artifacts, relative to the survival directory:

- Original receipt: `results/replaceable_sites/native-map-10144-expanded-narrow-crowd33-continuous-replacement-93d0e1dc.json`
- Independent audit: same stem with `.retention-audit.json`
- Correction annotation: same stem with `.correction.json`
- Native replay: `results/replaceable_sites/replays/` plus the same stem and `.json.gz`
- Catalog verification: `results/replaceable_sites/full3000-catalog-check.json`
- Original final pixels: `results/visualizer_validation/full33-final-native.png`

Playback: http://127.0.0.1:9055/research/real_map_visualization/index.html?run=93d0e1dc

Reproduce with the installed simulator environment:

```sh
python research/replaceable_sites/run_crowd_handoff.py --seed 10144 --expanded-narrowest --seconds 3000
```

The original run used a capped worker, later raised to6GiB max/5GiB high as native state history grew; peak observed RSS was about4.8GiB. Disk use is severalGiB for the complete replay and frame chunks. The resource watcher enforces system headroom. The native spawn fixture allows overlapping predators, consistent with their absence of mutual collision. This is one continuous replacement on one map, not autonomous repeated replacement or a population-level retention guarantee.
