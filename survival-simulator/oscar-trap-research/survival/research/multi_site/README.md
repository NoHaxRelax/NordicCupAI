# Multi-site feasibility

`dispersed_sites(enumerate_sites(static_map), limit=8)` removes opposite-mouth
duplicates of the same obstacle corridor and spreads the remaining mouths over
the arena. A setup harness can place one native bait at each site's `goal` and
pass the static `bait_id -> site` mapping to `MultiSiteGuide`.

The guide chooses the nearest site's `far` point once, using only its own
landmark-localized DTO pose. Predator position does not enter selection. Every
bait remains stationary, including baits that may already retain a predator.
The current native harness supports one designated bait, so a valid experiment
still needs a multi-bait setup and scorer that records causal target handoff and
stable containment for any designated bait. Cost is up to eight predeployed
bait agents plus one guide; the selection logic itself adds no births.
