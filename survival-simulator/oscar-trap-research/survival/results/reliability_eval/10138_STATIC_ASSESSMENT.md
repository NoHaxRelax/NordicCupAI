# Map 10138 / fixture 20138 static escape assessment

This is an offline static-map and movement-equation assessment. It did not step
the native simulator or create a replay.

The guide starts at `(1439.8235, 46.3841)` and the predator at
`(1490.6028, 67.5138)`. The map's top boundary is 30 units thick, so a radius-5
guide cannot move its center below `y=35`. The west obstacle's radius-expanded
right edge is `x=1438.548`, leaving about 1.275 units beside the guide, and the
southern obstacle closes the downward route. The usable exit is eastward, where
the predator starts. Both creatures start on river terrain: the guide's native
20-unit sprint moves 6 units and the predator's 15-unit sprint moves 4.5 units.

After the mandatory blind observation tick, separation is about 50.5. A bounded
offline search used the static terrain, native endpoint collision rotation,
native chase turn, 32 guide headings per tick, and 1,000 diverse retained
states. Its best minimum separation fell to 15.5 by tick 12, and it found no
survivor at tick 13. Discretization prevents a mathematical impossibility
claim, but the geometry and search strongly indicate the supplied fixture is a
forced-loss pocket.

A replacement fixture protocol should retain every fixed map/fixture seed in
the denominator while evaluating all 72 radius-55 adjacent angles for a frozen,
disclosed blind-tick escape criterion. It should sample uniformly among eligible
angles with fixture RNG. A seed with no eligible angle remains a setup failure.
This requires a new protocol and cannot revise prior denominators.

Native children should still spawn at their parent. Static staging routes may
be selected after ordinary localization, and fruit locations must come from
ordinary observations. Any pre-positioned third agent belongs to a separate,
explicit fixture rather than a native-child claim.
