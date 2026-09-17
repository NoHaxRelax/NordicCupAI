# Native map availability at bait depth 5

128 fresh native maps, seeds 10000 through 10127, were generated with no
actors and no simulation steps. All obstacle rectangles and candidate mouths
are preserved in `results/real_map_census/map-<seed>.json`.

| Criterion | Maps | Share | 95% Wilson interval |
| --- | ---: | ---: | ---: |
| Current policy: gap 10.9–19.1, overlap at least 54.9, free bait at depth 5, clear far-125 to hold-75 staging | 58/128 | 45.3% | 37.0–53.9% |
| Also clear from hold point to the first blocking wall face | 53/128 | 41.4% | 33.2–50.1% |
| Interior obstacle pairs with a free depth-5 refuge, before staging checks | 74/128 | 57.8% | 49.2–66.0% |
| Refuges including arena-boundary pairs, before staging checks | 118/128 | 92.2% | 86.2–95.7% |

For the currently tested geometry, use **roughly 40–45%** as the conservative
map-availability estimate. This is not a delivery success rate: it does not
establish that a guide can route from every point to a candidate or retain
predators there. Boundary-adjacent narrow corridors often have no usable
mouth/staging area for this strategy, so the 92% refuge count is not an
interchangeable estimate.

An exploratory geometric relaxation allowing overlap as short as 10.1
instead of 54.9 found a clear-approach candidate on 127/128 maps. These
short-overlap configurations have **not** been validated for delivery or
retention, so they are not included in the conservative estimate. Bait depth
remains fixed at 5 in every criterion; no new depth simulation was performed.

The same 127/128 candidate count holds with overlap at least 20. Increasing
the minimum overlap to 30 gives 121/128, and to 40 gives 109/128. Thus the
long-overlap restriction, rather than the fixed bait depth, accounts for much
of the conservative coverage limit. These are geometric candidates only.

The first version of the optional approach-corridor calculation incorrectly
ended on/inside the walls' expanded blocking rectangles. This was corrected
to end just outside the earliest expanded face, accounting for unequal face
offsets. The current-policy 58/128 count was unaffected.

This static census uses complete map information, as authorized. It neither
gives dynamic predator information to a controller nor claims replay footage
for unstepped maps. Re-running uses the preserved exact map geometry.
