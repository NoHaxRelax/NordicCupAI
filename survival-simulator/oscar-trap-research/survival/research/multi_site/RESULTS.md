# Multi-site handover

The corrected static-ID multi-site policy passed **5/8** fresh maps under the
strict any-bait causal protocol. All eight runs reached 300 seconds and all
eight replays verified at 3,001/3,001 native frames. The passes were 10301,
10303, 10304, 10305, and 10306. The fixed denominator includes all maps; there
were no unsupported maps or action-contract errors.

The experiment used 38 predeployed bait agents across eight maps: respectively
1, 7, 4, 5, 3, 4, 7, and 7, for a mean of **4.75 baits plus one guide per
map**. It used no births. The earlier 10138 pilots `47fb1383` and `420bbb72`
used hidden fixture coordinates to reorder bait IDs. Correction sidecars mark
both invalid, and neither is included here.

## Failure classification

All three failures are initial encounter safety failures. None reached a guide
to bait handoff, so trap choice, final approach, and follower reacquisition
were not the first failure.

- **10300:** The guide observed and was chased by the predator from 0.1 seconds,
  alternated observation-only escape and velocity-spacing actions, encountered
  the terrain-barrier rule at 1.4 seconds, and died at 2.5 seconds. Its last
  separation was about 19.5 units. This map exposed only one static bait site;
  the predator finished 903 units from its mouth.
- **10302:** The guide was observed and chased from 0.1 seconds and died at 6.0
  seconds. It maintained roughly 35 units for several seconds, then oscillated
  among escape-lookahead and spacing actions near the selected region. The
  terrain-barrier rule appeared at 5.5 seconds and the slow-terrain lead rule at
  5.8 seconds; separation fell to about 25 units immediately before death. The
  predator later found bait 2 naturally, but the first bait target was at
  117.3 seconds, long after guide death, so the strict causal chain correctly
  rejected it.
- **10307:** The guide was observed and chased from 0.1 seconds and died at 8.5
  seconds. It repeatedly switched among spacing, escape-lookahead, terrain
  barrier, and slow-terrain lead actions. The predator later began targeting
  bait 7 at 54.3 seconds and ended physically inside that site, but the
  45.8-second gap after guide death makes this an incidental capture rather
  than delivery.

The guide survived to 300 seconds on none of the three failures. The next
concrete fix should target the first ten seconds of engagement: use a
DTO-triggered emergency escape/reserve path when the terrain-aware escape cone
is blocked, while keeping one active guide so multiple guides do not cause
target chatter. Adding more sites or changing final intake geometry is not
supported by these failures because every guide died before committing a
causal bait handoff.

Strict scores and costs are recorded in
`results/multi_site/fresh8/strict-scores/SUMMARY.json` and
`results/multi_site/fresh8/cost-summary.json`. For comparison, frozen Burst
Guides finishes at **6/8 under v4**; the earlier 1/8 value was an obsolete v3
role-attribution metric and should not be used as the Burst headline.
