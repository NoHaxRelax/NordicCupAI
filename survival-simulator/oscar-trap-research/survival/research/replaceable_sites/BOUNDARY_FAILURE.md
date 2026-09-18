# Boundary approach failure c18d1c7b

The guide did not fail near the selected boundary trap. It died at `(455.72,
225.59)`, roughly 950 units from the trap mouth, while routing around ordinary
obstacles 36, 49, and 60 near its arranged encounter.

The first 3.3 replay seconds show the guide in river terrain, so its requested
20-unit sprint produced about 6 units of native motion. The v21 escape planner
selected its explicit stationary action at 2.2, 2.3, 2.9, and 3.2 seconds.
Across those pauses the physical guide-predator separation fell from 35.7 to
21.2 units; the guide was eaten on the following native predator step.

This was a planner choice rather than hidden-state or localization evidence.
Every pause is present in the recorded action and tagged
`observation_only_escape_lookahead`. The policy had fresh Predator DTOs, and
the recorded agent pose agreed with the terrain-limited commanded displacement.

`policy_boundary_no_hold.py` is a fitted diagnostic subclass. It retains v21
and changes only a stationary first escape result during moving pursuit inside
50 units. It chooses the best radius-5.01-clear one-step direction using the
same observation-derived predator point, static map, and public terrain. A
Its recorded 300-second rerun still lost the guide at 3.3 seconds. It removed
the four pauses, but the newly selected steps reversed direction: for example,
the guide moved northeast at 2.2 seconds and southwest at 2.3 seconds. It later
oscillated vertically at the obstacle-49 corner and was eaten. The harness
reported success after the predator independently found bait at 86.7 seconds;
the standard post-hoc audit flags this as a distant autonomous capture.

`policy_boundary_commit.py` records the next bounded hypothesis without a run:
hold the first clear emergency direction for four ticks unless static collision
invalidates it. This follows directly from the fitted reversal, but remains
untested and should not be treated as a repair.
