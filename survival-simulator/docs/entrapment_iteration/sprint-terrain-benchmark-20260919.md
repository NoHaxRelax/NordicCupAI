# Sprint survival benchmark

Target: **zero premature predator captures with sprint available**, across
all roles. Intentional delivery sacrifices are separate. Slower mutated
agents still count as failures; a speed diagnostic must not excuse them.

`fast_entrapment_game.py` records every such capture, its action, energy and
last three seconds of observed biome labels and predator distances. The
new `scripts/sprint_benchmark.py` adds same-terrain speed advantage, current
physical sprint distance, whether full sprint was requested, and the timing
and observed gaps around any recent biome change. It can also audit existing
replays without rerunning a game or modifying their original manifests.
The viewer's failure events now expose these details.

The engine applies the starting biome's movement multiplier. For a default
agent, forest sprint is 20 units/tick but river sprint is 6; a predator still
on forest can move 15. Offspring can also have sprint speed below 15 before
terrain penalties. Availability is therefore a useful failure trigger, not
proof that a safe move exists at the final tick. Earlier positioning matters.

Guides already sample river slowdown beginning on either of the next two
ticks in their three-tick search, using observed biome samples only. This
does **not** establish safety for a long river crossing or reveal exact
unobserved borders. Non-guide avoidance currently evaluates the current
terrain and does not have that future-terrain search. Both limitations remain.

The saved audit compares five full native replays of seed 1883894846:

| Version | Premature sprint-available captures |
| --- | ---: |
| Before escape-priority fix | 3 |
| Corrected escape | 0 |
| Optional terrain-based bait ETA | 1 |
| 44-unit delivery handoff | 0 |
| Plus isolated-target tracking | 0 |

None of those four failure cases had an observed slowdown in the retained
three-second window. The three original failures have separately verified
native counterfactual escape actions. The optional bait-ETA failure was a
guide with sprint 10.56 in swamp, physical distance 5.28 per tick. The two
latest runs had zero failures but ended at 791.6 seconds with bait gaps;
this is **not** success across maps or for a full 3000-second game.

Exact audit: `sprint-terrain-audit-20260919.json`. No additional paid compute.

The ongoing frozen-source free-PC comparison has since completed two current-
setting maps: **204871 had 8 failures** (score 726.20, life 679.8 s), and
**730951 had 6** (score 1025.18, life 964.0 s). Of those 14, nine lacked a
same-terrain sprint speed advantage. One had a slowdown recorded in the
preceding three seconds. All 14 still count; no general survival guarantee
has been achieved. These runs use the earlier 55-unit handoff and corrected
escape, before the isolated-tracking change. Exact case diagnostics are in
`sprint-terrain-pc-partial-20260919.json`; remaining comparison jobs continue.

The slowdown case was guide 160 on seed 204871: repeated forest/swamp
transitions from 545.9 to 547.4 seconds accompanied an observed gap shrinking
from 115 to 43 units; at the fatal 547.9-second tick it saw a predator 24.72
units away and requested walking despite sprint availability. This is a
concrete failure to investigate, not proof that a particular crossing alone
caused capture.

A local attempt to reproduce PC seed 730951 diverged despite identical saved
policy and C++ source hashes; even bait energy at 15.1 seconds differed at
floating-point precision. It did not reproduce the original guide-36 death
and must not be presented as that replay. A same-machine replay is being
recorded on PC. New manifests record numerical library versions, platform,
thread configuration and engine binary hash to help track reproducibility.
