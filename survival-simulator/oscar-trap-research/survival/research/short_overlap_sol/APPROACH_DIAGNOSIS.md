# Map 10008 / fixture 9308 approach diagnosis

The v8 failure was not caused by losing the predator at the random start or by
the trap geometry. The guide and predator began following immediately. By
6.0 seconds the guide was 51.6 units from the mouth with the predator 38.6
units away; at 7.1 seconds the predator briefly entered the site's physical
scoring region. The controller nevertheless stayed in its ordinary spacing
mode because its final transition required the guide to come within 8 units
of one exact `far` waypoint, plus a narrow relative-facing test.

Spacing and collision avoidance pushed the guide laterally around that point.
It orbited the entrance, then ran toward the north map boundary while the
predator alternated between chase and native rest. It returned to the site on
the next active cycle but again failed the point transition. The guide died at
52.5 seconds; no predator was delivered. This was a waypoint/state-transition
failure, not a failure to establish pursuit.

## Minimal observation-only change

`approach_lane_v1.py` freezes the v10 lane transition at source hash
`7e6cfad20eea082aa83e66a8cb505fe9ee9d19a5efb6aa601d6877a8bbd2d1e3`.
It retains v9's raw-observed 55-unit safety margin. The only approach change
replaces the 8-unit point requirement with a geometric lane detectable from
the static map and ordinary observations:

- guide 80--150 units outside the mouth;
- guide cross-track error below 15;
- visible predator within 70;
- predator behind the guide, cross-track error below 20, and observed moving.

No fixture coordinates, predator energy/rest state, hidden target, or hidden
position enters the controller.

The recorded map-10008/fixture-9308 rerun passed at 120 seconds. Physical entry
occurred at 55.5 seconds, guide sacrifice and scored delivery at 56.0 seconds,
the predator remained contained for the entire final 30 seconds, and there
were zero physical or target losses. Its final distance from the mouth was
13.5553. The receipt is in `results/short_overlap_sol/approach_probe/` and its
replay contains all 1,201 native frames.

This one repaired case validates the diagnosed transition failure. It does
not establish greater-than-95% approach reliability; the broader held-out
guide batch must remain the reliability denominator.

## Privilege and scoring audit

No runtime controller leak was found. The constructor receives a detached
perfect static map and the bait/guide role IDs. The harness hashes that static
payload before construction and after every policy call. Runtime `act` calls
receive only JSON-cloned native observation DTOs and public simulation time.
Hidden tracked-predator association, coordinates, chosen target, rest state,
and collision outcome are used by the evaluator after the policy action and
are never passed back to the controller.

The experiment still has substantial disclosed setup privileges. The bait is
teleported to the selected depth-5 goal. The guide is placed exactly 55 units
from the predator and faces it; the predator is forced awake at full energy
and faces the guide. Therefore this evaluates approach after immediate pursuit
has already been established. It does not test finding a predator, obtaining
initial pursuit, or deploying the bait.

Scored delivery requires 20 consecutive ticks in a broad physical region
(within 75 of the mouth and no more than 10 inward) while the predator is
resting or selects the bait. That predicate alone is broader than literal
occupancy inside the narrow passage. In this repaired case, the stronger
recorded evidence is consistent with actual capture: maximum mouth distance
after delivery was 17.7756, final distance was 13.5553, and both physical and
contained counts remained one throughout the final 30 seconds.

## Recommended next controller refinements

Keep the lane transition and the raw-observed 55-unit safety floor frozen
while measuring fresh held-out maps. Those two rules address separate risks:
the lane prevents endless orbit around an exact waypoint, while the raw floor
covers observation latency when a predator turns sharply and invalidates its
velocity extrapolation.

If fresh failures still occur before reaching the lane, add one invariant at a
time using ordinary observations:

1. Near the selected site, reject candidate translations that increase
   cross-track distance from the mouth axis unless required by the 55-unit
   safety floor. This prevents another lateral orbit without forcing an unsafe
   straight-line move.
2. During native predator rest, hold at a safe visible spacing. On wake, wait
   for observed motion before resuming axial progress. The v10 moving-follower
   gate already prevents committing on a resting predator.
3. If the Predator DTO disappears, use a bounded sweep around the last
   observed point while facing that point. Resume route progress only after a
   new DTO confirms pursuit. Do not infer current position, rest, or target
   from the evaluator.
4. Continue selecting walking caps (3, 6, or 10) when spacing is stable. Use
   15 or 20 only when the observed raw distance approaches the safety floor or
   when a clear axial final lead has begun.

Each change should be evaluated on fixed fresh map/fixture pairs. Development
regressions and repeated versions of the same fixture must not be counted as
independent evidence toward the greater-than-95% goal.

## Stale-observation reacquisition repair

The frozen v10 held-out case map 10012 / fixture 9412 exposed a separate state
trap. It lost the Predator DTO and native pursuit at 4.4 seconds. Because the
distance to the stale last-observed point remained below 75, the old
reacquisition branch never ran; holding remained the best spacing candidate.
The guide stayed at exactly the same position from roughly 10 seconds through
the 180-second horizon while the predator wandered.

`reacquire_v11.py` adds a bounded observation-only recovery rule. After one
second without a Predator DTO, it walks toward the last observed point. Once
there it sweeps a free 30-unit ring while scanning. A fresh DTO immediately
returns control to the unchanged v10 spacing and lane logic. It never reads
the predator's evaluator position, target, energy, or rest state.

The fitted map-10012 rerun reacquired native pursuit at 11.1 seconds, sacrificed
the guide at 22.7, and recorded physical entry and delivery at 23.0. It passed
the 180-second strict success criterion with zero losses, maximum post-delivery
mouth distance 22.202, final distance 19.4093, and full final-30-second
containment. Source hash:
`403fc240c9ad6110d2326de54c4d0efb5d849d23d1c2fabf4097bcd988ec6083`.

This is a fitted repair, not independent reliability evidence. A closer
hearing-range controller can reduce how often pursuit is lost; bounded v11
recovery remains a compatible fallback, but the combination needs a frozen
source and fresh held-out cases before it contributes to the reliability
denominator.
