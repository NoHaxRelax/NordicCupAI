# Frozen v15 development regressions

The frozen `policy_v15_direct_escape.py` was run for 180 seconds on four
previously failing development fixtures. Each run used the native simulator,
the v2 streaming recorder, a 320-pixel native render, every-frame recording,
infinite guide energy, and the adjacent-awake arranged start. The policy was
not changed between cases.

| Map / fixture | Result | Entry | Delivery | Guide death | Max distance after delivery | Final distance | Losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10012 / 9412 | pass | 10.0 | 10.0 | 9.9 | 22.3348 | 19.2457 | 0 |
| 10014 / 9414 | pass | 56.4 | 57.2 | 1.9 | 35.7641 | 17.2151 | 0 |
| 10015 / 9415 | pass | 11.3 | 11.3 | 11.2 | 20.0146 | 13.4372 | 0 |
| 10016 / 9416 | pass | 9.6 | 9.6 | 9.5 | 22.1871 | 12.4494 | 0 |

This is a 4/4 fitted harness regression check, not an independent reliability
estimate. Only 3/4 are attributable to the attempted guide route. In map 10014
the guide died at 1.9 seconds, 1,256.7 units from the mouth, while the predator
was 1,248.4 units from it. The predator traversed the map without a guide and
entered 54.5 seconds later. The result is therefore an autonomous post-guide
capture rather than evidence of successful guided delivery. The preserved
audit sidecar records this correction without modifying the original receipt.

The fresh frozen-v15 successes available at audit time did not show the same
long-delay pattern. Five guide deaths preceded delivery by 0.2 seconds, one
(map 10035) preceded it by 2.0 seconds, and three guides survived through
delivery. Map 10035 is a real handoff despite its longer arrival delay. At its
last alive frame the guide and predator were 184.55 and 168.30 units from the
mouth. After the guide died at 12.4, the target trace became temporarily null
and then selected bait agent 0 at 13.2, before physical arrival at 14.4.

Receipts and full native replays are in
`results/short_overlap_sol/v15_regression/` and its `replays/` directory.

## Fresh map 10032 diagnosis

The separate frozen-v15 fresh case 10032 / 9632 failed after the guide was
already being followed. The guide died at 8.9 seconds without delivery.
During the final emergency sequence it alternated between exactly
`(1272.4839, 964.7980)` and `(1272.4839, 980.7980)`. The predator-to-guide
distance fell from 38.69 at 8.6 seconds to 22.29 at 8.7 and 22.07 at 8.8.

All three actions used `maximize_observed_escape_distance`. The selected move
did execute; this was not a native collision or stationary-action failure.
The emergency selector independently maximizes distance from the latest stale
observed predator position on every tick. In a locally constrained corridor,
the winning endpoint can flip to the opposite side on the next observation,
so the guide reverses its previous escape and gives up its separation.

A structural repair should remember the committed escape direction and reject
an immediate reversal while danger remains. It should require a candidate to
improve separation from the currently observed predator; when no candidate
does, it should continue along one collision-clear direction for a bounded
number of ticks instead of alternating between local maxima. This diagnosis
does not modify or relabel the frozen v15 batch.

## Bounded emergency commitment probe

`policy_v16_emergency_commit.py` changes only v15's emergency escape. It
commits to the selected world-space direction for four emergency decisions.
Every decision projects a new endpoint from the current localized pose and
requires a fresh static-clearance check; a blocked direction is dropped
immediately.

On the exact failed case 10032 / 9632, the fixed policy completed the 180-second
horizon, kept the guide alive, entered and delivered at 17.1 seconds, and held
the predator with zero physical or target losses. Maximum distance after
delivery was 14.6303 and final distance was 11.1881. All 1,801 native frames
are retained in `results/short_overlap_sol/v16_emergency_probe/`.

This is one fitted repair case. It proves bounded direction commitment removes
the observed two-point reversal on this fixture, but it does not measure
general reliability or interactions with other emergency geometries.

Four other predeclared fresh-v15 emergency-death fixtures were then rerun with
the frozen v16 policy:

| Map / fixture | Result | Guide death | Delivery |
| --- | --- | ---: | ---: |
| 10038 / 9638 | fail | 3.7 | none |
| 10040 / 9640 | fail | 3.3 | none |
| 10043 / 9643 | fail | 0.7 | none |
| 10045 / 9645 | fail | 2.0 | none |

Thus v16 repairs 1/5 fitted v15 emergency-death cases when the original 10032
case is included. The other four completed the 180-second horizon as preserved
failures with no entry or delivery. Direction commitment addresses the exact
two-point reversal but does not solve the broader emergency-death mechanism.
