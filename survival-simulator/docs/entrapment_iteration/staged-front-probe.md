# Staged front approach experiment

Isolated 12-case paired check against `track-rejoin-12`. Production files were
not edited. The frozen variant first routes to `mouth - 125 * inward` with the
existing handoff's lateral offset, excludes a 100-unit circle around bait, then
clears route, lane-rejoin, recovery, and steering caches before using the
ordinary handoff and 55-unit stopping rule. Inputs remain ordinary local bait,
mouth, handoff, map edges, and observations.

## Result

- Variant: 10/12 delivery passes.
- Paired baseline: 9/12 delivery passes.
- Paired changes: two wins, one loss, eight unchanged passes, one unchanged
  failure.
- No runtime errors occurred. Two local workers were used.

| Index | Seed | Baseline | Staged | Baseline duration | Staged duration |
|---:|---:|---|---|---:|---:|
| 0 | 412572006 | delivery/retention failed | pass | 60.0 | 33.0 |
| 1 | 424736271 | pass | pass | 30.6 | 32.2 |
| 2 | 780566658 | pass | pass | 20.7 | 28.4 |
| 3 | 882536520 | initial escaped | initial escaped | 49.8 | 50.3 |
| 4 | 1152318527 | initial escaped | pass | 23.1 | 12.7 |
| 5 | 155568418 | pass | pass | 28.8 | 28.1 |
| 6 | 1743629318 | pass | pass | 34.6 | 30.1 |
| 7 | 148874829 | pass | pass | 21.3 | 22.0 |
| 8 | 2028602704 | pass | pass | 57.0 | 57.0 |
| 9 | 2050567370 | pass | initial escaped | 18.9 | 21.5 |
| 10 | 1938596677 | pass | pass | 15.8 | 15.8 |
| 11 | 1640564005 | pass | pass | 20.5 | 21.4 |

The new loss on index 9 is a strict retention failure. Original predator 21
left the held set at 21.0 seconds, while the guide was stopped 4.46 units from
the handoff and 43.97 units from bait. It returned by 22.0 seconds, but the
success rule requires all original predators to remain held. Compared with the
baseline, the staged route extended this delivery from 18.9 to 21.5 seconds
and changed the final approach enough to perturb the existing group briefly.

This sample provides a weak positive signal, not promotion evidence. The net
change is one case, the same case set was previously used for development, and
the variant introduces a measurable retention regression. A fresh, larger
paired evaluation would be needed before considering integration.
