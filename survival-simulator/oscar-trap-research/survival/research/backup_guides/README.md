# Bounded backup guides

`BackupGuides` wraps independent `policy_v24_short_fallback` instances. The
original guide requests one native child on its first Predator DTO, then may
request another while its closest observed predator is under 50 units and at
least two seconds have elapsed. Requests stop after three, giving at most four
guides including the parent. Native birth position, traits, mutation, energy
cost, and rejection remain unchanged.

Every living non-bait agent owns separate localization, route, phase, pursuit,
and escape state. Controllers share only the immutable static map. The wrapper
reports an active guide using minimum Predator distance from public DTOs with a
five-unit hysteresis; this reporting choice never changes any guide action.
The bait remains stationary.

This strategy has a separate denominator from all single-guide results. Birth
requests and surviving guides are explicit costs, and a success cannot support
a single-guide reliability claim.

## Map 10138 result

The single authorized 300-second run requested one birth at 0.1 seconds. Child
2 appeared at its native random parent-proximity position and registered at
0.2 seconds. The predator switched between parent and child, then ate both at
1.0 seconds. The parent therefore never reached the two-second interval needed
to request another reserve. The harness later counted delivery at 93.9 seconds,
but the standard active-bait audit classifies it as a distant autonomous
capture. This is a redundancy failure, regardless of the original receipt's
success field. Map 10139 was not run.

Cost for this failed case was one requested and successful native birth, two
guides dead, and no attributable delivery. The replay retains all 3,001 native
frames.

`mutation_aware.py` records an untested follow-up for mutated children. v24's
ordinary spacing search tries synthetic caps of 3, 6, 10, 15, and 20; the
adapter clamps its endpoint projections to the child's actual DTO speed and
sprint speed. It does not alter native traits. This was not causal in the
observed one-second double death and was not rerun.
