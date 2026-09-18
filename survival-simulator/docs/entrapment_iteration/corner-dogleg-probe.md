# Corner-arrangement probe

## Result

The useful native geometry is a **long staggered dogleg**, not a zero-depth
diagonal corner aperture. Two rectangles face across a 14--18 unit gap and
their ends overlap by 15 units. The bait sits in the agent-width dogleg. A
radius-5 agent can enter from the rear, while a radius-10 predator cannot enter
or contact the bait. The overlap is longer than one native predator sprint
step, preventing endpoint-only collision tunnelling.

Nine constructed native cases passed (apertures 14, 16, and 18; simulator
seeds 1, 7, and 19). In every case:

- all 30 preloaded predators were held before delivery and the minimum held
  after delivery remained 30;
- one native newcomer settled for at least 10 seconds, with 31 held at end;
- no predator entered the rear replacement region;
- a size-5 replacement walked from the rear to the bait, the old bait was
  exhausted, and the replacement remained alive through the final hold.

The guide and replacement commands use only the fixed fixture bait target.
Predator positions and identities are read only for evaluation. Predators use
the unmodified native movement and sensing code. Each case runs 75 seconds at
the native 0.1-second timestep and retains every tick.

## Important negative finding

A point-like diagonal gate (two facing corners offset by 14 units in both axes,
with no axial overlap) did not retain the group. Native predators move up to 15
units per tick and collision is checked at the proposed endpoint, so they can
tunnel through a zero-depth diagonal aperture. That class should not be added
to site selection based on static radius reachability alone.

The passing dogleg is physically a short narrow channel. The current ordinary
site enumerator already accepts 15 units of overlap (`min_overlap=10.3` in the
observed selector), so this construction does **not** establish additional map
coverage beyond existing crevices. It does establish why a corner generalizer
must measure traversal depth in native-step terms, rather than only checking
that the bait point is predator-invalid.

## Bounded proposal

The next corner-specific detector should look for a **two-stage corner
cascade**: three obstacle ends forming two successive agent-width turns, where
each individual overlap is below the existing 10.3-unit channel threshold but
the sum of collision-forced traversal depths is at least 15.1 units. Accept it
only when all of the following static checks pass:

1. a radius-5.01 path connects the rear staging point to the bait;
2. every radius-10.01 route from the front to the bait crosses both corner
   gates, and the shortest collision-free endpoint jump across their union is
   greater than 15 units;
3. bait-to-every-predator-valid-centre clearance is greater than 15.05 units;
4. the front hold point is within 40 units and the rear staging region remains
   outside the front predator component.

This is a concrete way a corner arrangement could add maps without being an
ordinary crevice. It needs a triple-obstacle survey before implementation; the
present evidence does not claim that native random maps contain such triples.

## Artifacts

- `diagonal_corner_gate.py`: standalone native experiment.
- `trials-final2/results.json`: all nine final results.
- `trials-final2/aperture-*/result.json`: per-case summary.
- `trials-final2/aperture-*/ticks.jsonl.gz`: every native tick.
- `trials-v2/`: failed zero-depth diagonal-gate evidence.

Earlier `trials-v3`, `trials-v4`, and `trials-final` folders are development
runs that exposed fixture placement sensitivity; they are not supporting
evidence for the final result.
