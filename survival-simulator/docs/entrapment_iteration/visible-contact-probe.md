# Visible-contact delivery experiment

This experiment starts from the exact frozen source used by
`track-rejoin-fresh100`.  It changes one branch in `my_guide.py`: a motion-
associated predator that remains visible no longer triggers route recovery
when the following heuristic temporarily disagrees.  Recovery still occurs
when the target is no longer observed.

The mechanism is route churn.  A guide turning around walls can make observed
motion temporarily incompatible with the following model.  The baseline then
walks back toward a predator it can still see, loses route progress, and spends
more time and energy before handoff.  Fresh case 29 spent 67 ticks returning
and failed 52.43 units from handoff; the variant spent zero return ticks and
passed at 36.8 seconds.  Case 37 also flipped from failure to pass.  Case 20
did not improve despite reducing reacquisitions from 18 to 4, showing that
heuristic disagreement is only one delivery failure mechanism.

Paired native result on the same 12 fresh-map encounters:

| policy | passes | failures |
| --- | ---: | ---: |
| frozen baseline | 5 | 7 |
| visible-contact variant | 7 | 5 |

All five baseline successes stayed successful.  Two of seven selected
clean-retention failures flipped to pass.  The sample was chosen after seeing
the baseline outcomes, so this is mechanism evidence, not a reliability
estimate; an independent repeated-site benchmark is required before promotion.
The evaluation success gate is unchanged.

Frozen baseline `my_guide.py` SHA-256:
`1365e4a42c097ef14e6c99b7cdc23acf636c1447686e8a61613bb94f499b5b51`.
Visible-contact variant SHA-256:
`b64a8be8e8c5eab10b819ebb543f719fb9b901e7ad3c5c94c4dae8cf3e4df02b`.

Separately, among all 12 `initial_predators_escaped` outcomes in the fresh-100
baseline, five recorded an original predator beyond 60 units from bait and
seven only recorded excursions beyond the strict 40-unit held threshold.  No
success criterion was relaxed on that basis.
