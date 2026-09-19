# Hysteretic pacing, paired diagnostic

Predeclared cases: five fresh100 sprint-depletion failures (20, 29, 37, 41,
45) and each immediately preceding successful index (19, 28, 36, 40, 44).
Both default and candidate were rerun from frozen snapshots with two workers.

The candidate walked above 70 units, retained its pace state between 55 and 70,
and sprinted below 55. Final approach and lost-contact recovery retained the
default behavior. Steering considered paced candidates first and could add an
emergency sprint only if every paced candidate was unsafe.

## Result

- Baseline reproduced 5/10 passes: all controls passed and all diagnosed
  failures failed.
- Candidate passed 7/10: failures 20, 29, and 45 became passes; failures 37
  and 41 remained failures; control 19 regressed.
- None of the five diagnosed cases crossed below 20% energy under the candidate.
  Their sprint ticks changed from 87/73/69/64/58 to 43/46/2/14/27.
- The thresholds are unsafe. Cases 37 and 41 were caught by newcomer 31 only
  2.6 and 2.1 seconds after delivery start. A walking candidate was still safe
  in the controller's one-step test, so emergency sprint evaluation was never
  entered. Native predator motion closed the remaining gap before the 55-unit
  trigger could protect the guide.
- Control 19 still delivered the newcomer, but delivery lengthened from 44.1
  to 52.6 seconds and one original predator briefly left the held set. It
  therefore changed from pass to `initial_predators_escaped`.

This confirms that continuous sprinting causes avoidable reserve depletion,
but rejects the 55/70 controller and its one-step emergency condition. Any
next pacing attempt needs a predictive closing-speed guard or a substantially
earlier sprint trigger, and must retain the original-group regression check.
This ten-case development sample is not promotion evidence.

## Second iteration

Two further frozen variants used the same ten cases. A 75/95 hysteresis passed
8/10 but still regressed control 19, and four of five diagnosed cases still
crossed below 20% energy. It is rejected.

The predictive variant used only consecutive ordinary target observations. It
entered sprint when `observed distance - measured closing per tick - 15` was
at most 55, reserving one known maximum predator step for observation lag. It
passed 9/10 and preserved all five controls. Failures 20, 29, 41, and 45 became
passes. Case 37 changed from an early guide death to a live 60-second delivery
timeout; it never crossed 20% energy. Four of five diagnosed cases avoided the
20% threshold; case 41 crossed late but delivered.

This is the first candidate in this pacing sequence to satisfy the small-sample
control gate. It still needs an independent paired benchmark before promotion.
