# Class-agnostic anomaly localization

This experiment tests whether the inserted objects look statistically different
from the simulated terrain before trying to name them. The detector emits only
ranked boxes. Object labels are read after detection solely to measure recall.

The score combines three fixed, untrained cues: rare Lab colours, disagreement
with a multi-scale local background, and edge structure. Connected anomalous
regions generate several class-neutral box extents, followed by NMS. No
parameters were fitted to the annotations.

## Result

The committed `results.json` evaluates all 259 labeled appearances in the 25
organizer frames. `L0` processes the complete frame once. For `L1` and `L2`, a
deterministic camera window is centered on each target; those rows therefore
measure localization after the right region is visible, not autonomous camera
search.

| zoom | recall@25, IoU .25 | recall@25, IoU .50 | recall@300, IoU .25 | recall@300, IoU .50 |
| --- | ---: | ---: | ---: | ---: |
| L0 (fully out) | 0.0% | 0.0% | 2.3% | 0.0% |
| L1 (in once) | 0.4% | 0.4% | 27.4% | 9.3% |
| L2 (in twice) | 34.0% | 21.2% | 66.0% | 55.6% |

At L2 the method finds at least some appearances of 12 of 16 object types at
loose overlap, but finds none of `condor`, `hangar`, `small_launcher`, or
`ta-ta`. A 300-proposal budget per 960x540 crop is also too noisy for a practical
cascade. Thus the hypothesis is partially supported at native-pixel zoom but
rejected as a complete first-stage detector in this form. L0 does not retain
enough signal for these generic cues.

The experiment is deliberately a favorable upper bound for L1/L2: selecting the
camera window uses the ground-truth location. A deployable system would still
need a scanning policy, deduplication/tracking, a way to assign competition
classes, and evaluation on an independent scene. The 25 Helsinki frames are
development data, so these numbers are not a generalization claim.

## Reproduce

From `drone-flyby` with OpenCV and NumPy installed:

```bash
python anomaly/experiment.py
python -m unittest anomaly/test_experiment.py
```
