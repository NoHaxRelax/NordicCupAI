# Class-agnostic anomaly localization

This experiment tests whether the inserted objects look statistically different
from the simulated terrain before trying to name them. The detector emits only
ranked boxes. Object labels are read after detection solely to measure recall.

The score combines three fixed, untrained cues: rare Lab colours, disagreement
with a multi-scale local background, and edge structure. Connected anomalous
regions generate several class-neutral box extents, followed by NMS. No
parameters were fitted to the annotations.

## Result: hand-written anomaly score

The committed `results.json` evaluates all 259 labeled appearances in the 25
organizer frames. `L0` processes the complete frame once. For `L1` and `L2`, a
deterministic camera window is centered on each target; those rows therefore
measure localization after the right region is visible, not autonomous camera
search.

The original recall-only result was too forgiving because multiple boxes could
claim the same object. The current evaluator uses standard confidence ranking
and greedy one-to-one matching. Once an object is matched, every duplicate is a
false positive. AP is 101-point interpolated class-agnostic AP at the stated IoU.

| zoom, top 25/view | AP50 | TP | FP | FN | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L0 (fully out) | 0.000% | 0 | 625 | 259 | 0.000% | 0.0% |
| L1 (in once) | 0.005% | 15 | 6,460 | 790 | 0.232% | 1.9% |
| L2 (in twice) | 0.162% | 56 | 6,419 | 256 | 0.865% | 17.9% |

At 300 proposals per view, L2 reaches 160 TP and 77,540 FP at IoU 0.50
(51.3% recall, 0.21% precision, 0.344% AP50). Thus the hypothesis is partially
supported at native-pixel zoom but rejected as a complete first-stage detector
in this form. L0 does not retain enough signal for these generic cues.

## Result: simple binary CNN

`cnn_experiment.py` trains a 79k-parameter CNN from scratch to distinguish an
object crop from a background crop, then uses it only to rerank the same anomaly
boxes. It never predicts an object class. Eight object identities are used for
training and eight are held out completely.

| zoom, top 25/view | AP50 | TP | FP | FN | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L0, all objects | 0.000% | 0 | 625 | 259 | 0.000% | 0.0% |
| L1, all objects | 0.201% | 59 | 6,416 | 746 | 0.911% | 7.3% |
| L2, all objects | 0.437% | 72 | 6,403 | 240 | 1.112% | 23.1% |
| L2, held-out identities | 0.261% | 27 | 2,798 | 86 | 0.956% | 23.9% |

The CNN improves ranking, including on unseen object identities, but precision
remains around one percent. At top 100/view on held-out L2 identities it reaches
65 TP, 11,235 FP, 57.5% recall, 0.575% precision and 0.544% AP50. It cannot fix
objects absent from the anomaly proposal set.

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
python anomaly/cnn_experiment.py --steps 500
```
