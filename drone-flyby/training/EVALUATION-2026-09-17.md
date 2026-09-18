# First detector test: not ready for use

The trained YOLO26x recognizes its reference training scene well but fails on most of the manually labeled validation targets. This first baseline is fast enough for further development, but its detection quality is not acceptable yet.

## Primary result

A correct detection requires the correct class, confidence at least 0.25, and box IoU at least 0.50. These thresholds were specified before inference.

| Zoom | Reference training-view recall | Known validation targets correctly detected | Validation targets localized, any class |
| --- | --- | --- | --- |
| L0 | 243 / 259 = 93.8% | 0 / 71 = 0.0% | 0 / 71 |
| L1 | 563 / 600 = 93.8% | 1 / 71 = 1.4% | 6 / 71 |
| L2 | 75 / 77 = 97.4% | 0 / 71 = 0.0% | 9 / 71 |

The one correct L1 detection was a hangar. The model localized five L1 jet-plane targets and nine L2 jet-plane targets but labeled them as `condor`. Most other targets received no matching box. The reference training-view precision was 100.0%, 99.6%, and 100.0% respectively; these are training-fit numbers, not independent accuracy.

Lowering confidence to 0.10 gives only 1/71, 3/71, and 1/71 correct validation targets at L0/L1/L2. Raising it to 0.50 gives zero at all three zooms. A confidence-threshold adjustment alone does not repair this result. No threshold was selected or model retrained using these outcomes.

## What was tested

- The exact epoch-50 checkpoint from `mypc-yolo26x-20260917-v2`, SHA256 `14f34b0b97f228342f0b71c5e8fb2c603d0100abc725aa74b675458c3b7f5897`.
- 250 fully labeled reference camera views that the detector trained on.
- 213 full-context camera views, rendering the same 71 frozen manual holdout appearances at each of three zooms. Delivered images remain 960 by 540; downsampling follows the training camera renderer.
- Six physical tracks/classes: condor (15 appearances), hangar (12), helicopter (11), jet plane (15), large tower (10), and tank (8).
- The detector trained only on the organizer reference scene. Manual crops were prepared for the separate classifier; they were not detector training data. This test did not evaluate the classifier.

The validation camera windows use known target boxes to place the camera, with fixed seeded offsets. This is a favorable test of recognizing an already visible target, not a test of autonomous camera search. Appearances from the same tracks are correlated and are not 71 independent objects. Manual labels are participant pseudo-labels and incomplete, so **validation precision, false-positive rate, and mAP cannot be computed from them**. Other predictions may correspond to real unannotated objects.

## Checks and diagnosis

All checkpoint, image, label, source-frame and manifest hashes were verified, and model class names matched the frozen class map exactly. Source reconstructions were fully opaque, excluding missing pixels. Coordinate transforms kept the full target box inside each view. Representative median-frame examples for all six tracks were visually inspected at all three zooms; the target objects are visible and the missing/wrong predictions agree with the numerical result. Manual box quality still limits exact IoU conclusions, but it does not explain the widespread absence of detections.

The gap is consistent with severe overfitting and a change in background, object appearance, orientation and scale. This experiment does not isolate the contribution of each factor. The training scene contains only one physical object per class. Its supposedly zoom-weighted camera sampling also produces uneven per-class coverage:

| Class | L2 labeled training appearances | Median L1 training box | Median L1 validation box |
| --- | --- | --- | --- |
| Condor | 0 | 85 x 83 px | 45 x 38.5 px |
| Hangar | 5 | 93.5 x 64.2 px | 97.5 x 88.8 px |
| Helicopter | 21 | 57.5 x 45.5 px | 84.5 x 62 px |
| Jet plane | 1 | 38.5 x 40.5 px | 34.5 x 26.5 px |
| Large tower | 0 | 31.5 x 31.5 px | 76 x 65 px |
| Tank | 5 | 25.3 x 24 px | 55 x 32.5 px |

Counting camera zoom alone is insufficient: the next dataset should cover each class across actual object pixel sizes, orientations and backgrounds.

## Recommended next change

Build a balanced class-by-size/zoom training set with broader orientation and background variation. Add reviewed examples from the validation domain while keeping whole physical tracks out for evaluation. Full-frame detector training requires exhaustive boxes or an explicit way to ignore unlabeled objects; partial manual labels must not turn real objects into background negatives. Test the separate crop classifier when its checkpoint becomes available, to distinguish classification quality from detection/localization quality.

Do this before increasing training duration or choosing a larger model. The existing model's measured p95 inference latency is 21.12 ms on RTX 4080, excluding HTTP/network/camera logic. It has runtime headroom, but no acceptable recognition result yet.

## Reproducibility and artifacts

Evaluation code was committed and pushed as `c4ffb35059c7068f416bded337b669c51a6ed22a` before inference. It ran 463 views on the RTX 4080 in approximately 12 seconds, including image reads and hashing; this batch elapsed time is not the endpoint latency benchmark. W&B stayed disabled for this original model. No competition API or final evaluation attempt was called.

- Frozen inputs: `manifests/detector-eval-20260917.json`.
- Full metrics: `receipts/mypc-detector-evaluation.json`.
- Artifact paths and hashes: `receipts/mypc-detector-evaluation-artifacts.json`.
- Local visual review: `artifacts/drone-detector-eval-20260917/zoom-comparison.jpg` and `l1-full-views.jpg` in the preparation workspace.
- Raw predictions and report are preserved both locally and on mypc under the completed run's `evaluation-20260917/` folder.

The visual sheets select the median frame of each track without inspecting its predictions. Green boxes are manual targets; purple boxes are model predictions at confidence 0.25 or above. They deliberately show representative failures, not a selection of successful examples.
