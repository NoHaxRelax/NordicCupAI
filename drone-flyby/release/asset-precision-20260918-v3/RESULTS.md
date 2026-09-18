# Fixed rendered-asset recognition

The implementation combines foreground pixel matching, local-feature geometry,
tiny-object color/shape rules, and CNNs trained from scratch on mypc. Inference
consumes the delivered image and its scale. It does not consume frame numbers,
trajectories, expected object locations or evaluation labels.

The current frozen candidate is `asset-precision-20260918-v3`. Its code, trained
models, configuration, template bank and data-use records are saved under
`artifacts/drone-fixed-assets-20260918-v3/`. It retains 26/26 reference targets
with 26 proposals and 16/19 development targets with 20 proposals. Its separate
four-image test found **7/7 reviewed targets with 12 proposals**, versus the
original hybrid's **0/7 with 330 proposals**. The previous
v7 test and precision failure remain in the v2 directory, and the earlier recall
failure remains in v1.
This is an offline recognizer with runtime adapters, not an active flight-policy
replacement. No competition API or final evaluation was called.

## What was implemented

- **Foreground pixel matching.** Masked normalized correlation compares gray
  structure and high-pass detail, followed by foreground color agreement. It
  searches every translation at scales 0.85/1.0/1.15 and rotations -8/0/+8
  degrees. FFT evaluation runs these searches on the GPU. Tests compare the
  calculation numerically against OpenCV's spatial masked correlation.
- **Pose library.** The current bank has 241 crops: 210 reference and 31 validation
  crops. Twelve explicitly calibrated poses feed the GPU matcher. Earlier
  templates feed geometrically verified SIFT/RootSIFT matching. Calibration
  crops separate recognition pixels from projected output-box geometry.
- **Partial-object matching.** An additional jet mask recognizes its visible
  body when a roof hides the tail. A finer scale search handles medium launchers.
- **Tiny silhouettes.** Small-launcher detection uses training-derived Lab color
  ranges, pixel counts, connected components, rotated shapes and contrast with
  surrounding ground. The selected minimum contrast is 0.80.
- **CNN comparison and fusion.** A scratch detector contributes proposals that
  pass a separate crop verifier. A compact dense CNN contributes small towers.
  Geometry and pixel matches take priority for overlapping same-class proposals.
  Scores are similarities or model confidences, not calibrated probabilities.
- **Cross-checks for ambiguous matches.** Geometric matches scoring 0.70–0.80
  require both aligned foreground pixels and a same-class CNN verdict. Tiny
  silhouettes require the crop verifier at 0.80. Broad CNN proposals require
  verifier confidence 0.95; dense tower proposals retain their original class
  and require same-class confidence 0.40. Weak geometric matches cannot shift
  stronger boxes during fusion.

The selected settings are frozen in `selection-v3.json`. The bank and all three
selected checkpoints have SHA-256 identities. The bundle loader verifies them,
and the bank loader verifies every crop and mask.

## CNN experiments

All reported model training ran on the existing RTX 4080 in **mypc**.
Subsequent Runpod work is tracked separately in the v3 artifact directory.
The context continuation was inspected at epoch 24 and rejected after finding
only 1/19 development targets. Subsequent experiments used bounded step counts:

| Experiment | Training | Finding |
|---|---|---|
| Dense CNN v1, 1.61M parameters | 1,500 steps, about two minutes | Small-tower output complements other recognizers |
| Dense CNN v2 | 3,000 steps | Corrected mask fallback; no useful combined improvement |
| Dense CNN v3 | 3,000 steps, about 201 seconds | Calibrated-pose sampling and independent foreground/background variation |
| Crop verifier v3 | 800 steps, about 35 seconds | Synthetic object size mismatched detector proposal crops |
| Crop verifier v4 | 800 steps, 37 seconds | Corrected crop normalization; better development recall but excessive proposals |
| Crop verifier v5 | 800 steps, 23 seconds | Matched negative crop sizes/aspects and removed orange background from gray walker masks |

The v3 dense CNN plus v4 crop verifier, together with the deterministic branches,
found 26/26 reference and 17/19 development targets at permissive thresholds,
while emitting 392 and 739 proposals respectively. Raising thresholds traded
recall for fewer proposals. `development-threshold-diagnostic.json` preserves
that comparison. The selected runtime retains the earlier proposal CNNs, uses
the new v5 verifier, and combines them with geometric and GPU pixel matching.
Training-batch accuracy is not detection accuracy.

## Data use and annotation review

Across all runs, training or calibration used **41 distinct validation frames**:
35 early frames through frame 89, then 110, 150, 190, 206, 230 and 245. This counts
earlier crops later pruned from the bank. Repeated crops and zooms are not new
independent objects. The complete list is in `data-use-ledger.json`.

The first reserved test was frames 190, 206, 230 and 245. The earlier selected
candidate failed: **0/6**, versus **1/6** for the original hybrid. That result
has not been erased. Afterward, those four images were explicitly retired into
calibration. Correcting their displaced boxes did not rescue the old detector's
blind predictions. Six complete foreground crops were added to the new bank.

Eight new test frames were predeclared before viewing their pixels:
**183, 198, 204, 212, 220, 228, 241 and 249**. They contain 13 labeled targets
across five classes: tank, mine roller, large launcher, large tower and small
tower. They share physical tracks with calibration. They test different views
of those assets, not new objects or new scenes. After preserving that v7 result,
these eight images became development data for the precision follow-up. Four
additional frames, **187, 201, 224 and 247**, were reserved before inspection.
The v3 configuration was frozen before their image-only annotation review.
They contain seven labels across the same five classes. Another 53 frames in
181–249 remain uninspected by this task; this is not a claim about other tasks'
access or an untouched competition test set.

After freezing the detector and before running any fresh predictions, all 13
boxes received an image-only visual review. Several original boxes cut off
barrels or supports or drifted above the object. Seven original/reviewed box
pairs have IoU below 0.50. Both fixtures are preserved and both are scored:

- Original: `data/drone/fixed-asset-fresh-test-v2/`
- Reviewed: `data/drone/fixed-asset-fresh-test-reviewed-v2/`
- Audit: `artifacts/drone-fixed-assets-20260918-v2/fresh-annotation-review.json`

These are participant annotations reviewed by the developer, not organizer
ground truth. Canonical annotation releases were not overwritten. The current
v7 release also includes earlier jet/condor and medium/small-launcher identity
corrections. Three unresolved development-label concerns remain visible in the
v1 artifacts and remain in the development denominator.

## Evaluation contract

Every image is searched using 25 overlapping native 960 × 540 views. Matching
requires the correct class and one-to-one IoU ≥ 0.50. Clipped image-boundary labels
are excluded consistently. Models run before labels are used for scoring.
Incomplete labels mean unmatched proposals cannot all be called false positives.
The exhaustive scan is an offline benchmark, not a legal 25-crop flight policy.

| Historical development method | Reference matched | Reference proposals | Validation matched | Validation proposals |
|---|---:|---:|---:|---:|
| Original deterministic features | 18/26 | 18 | 2/19 | 4 |
| Original hybrid | 26/26 | 37 | 5/19 | 39 |
| Earlier v6 fixed-asset candidate | 26/26 | 29 | 16/19 | 33 |
| New compact CNN combination, permissive | 26/26 | 392 | 17/19 | 739 |
| Frozen v7 fixed-asset runtime | 26/26 | 29 | 16/19 | 23 |
| Frozen precision v3 runtime | 26/26 | 26 | 16/19 | 20 |

Development frames 100, 127, 140 and 160 have been inspected repeatedly. The
reference frames share their flight with the reference crop bank. Their scores
are development evidence only. The v7 row is a full end-to-end run of all 175 native views with its frozen
settings. The fresh-test comparison also reports original-label sensitivity.

The six-object masked-pixel retrieval check is 6/6 with six proposals at 0.70
similarity against the corrected calibration boxes. That verifies retrieval of
training assets; it is not a holdout result.

## Run the frozen bundle

```sh
python -m drone.scratch_objects \
  --bundle artifacts/drone-fixed-assets-20260918-v3/model/manifest.json \
  --image /absolute/path/to/delivered-view.png \
  --source-region 2160 1080 3120 1620 --device 0 \
  --output artifacts/my-fixed-asset-predictions.json
```

Use the actual source region represented by the image. Outputs use normalized
full-source-image coordinates. The Python API is
`drone.scratch_objects.bundle.load_bundle(path, device='0')`, with `detect`,
`predict`, and `tracking_detections` adapters. Use `device='cpu'` for diagnostics.
Tested training/evaluation runtime: Python 3.11, PyTorch 2.6, CUDA on RTX 4080,
and Ultralytics 8.4.155. CPU and GPU timings are not interchangeable; mypc also
hosted a separate CPU-heavy simulation during evaluation.

The exact source archive, launchers, checkpoints, training receipts, label audits,
raw predictions, overlays and tests are retained in the artifact directories.
All 44 local tests pass. The packaged v3 CLI smoke check retrieves a known tank
crop and emits valid normalized coordinates; this verifies the interface, not
generalization. The fresh baseline runs on the laptop CPU in parallel with the
selected detector on mypc. Checkpoint and data hashes are identical to the
previous baseline; platform and library differences are recorded, and their
timings must not be compared as an algorithmic speedup.
Do not use the old v1 bundle's development success as evidence that it passed its
reserved test.

## Precision follow-up

The strict GPU pixel matcher alone found 5/13 fresh targets with five proposals
at its frozen 0.70 threshold. Geometric feature matching is more tolerant of
viewpoint change. The full combined v7 run is preserved, but obvious roof and
vegetation false detections mean its broad output is not a finished high-precision
detector. Follow-up experiments are in `artifacts/drone-fixed-assets-20260918-v3`.
A separate four-frame check (187, 201, 224, 247) was predeclared before viewing
those images. Training/calibration remains at 41 validation frames.

The completed v7 fresh comparison is 12/13 reviewed targets with 233 proposals,
versus 4/13 with 799 proposals for the original hybrid. Against the preserved
original boxes the scores are 8/13 and 2/13. The geometric/pixel/color subset
retains 12/13 with 26 proposals. Three extra geometric detections were visually
confirmed as real unlabelled objects (tank and mine roller at 212, large tower
at 241). That audit did not change the holdout labels or scores. The remaining
obvious background proposals motivate the precision follow-up.

## Completed precision v3 result

The full eight-image consumed-data comparison improved from v7's 12/13 targets
with 233 proposals to **13/13 with 22 proposals**. Manual review of its nine
unpaired proposals identified three real unlabelled objects and six background
mistakes. These images were used during development of v3 and are not its test.

The four reserved images (187, 201, 224, 247) contain seven participant-labelled
targets across five classes. The frozen v3 detector found all seven with twelve
proposals. Reviewing the five extras revealed one real mine roller missing from
the labels and four mistakes: two vegetation regions, pale material stacks, and
a roof window. Labels and test scores were not changed after predictions.
The original boxes score 4/7, compared with 0/7 for the baseline.

| Reserved image | Reviewed targets matched | Proposals |
|---|---:|---:|
| 187 | 1/1 | 1 |
| 201 | 3/3 | 5 |
| 224 | 2/2 | 3 |
| 247 | 1/1 | 3 |

Every labelled test target is visible in
`artifacts/drone-fixed-assets-20260918-v3/fresh-detection-gallery.jpg`.
`comparison.json` retains the original-label sensitivity and all raw totals.
The geometry/pixel/shape proposal subset alone retains 7/7 with ten proposals,
but its weak geometry and tiny shapes already passed the CNN verifier, so this
is not a purely deterministic ablation.

The result demonstrates reliable matching for these close views, with remaining
background confusion. It does not establish perfect recognition of all sixteen
classes, generalization to a new scene, or readiness within a live flight budget.
The images share physical tracks with calibration. Median per-view time on the
loaded mypc machine was about 4.5 seconds; each full image used 25 views.
