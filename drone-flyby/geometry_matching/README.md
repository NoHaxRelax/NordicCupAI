# Drone flyby: geometry and training-object matching experiment

The camera model fits the validation motion well, but this prototype does **not** find all validation objects. The successful result is a calibrated flight model. Sparse 3D reconstruction and rotated-template retrieval are not yet reliable enough for autonomous discovery.

## Camera and flight

The supplied training metadata specifies 600 m altitude and 13.8888889 m between frames. At 3 fps this is 41.67 m/s, or 150 km/h. The model uses a constant horizontal trajectory, a fixed camera, and a local flat ground plane.

| Quantity | Training box-centre fit | Validation image-motion fit |
|---|---:|---:|
| Focal length | 3140.14 px | 3309.42 px |
| Tilt forward from nadir | 18.25° | 19.90° |
| Pitch below horizontal | 71.75° | 70.10° |
| Horizontal × vertical field of view | 62.89° × 37.96° | 60.24° × 36.15° |

The validation result agrees closely with the proposed 3340 px and 70° camera. It was measured from the saved pixels locally; no new pod was provisioned.

Validation calibration used background SIFT correspondences in seven frame pairs separated by three frames. Focal length, pitch and yaw were fitted with the physical flight step and altitude fixed. Alternating correspondences were withheld from fitting: across 6,172 held-out matches, median reprojection error was **0.70 px**, with **2.05 px at the 90th percentile**. Applying the nominal training camera gave about 6.95 px median error in the corresponding initial comparison.

This is a conditional calibration, not a unique recovery of every camera parameter. Holding focal length at 3140 px and fitting an effective step instead gives essentially the same image-motion residual. Known altitude and speed are needed to choose the physical scale; terrain height departures from the assumed plane remain a source of error. The held-out correspondences come from the same sampled frame pairs, not a separate flight.

## Inputs and separation of training from validation

All 249 reconstructed validation frame files were inspected by the full-resolution feature scan. Frames 1–3 contain no recoverable native image information; frame 4 has 81.25% native coverage; frames 5–249 have complete native coverage. Consequently, 246 frames supplied features, totalling 3,936,043 SIFT features and 41,458 tentative reference correspondences.

Only original Helsinki training examples were used to build the reference models. Existing sprite banks include validation-derived examples; those were explicitly excluded by source. The original dataset has one instance per class, observed over a limited range of views.

Validation annotations were read only for the final audit and the separately labelled conditional recognition diagnostic. They are participant-created annotations, not organizer ground truth, and some labels are known to be disputed. The strict audit subset contains 606 boxes on 25 directly reviewed tracks. It is not a complete validation census.

## Experiments and results

### 1. Sparse 3D reconstruction and rigid matching

Training-object foreground features were linked across views and triangulated under the fitted camera. Points required at least three distinct frames, a baseline of at least three frames, and low reprojection error. Original training sprite masks were used where available; other appearances used GrabCut inside organizer boxes.

This produced **293 sparse landmarks across 12 classes**. Hangar, helicopter, medium launcher and mine roller yielded no stable landmarks; large tower and small launcher yielded one each. Only nine classes had at least six landmarks. The exported PLY files are sparse observed-feature fits, not textured or watertight object meshes. Some inferred heights lie below the assumed ground plane, indicating calibration, terrain, segmentation or correspondence error; these are not trustworthy physical height measurements.

Validation matches were tested against a shared object yaw and ground position across time. A loose three-landmark gate produced a visually false forest match. After raising the requirement to six distinct landmarks, there were **zero accepted rigid 3D instances**. The rejected example is saved in `rejected_terrain_match.jpg`.

This experiment exposes an implementation limitation: sparse features are insufficient for several tiny or smooth objects, and foreground/background contamination can produce geometrically consistent false matches. It does not establish that dense multiview reconstruction or a better object model cannot work.

### 2. Rotated-template discovery fallback

A complementary scan covered all 16 classes using 1,152 template variants: 15° rotation steps and scales 0.85, 1.0 and 1.15. It used full frame 5, followed by the top 500 source rows every four frames, assuming static objects enter through the top of the view. Proposal search ran at half resolution; verification used native-resolution crops across several frames. The complete sequence was independently scanned at native resolution by the SIFT stage.

High-pass image features and normalized correlation reduce sensitivity to brightness and offset. Candidate verification required at least three supporting frames. These are approximate 2D transformations with ground-plane motion, not true relit 3D renders, and they cannot expose surfaces missing from the reference view.

The search generated 1,743 proposals. After temporal checks and spatial deduplication, 888 hypotheses produced 3,950 boxes. Against the partial reviewed reference:

| Measurement | Result |
|---|---:|
| One-to-one true positives at IoU ≥ 0.50 | 3 |
| Unmatched predictions | 3,947 |
| Reviewed boxes missed at IoU ≥ 0.50 | 603 |
| Precision against this partial reference | 0.076% |
| Recall against this partial reference | 0.495% |
| Reviewed tracks with a correct-class match | 1 / 25 |

Unmatched predictions are provisional false positives against this incomplete reference. They are not proven absent objects. Nevertheless, inspection and the large number of small-launcher-like terrain patches show that this candidate generator is unusably noisy. Correlation scores are exploratory ranking scores, not probabilities. These are not official competition AP results.

### 3. Recognition with object locations supplied

To separate discovery from recognition, a final diagnostic used existing reviewed boxes and up to five native-resolution appearances per track. GrabCut supplied query masks. The reference bank contained 90 Helsinki templates across all 16 classes. No validation object was added to that bank.

| Diagnostic | Top-1 agreement with reviewed label | Top-3 inclusion |
|---|---:|---:|
| Rotation and brightness-normalized appearance | 6 / 22 (27.3%) | 13 / 22 (59.1%) |
| Appearance plus camera-derived physical-size penalty | **10 / 22 (45.5%)** | **15 / 22 (68.2%)** |

Only 22 tracks had usable unclipped, segmented examples. These numbers are conditional on supplied locations, not autonomous detection accuracy. The size penalty was an exploratory implementation choice, not an independently calibrated probability model. Rotation was optimized per appearance; this diagnostic does not enforce a single 3D yaw or lighting model. Participant label errors also limit interpretation.

`recognition_examples.jpg` shows successful and unsuccessful matches beside training examples. The physical size constraint is useful, but recognition still confuses classes and cannot yet be trusted to complete the validation labels.

## What the 3D deliverables contain

- `flight_geometry.obj`: flat ground, the 249-position camera path, and six camera frusta in metres. It deliberately contains no invented validation objects.
- `results/*.ply` inside the ZIP: 12 sparse training-object landmark models. Local X/Y origins and source metadata are in `models.json`; Z refers to the assumed scene ground plane.
- `training_landmarks.png`: an overview of the sparse reconstructions, including failed classes.
- `scene.json`: fitted camera parameters and an empty accepted 3D instance list.
- `template_detections.json`: exploratory template hypotheses, **not** final validation annotations.
- `recognition_diagnostic.json`: ranked training-class suggestions for supplied reviewed tracks.

The next useful extension would be a foreground-aware object proposal stage followed by multiview verification with fuller object surfaces or original asset meshes. The fitted camera can constrain position, apparent size and temporal projection, but the experiments show that those constraints alone do not distinguish inserted objects from persistent terrain.

## Reproduction and validation

The ZIP includes source code, reference-bank descriptors, point clouds, camera fits, tentative correspondences, predictions and audit metrics. Original image data and the large per-frame feature cache are excluded. The code lives on local branch `codex/drone-geometry-matching`, based on `origin/drone/elias-verifier`; it has not been pushed or deployed.

Create a Python environment with NumPy, SciPy, OpenCV, Pillow and Matplotlib. The tested environment used Python 3.13.14, NumPy 2.5.3, SciPy 1.18.1 and OpenCV 5.0.0.93. Set `DATA` to a drone-flyby directory containing original Helsinki images and annotations, `elias/sprites`, and the reconstructed validation images. Then run:

```bash
python code/experiment.py --data "$DATA" --out results --stage all
python code/template_fallback.py --data "$DATA" --out results
python code/template_fallback.py --data "$DATA" --out results --complement --prefix dense
python code/evaluate_templates.py --data "$DATA" --out results
OPENBLAS_NUM_THREADS=4 python code/recognition_diagnostic.py --data "$DATA" --out results
```

Four geometric tests pass: projection/inverse projection at different heights, multiview triangulation, yaw-and-translation recovery without scale, and forward-flight image motion. The interactive flight diagram was checked at 360 px and 736 px, including the frame slider.

The calibrated projection and triangulation approach follows the standard pinhole-camera formulation documented in [OpenCV's calibration and reconstruction reference](https://docs.opencv.org/4.8.0/d9/d0c/group__calib3d.html). The numerical findings above come from this experiment's saved inputs and outputs.
