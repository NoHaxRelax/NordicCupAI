# Drone flyby: night of 18 to 19 September, what was found and what to deploy

Branch `drone/elias-verifier`, worktree `../NordicCupAI-drone`. Everything below is measured; where a number
comes from the team's own pseudo-labels instead of organiser ground truth it says so.

## The one-paragraph version

The team had no detector that works on a scene it was not built from (the earlier YOLO found 0 of 71 objects
on the other scene). It has one now. A YOLO trained only on synthetic views made from Helsinki sprites,
with object and terrain exposure varied independently, a medium backbone and inference at 1280 px, scores
**0.565 on the validation portal against organiser ground truth without ever having seen that scene**
(two concealed half runs, 0.350 + 0.215; the public board stayed at the team's 0.511). Four settings in
Oscar's pipeline matter as much as the model. The evaluation flight is a third unseen scene, so this
unseen-scene number, not the in-scene one, is the estimate to plan with.

## What to deploy (current best, will be updated at the end of this file)

    DRONE_DETECTOR=ultralytics DRONE_WEIGHTS=<medium model> DRONE_IMGSZ=1280
    DRONE_OVERVIEW_BETWEEN_SIDES=0      # L1 left, centre, right, centre: no L0 overviews
    DRONE_MISS_RULE=seen                # a miss only counts when the view gave the object enough pixels
    DRONE_CONF=0.05 DRONE_BIRTH_CONFIDENCE=0.25 DRONE_UPDATE_CONFIDENCE=0.15

Serve it from a GPU machine, not the laptop: at 1280 px the laptop needs about 200 ms per frame while
training something else, and the portal run lost 10 to 13 of 249 frames.

## Answers to the six questions

| question | answer | evidence |
|---|---|---|
| 1. A vision model that verifies the expert system | The expert system's recogniser cannot run live (4 s per view) and only recognises the instances it was calibrated on. Replace it: a standard detector trained on synthetic views. A window classifier used as verifier adds about +0.02 by re-ranking; letting it relabel boxes hurts (0.329 to 0.258). | `elias/views.py eval --yolo --verifier` |
| 2. Which pixel size works | Recognisable size is the binding constraint of the whole system: with a perfect detector that needs N delivered pixels the deployed cycle scores 0.92 / 0.75 / 0.52 / 0.38 for N = 0 / 16 / 24 / 32. Upsampling the view to 1280 before detection is worth +0.09 end to end; small_launcher goes 0.02 to 0.42. 1536 is worse. | oracle sweep, `run_harness.sh --imgsz` |
| 3. One model or one expert per class | pending (classifier matrix) | `elias/out/runs/*4_experts.json` |
| 4. Synthetic data | Essential, and two properties decide everything: exposure and backbone size. Real-only training transfers at 2 to 5 %. The same object is 1.3 to 2.0 times brighter in Helsinki than in validation while Helsinki terrain is darker, so object and background exposure must be jittered independently and widely (window accuracy 0.29 to 0.53). Blending sprites softly into the background HURTS (0.316 to 0.268): the real frames are hard-pasted renders. Colour jitter hurts. | photometric ablation, pod 2 |
| 5. Is L0, L1 left, L0, L1 right the right cycle | No. L0 overviews give most classes too few pixels, and every L0 frame counted as a "visible miss" that retired small-object tracks. L1 left, centre, right, centre (a flag that already existed) plus a zoom-aware miss rule: 0.517 to 0.673 with a 24 px oracle, 0.185 to 0.228 with a real detector. A cued hybrid (sweep plus targeted zooms) simulates at 0.82 to 0.93 and is not built yet. | `elias/policy_sim.py`, harness |
| 6. One recognising frame or two | One. Motion is a per-flight perspective field; one sighting plus that field beats two-frame velocity at every horizon. Forecasts hold 8 to 12 frames, so tracks need refreshing. | three independent measurements, `01-subsystem-map.md` |

Rotation (your tilt question): pitch is 18 degrees, the nadir sits at the bottom edge. Low objects rotate
freely; tall ones (towers, ta-ta, launchers) lean away from the nadir, always into the upper half of the
image, so they get a limited turn and no vertical flip. No cast shadows, no fixed-sun shading.
See `elias/out/tilt_report.md`.

## Numbers

Unseen scene = trained on Helsinki-derived data only, tested on the validation scene.

| model and settings | real views mAP50 | end to end (local, pseudo-labels) | portal (organiser truth) |
|---|---:|---:|---:|
| window classifier as sliding detector | 0.03 | | |
| yolo26s, defaults, deployed cycle | 0.316 | 0.185 | |
| yolo26s, no-overview cycle + zoom-aware misses | | 0.228 | |
| + thresholds 0.05 / 0.25 / 0.15 | | 0.329 | about 0.26 (one fifth of a run, extrapolated) |
| + inference at 1280 | | 0.346 | |
| yolo26m | 0.423 | 0.441 | |
| yolo26m + inference at 1280 | | **0.533** | **0.565** (0.350 + 0.215) |
| in-scene: yolo26s on both scenes | 0.99 | 0.808 | 0.50 (0.215 + 0.209 + 0.080) |

The last row is the warning: the local harness scores against the labels the model was trained on, so it is
circular for in-scene models. Trust it for unseen-scene models only.

Ablations on the unseen scene (trained on Helsinki-derived views only, scored on 500 REAL views of the validation
scene, 736 boxes, best epoch; run-to-run noise is about 0.02):

| variant | mAP50 | reading |
|---|---:|---|
| yolo26s at 960, baseline recipe | 0.316 | |
| low colour jitter | 0.296 | no gain |
| frozen backbone | 0.267 | worse |
| soft blending into the background | 0.268 | worse: the real frames are hard-pasted renders |
| hard paste (no feather) | 0.324 | same as the 1 px feather |
| free rotation (no lean limits) | 0.257 | worse: tall objects must keep their lean |
| narrow exposure (8 %) | 0.277 | worse: wide independent exposure matters for the detector too |
| low learning rate, no mosaic | 0.279 | worse |
| trained at 1280 | 0.356 | better |
| trained at 1536 | 0.384 | better still, but slow to serve |
| **yolo26m at 960** | **0.423** | backbone size is the biggest lever |
| **yolo26m trained at 1280** | **0.528** | best on this metric |

End to end through the real pipeline the ranking of the two medium models flips (0.533 for the 960-trained one at
inference 1280, 0.44 to 0.46 for the 1280-trained one), almost entirely through one class: helicopter, of which
Helsinki offers a single real example. The validation flight holds about 31 physical objects, so one object moves
the end-to-end score by several points; the real-view metric above has the larger sample. An ensemble of the two
scores 0.507. Policy variants with the real detector: L2 revisits every fourth frame 0.443, L1 band 10 % lower
0.421 (baseline 0.533), both worse. Pending at the time of writing: large and extra-large backbones, UC Merced
extra backgrounds, the reverse direction (validation-derived data tested on Helsinki).

## Things found along the way

- The validation flight contains a row of **small_plane** that the team never labelled (frames 110 to 142,
  x 2600 to 2850). The portal averages over 13 classes, the labels cover 11; the second hidden class was not found.
- Two validation "mine_roller" tracks are tanks; large-launcher tracks c and c2 are a smaller vehicle.
- Tracker: `miss_rule='seen'` added (default unchanged, 44 tests pass). Two more fixes measured by a
  reader with the oracle and not yet applied: detector extents instead of the Helsinki-fitted size prior,
  and a bug that never births the middle object of a same-class cluster.
- Concealed validation runs: `DRONE_ANSWER_WINDOWS="a:b"` runs the pipeline over the whole sequence and only
  emits answers inside the window. Disjoint windows add up to the full score within about 0.015.

## Cost

RunPod: two RTX 4090 pods at 0.74 USD per hour. Final figure at the end of the night.
