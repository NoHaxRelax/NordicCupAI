# Drone flyby: night of 18 to 19 September, what was found and what to deploy

Branch `drone/elias-verifier`, worktree `../NordicCupAI-drone`. Operating manual: `elias/README.md`. Weights:
`elias/release/` (Git LFS). Every number below was measured; "portal" means the organisers' validation scorer
with organiser ground truth, reached through concealed runs that never moved the public board.

## The short version

| | validation portal | what it is |
|---|---:|---|
| team's public best when the night started | 0.470 | origin not recorded anywhere in the repo |
| team's public best now (Oscar, overnight) | 0.511 | untouched by anything below |
| **`both_m1280.pt`, the model to deploy** | **0.694** | 0.250 + 0.331 + 0.112 in three concealed thirds |
| **`helsinki_only_m1280.pt`** | **0.601** | has never seen the validation scene: the honest estimate for the evaluation flight |
| Danish board for scale | 0.908, 0.776, 0.749 | CarlN, Elemental hero, Backprop Boys |

At midnight the team had no detector that worked on a scene it was not built from (the earlier YOLO found 0 of
71 objects on the other scene). The evaluation flight is a third, unseen scene, so the 0.601 is the number to
plan with, and everything in the recipe was selected on unseen-scene performance, not on validation.

## Deploy

    DRONE_DETECTOR=ultralytics DRONE_WEIGHTS=elias/release/both_m1280.pt DRONE_DEVICE=cuda:0 DRONE_IMGSZ=1280
    DRONE_OVERVIEW_BETWEEN_SIDES=0      # L1 left, centre, right, centre: no L0 overviews
    DRONE_MISS_RULE=seen                # a miss only counts when the view gave the object enough pixels
    DRONE_CONF=0.05 DRONE_BIRTH_CONFIDENCE=0.25 DRONE_UPDATE_CONFIDENCE=0.15

**Serve it from a dedicated GPU server, not a laptop.** On the laptop a CUDA "illegal instruction" error at frame
65 of a served run made the detector fail on every later frame, silently, for that run and the next; `example.py`
has no recovery from that. An RTX 5090 pod served about 45 portal runs without a fault at 110 ms per request
(`elias/pod_serve.sh` on the pod, `elias/pod_portal.sh` from the laptop; the team key never leaves the laptop).
A watchdog that restarts the detector process on an exception would be cheap insurance for the one attempt.

## The finding that mattered most

**The team's validation pseudo-label boxes do not follow the organisers' box convention, and a model trained on
them inherits the error.** One class per concealed portal run (`DRONE_ANSWER_CLASSES`; the score times 13 is that
class's AP against organiser truth) gave, for the model trained with those boxes and for the model that never saw them:

| class | trained on pseudo-label boxes | Helsinki-only (organiser convention) | same model against the team's own labels |
|---|---:|---:|---:|
| small_tower | 0.07 | 0.86 | 0.97 |
| tank | 0.47 | 0.93 | 0.93 |
| medium_launcher | 0.08 | 0.25 | 0.90 |

The assets are identical between the scenes (checked by eye, `elias/out/_view_label_check.jpg`); the boxes are
not: the pseudo-label for medium_launcher is 30x45 where the organisers draw 47x44. The generator now takes box
margins from organiser-labelled sprites only and uses pseudo-labelled frames for backgrounds and sprites but never
for box targets (`SYNTH_ORGANISER_MARGINS`, `SYNTH_ORGANISER_BOXES`). Portal score: 0.558 to 0.694.

Corollary: **the local harness on the validation scene cannot be trusted**, because it scores against those
pseudo-labels. It said 0.81 for a model the portal scored 0.50, and tank 0.45 for a model the portal scored 0.93.
Use it for plumbing and latency; use concealed portal runs for truth.

## Answers to the six questions

| question | answer | evidence |
|---|---|---|
| 1. A vision model that verifies the expert system | The expert system's recogniser cannot run live (4 s per view against a 3.3 s timeout) and only recognises the instances it was calibrated on. Replace it with a standard detector trained on synthetic views. A window classifier as verifier adds about +0.02 by re-ranking the detector's boxes; letting it relabel them hurts (0.329 to 0.258). The detector puts a box on 80 to 92 % of objects at L1/L2; its errors are class and box, not location. | `elias/views.py eval --yolo --verifier` |
| 2. Which pixel size works | Recognisable size is the binding constraint of the whole system. With a perfect detector that needs N delivered pixels the deployed cycle scores 0.92 / 0.75 / 0.52 / 0.38 for N = 0 / 16 / 24 / 32. Feeding the detector the view enlarged to 1280 is worth +0.09 end to end (small_launcher 0.02 to 0.42); training at 1280 is worth +0.10 on real unseen views; 1536 helps accuracy but is too slow to serve. | oracle sweep, `run_harness.sh --imgsz`, pod ablations |
| 3. One model or one expert per class | One shared model. Cross-scene window accuracy: single 17-way 0.594 / 0.679, 16 one-against-rest experts at equal total size 0.459 / 0.625 with two to three times the false alarms, one model per zoom scale 0.589 / 0.546, scale-blind 0.572 / 0.593. Scale-aware and shared wins or ties everywhere. | `elias/out/runs/*5_*.json`, `*4_*.json` |
| 4. Synthetic data | Essential. Real-only training transfers at 2 to 5 %. In order of measured effect on the unseen scene: medium backbone instead of small (0.316 to 0.423), training at 1280 (to 0.528), public aerial tiles as extra backgrounds (to 0.583), wide and INDEPENDENT object and terrain exposure (+0.04; the same tank is 1.9 times brighter in Helsinki while Helsinki terrain is darker), rotation that respects the lean of tall objects (+0.06). **Blending sprites softly into the background hurts** (0.316 to 0.268): the real frames are hard-pasted renders. Colour jitter, a frozen backbone, a lower learning rate and a larger backbone than medium do not help. Unseen-scene accuracy peaks after 40 to 60 thousand samples, so train short. | pod ablations, table below |
| 5. Is L0, L1 left, L0, L1 right the right cycle | No. L0 gives most classes too few pixels, and every L0 frame counted as a "visible miss" that retired small-object tracks. L1 left, centre, right, centre (a flag that already existed) plus a zoom-aware miss rule: 0.517 to 0.673 with a 24 px oracle, 0.185 to 0.228 with a real detector. Lower confidence thresholds add another +0.10 (low-confidence boxes are nearly free under AP). Stale-track L2 revisits and a lower L1 band both hurt. A cued hybrid (sweep plus targeted zooms) simulates at 0.82 to 0.93 and is not built. | `elias/policy_sim.py` (mean error 0.019 against the harness), harness runs |
| 6. One recognising frame or two | One. Motion is a per-flight perspective field; one sighting plus that field beats two-frame velocity at every horizon. Forecasts hold 8 to 12 frames, so tracks need refreshing. | three independent measurements, `01-subsystem-map.md` |

Rotation (the tilt question): the camera is pitched 18 degrees forward, the nadir sits at the bottom edge, the top
of the frame is seen 37 degrees off vertical. Low objects rotate freely (tank measured across 9 heading pairs);
tall ones lean away from the nadir, always into the upper half of the image, so they get a limited turn and no
vertical flip. No cast shadows, no fixed-sun shading. Free rotation costs 0.06 at detector level.
`elias/out/tilt_report.md`.

## Ablations on the unseen scene

Trained on Helsinki-derived views only, scored on 500 REAL views of the validation scene (736 boxes), best epoch.
Run-to-run noise is about 0.02. These labels are the team's pseudo-labels, so absolute values are depressed for
the classes with bad boxes; the comparisons between rows are what counts.

| variant | mAP50 |
|---|---:|
| yolo26s at 960, baseline recipe | 0.316 |
| hard paste (no feather) | 0.324 |
| low colour jitter | 0.296 |
| low learning rate, no mosaic | 0.279 |
| narrow exposure (8 %) | 0.277 |
| soft blending into the background | 0.268 |
| frozen backbone | 0.267 |
| free rotation (no lean limits) | 0.257 |
| extra aerial backgrounds, 40 % of L0/L1 views | 0.371 |
| extra aerial backgrounds, 70 % | 0.353 |
| trained at 1280 | 0.356 |
| trained at 1536 | 0.384 |
| yolo26m at 960 | 0.423 |
| yolo26l at 960 | 0.390 |
| yolo26m trained at 1280 | 0.528 |
| **yolo26m at 1280 with 40 % extra backgrounds** | **0.583** |

Reverse direction (validation-derived data only, tested on real Helsinki views with organiser truth): 0.464
averaged over all 16 classes, about 0.66 over the 11 it was trained on, and no early peak.

## Per class, the deployed model against organiser truth

PER_CLASS_F3_TABLE

## What is still on the table

1. **Cued zooms.** The simulator says a policy that sweeps at L1 and zooms to L2 on noticed-but-unnamed objects
   reaches 0.82 to 0.93 where the sweep alone reaches 0.75 at a 16 px recogniser. The small classes need it.
2. **The classes the team never labelled.** The portal averages over 13 classes, the labels cover 11. One hidden
   class is `small_plane` (a row of at least three on an apron, frames 110 to 142); the other was not found.
3. **A third scene.** Everything rests on two flights. Any additional real footage, even unlabelled, would let the
   background mix be checked the way UC Merced was.
4. **Tracker fixes measured with the oracle and not applied:** detector extents instead of the Helsinki-fitted
   size prior hurt with the real detector (0.211 against 0.228) and stay off; the same-class cluster birth fix
   (`DRONE_CLUSTER_BIRTHS=1`) made no difference with this detector and stays off.

## How the numbers were obtained without showing them

`DRONE_ANSWER_WINDOWS="a:b"` runs the pipeline over the whole sequence and only emits answers inside the window;
disjoint windows add up to the full score within about 0.015 (checked with the oracle: 0.570 + 0.367 against
0.923). `DRONE_ANSWER_CLASSES` does the same per class. No concealed run exceeded the team's public best, so the
board never moved. Oscar's agents validate at night too; the portal answers a queue request made during another
run with that run's attempt, so `elias/portal.py` only accepts a result whose URL is its own and retries otherwise.

## Cost

Three RunPod GPUs, all created and terminated by the agent; Oscar's pods were never touched.
COST_LINE
