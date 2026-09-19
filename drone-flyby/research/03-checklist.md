# Drone flyby: the checklist (living document)

Deadline Sunday 2026-09-20 16:00 CEST; top five submit code by 20:00. One evaluation attempt, only Elias queues it.
Every item is measured the same way: concealed validation runs, one class per run (score x 13 = that class's AP)
and the three thirds (0:83, 83:166, 166:100000, which add up to the full score). Public best stays 0.511 unless we
decide otherwise. Update the status column as things happen; do not delete rows, rejected ideas stay so nobody
repeats them. Started 2026-09-19 13:25 by Claude at Elias's request.

Status codes: TODO, DOING, DONE, REJECTED (with the number that killed it), BLOCKED (with what on).

## A. Detector and data

| # | item | expected | hours | status | evidence |
|---|---|---|---:|---|---|
| A1 | Retrain the F3 recipe with the unlabelled validation objects fixed: keep-out zones for 19 tracks, 42 new sprites (18 leaning ta-ta, 18 second small_tower, 6 second medium_launcher) | ta-ta 0 to about 0.5, small_tower and tank back to 0.86/0.93, medium_launcher up; +0.05 to +0.08 on validation | 3 | DOING | `runs/detect/elias/out/runs_yolo/F4_fixed_m1280`, chain `elias/out/logs/after_train2.sh` writes `elias/out/logs/after_train.log` |
| A2 | P2 detection head (stride 4) on the same data, YOLO26m-p2 at 1280 | small_launcher and ta-ta at L1 (10 to 15 px delivered) | 2 + 1.5 h GPU | TODO | VisDrone standard trick: arXiv 2512.07379, SAHI arXiv 2202.06934 |
| A3 | Large backbone (yolo26l) on the FIXED data | unknown; lost to medium on the old data (0.39 vs 0.42) | 0.5 + 2 h GPU | TODO, needs a pod | Elias asked for the big-model check |
| A4 | Spend the latency headroom: 1536 input, flip averaging, two-checkpoint ensemble (`elias/ensemble.py`) | +0.02 to +0.05 at detector level | 2 | TODO | 110 ms of a 300 ms budget used on an RTX 5090; laptop 85 ms at 1280 |
| A5 | DINOv2 prototype verifier (DE-ViT style): one prototype per class from the sprites, re-rank YOLO boxes by similarity, never delete | +0.02 like the CNN verifier, maybe more on sibling confusion | 3 | TODO | github.com/mlzxy/devit (MIT, weights). The CNN verifier relabelling HURT (0.329 to 0.258): re-rank only |
| A6 | ZoomDet learned warp inside the view | +8 mAP on SeaDronesSee in the paper | 6 + training | REJECTED for this deadline | arXiv 2602.07512; too new to trust for one attempt |
| A7 | Soft blending of sprites | | | REJECTED | 0.316 to 0.268: the real frames are hard-pasted renders |
| A8 | Free rotation of tall objects | | | REJECTED | costs 0.06; lean limits stay |
| A9 | Train on the real validation frames | would push the board to about 0.9 | 2 | REJECTED by Elias's rule | no evaluation gain, reveals our strength |
| A10 | Second small_launcher sprites from the unlabelled ones | | 1 | BLOCKED | GrabCut returns slivers on 25 px objects; needs a manual mask or a bigger box recipe |
| A11 | Jammer, condor, spacecraft never measured on a real unseen instance | unknown | | BLOCKED | absent from validation; only Helsinki (training set) has them |

## B. Answer policy (what we emit)

| # | item | expected | hours | status | evidence |
|---|---|---|---:|---|---|
| B1 | Sibling-class hedging: for launcher, plane and tower detections also emit the sibling classes at lower confidence | confusion stops being a miss; absent classes cost nothing, low-confidence extras cost almost nothing | 1 | TODO, next after A1 | scorer analysis in `research/02-night-report.md`; test with one-class runs for medium_launcher, medium_plane, large_tower |
| B2 | Confidence floor and thresholds (`DRONE_CONF=0.05 BIRTH 0.25 UPDATE 0.15`) re-tuned for the new model | +0.10 was the gain of lowering them the first time | 1 | TODO | |
| B3 | Emit forecast boxes for retired tracks at a low confidence instead of dropping them | small | 1 | TODO | low-confidence extras are nearly free |

## C. Camera policy

| # | item | expected | hours | status | evidence |
|---|---|---|---:|---|---|
| C1 | Zoom to L2 on cue: sweep at L1; when a detection is small or ambiguous, spend the next frame at L2 on its forecast position, then return | simulator 0.82 to 0.93 against 0.75 for the sweep at a 16 px recogniser; the only route to ta-ta (0 of the L1 looks, 22 to 26 L2 hits) | 4 to 6 | TODO, after B1 | `elias/policy_sim.py` policies `cued`, `cued_lcr` |
| C2 | Mid-life refresh: a lower-band look every ~8 frames so forecasts (hold 8 to 12 frames) get renewed during the ~16 frames below the band | part of the tracker's 0.08 | 2 | TODO, simulate first | add a policy to `policy_sim.py` |
| C3 | Density-crop rule: zoom where several noticed objects cluster so one L2 look pays for many | small | 2 | TODO, simulate first | ClusDet / DMNet family, survey doi 10.1007/s10462-025-11150-9 |
| C4 | Dwell two frames at L2 on a walker | small | 0.5 | TODO with C1 | |
| C5 | Bottom-band pass at the start | | | REJECTED | neutral on validation (+0.00 to +0.02), worse on Helsinki; `l1_lcr_bottom` in the simulator |
| C6 | Lower L1 band | | | REJECTED | measured worse |
| C7 | L0 overviews between the sides (Oscar's default) | | | REJECTED | 0.517 to 0.673 with the 24 px oracle when removed |
| C8 | Stale-track L2 revisits (`DRONE_REVISIT_EVERY`) | | | REJECTED | 0.443 against 0.583 |

## D. Serving and the attempt

| # | item | hours | status | evidence |
|---|---|---|---|---|
| D1 | Dedicated GPU server for the attempt (RTX 5090 pod: 45 runs without a fault at 110 ms) | 1 | TODO Sunday morning | laptop CUDA "illegal instruction" at frame 65 killed detections silently |
| D2 | Watchdog: restart the detector process on an exception, answer empty meanwhile | 1 | TODO | `example.py` has no recovery |
| D3 | Rehearsal on the pod: one concealed half run of the final weights from the pod before the attempt | 0.5 | TODO Sunday | `elias/pod_serve.sh`, `elias/pod_portal.sh` |
| D4 | Pick the final weights and settings, write them in `elias/README.md` deploy block | 0.5 | TODO | |
| D5 | Elias queues the evaluation attempt; nobody else, no agent | | TODO Sunday | hook `.claude/hooks/block-evaluation.py` |
| D6 | Terminate every agent pod afterwards, check `list-pods` | | TODO | |

## E. Known unknowns

- Origin of Oscar's 0.470 and 0.511 (live detector or label replay). Ask him.
- Whether the evaluation flight holds jammer, condor or spacecraft; the detector has seen one Helsinki instance of each.
- The organisers' box convention for ta-ta at its leaning appearance (our replay of the walkers scored AP 0.74 with three box sizes, so the convention is close to the detector's box).
- Run-to-run noise on the portal is about 0.01 to 0.02; differences smaller than that are not results.

## Done today (2026-09-19)

- ta-ta confirmed as the 13th validation class by portal replay (0.0571 = AP 0.74). Fourth candidate by the sea wall: 0.
- Unlabelled objects found: second small_tower, second medium_launcher, three ta-ta, probable small_launchers, a tank frame set with a 27x16 box.
- Cross-scene probe: every labelled class is located and named at L1 by the Helsinki-only model; L1 recognition is not the bottleneck for the labelled classes.
- Visualizer `elias/visualize_run.py` (H.264 MP4 plus one-file HTML), render queued behind the first half run.
- Public board untouched; two concealed half runs of both_m1280 replace the full public run.
