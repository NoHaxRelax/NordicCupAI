# Drone flyby: the checklist (living document)

Deadline Sunday 2026-09-20 16:00 CEST; top five submit code by 20:00. One evaluation attempt, only Elias queues it.
Every item is measured the same way: concealed validation runs, one class per run (score x 13 = that class's AP)
and the three thirds (0:83, 83:166, 166:100000, which add up to the full score). Public best was 0.511 until Oscar's full validations from his Swedish pod on Saturday 16:30 to 16:47 scored 0.705 to 0.716
(team best now 0.7162, visible to every team); his runs take the single per-team portal slot every two minutes. Update the status column as things happen; do not delete rows, rejected ideas stay so nobody
repeats them. Started 2026-09-19 13:25 by Claude at Elias's request.

Status codes: TODO, DOING, DONE, REJECTED (with the number that killed it), BLOCKED (with what on).

## A. Detector and data

| # | item | expected | hours | status | evidence |
|---|---|---|---:|---|---|
| A1 | Retrain the F3 recipe with the unlabelled validation objects fixed: keep-out zones for 19 tracks, 42 new sprites (18 leaning ta-ta, 18 second small_tower, 6 second medium_launcher) | ta-ta 0 to about 0.5, small_tower and tank back to 0.86/0.93, medium_launcher up; +0.05 to +0.08 on validation | 3 | F4 (laptop, batch 2) REJECTED: thirds 0.235+0.227+0.068 = 0.53 against 0.63 for the deployed model on the same laptop; large_launcher confidence 0.93 to 0.39, small_launcher boxes 173 to 46 per run. Batch 2 starves the normalisation layers. F5 = same data at batch 16 on a Secure 4090 pod, TRAINED 15:20 (checks 0.867 0.906 0.918 0.922 0.915). Per-class from Oslo (score x 13): ta-ta 0.50, small_tower 0.89, tank 0.93, medium_plane 0.53, large_launcher 0.58 (run had 3333 ms timeouts on frames 96 and 106), small_launcher 0.31, medium_launcher 0.09. Against the deployed F3 (night pod): +0.50, +0.09, +0.06, +0.03, -0.14, -0.25, -0.05. Thirds running 16:07 | F4 weights `elias/out/weights/F4_fixed_m1280.last.pt`. Laptop per-class so far: small_tower 0.93 (F3 0.80). ta-ta 0.00 with the size prior: the tracker replaced the detector's box with the flat Helsinki prior (31x25 on a 19x33 walker); rerun with `DRONE_CLASS_EXTENT` detector pending. medium_launcher 0.11 with detector extents (F3 0.14): the loss is false positives on dark blobs, see B1. Laptop runs score about a tenth under pod runs (0.627 vs 0.694 for the same model) |
| A2 | P2 detection head (stride 4) on the same data, YOLO26m-p2 at 1280 | small_launcher and ta-ta at L1 (10 to 15 px delivered) | 2 + 1.5 h GPU | TRAINED 15:40 on pod 4cezfts69abgzf (F7_fixed_mp2_1280, batch 8, pretrained from yolo26m): checks 0.757 0.876 0.885 0.888 0.893, the lowest of the four; portal measurement queued after F6 | VisDrone standard trick: arXiv 2512.07379, SAHI arXiv 2202.06934 |
| A3 | Large backbone (yolo26l) on the FIXED data | unknown; lost to medium on the old data (0.39 vs 0.42) | 0.5 + 2 h GPU | TRAINED 15:31 on pod z02n16fwh2u1ob (F6_fixed_l1280, batch 8): checks 0.892 0.897 0.900 0.922 0.916; portal measurement queued after the F5 thirds | Elias asked for the big-model check |
| A4 | Spend the latency headroom: 1536 input, flip averaging, two-checkpoint ensemble (`elias/ensemble.py`) | +0.02 to +0.05 at detector level | 2 | REJECTED 17:18 on the harness: 1536 gives 0.582 and 1024 gives 0.556 against 0.600 at 1280 (F3 was trained at 1280); flip and scale augmentation (`ELIAS_AUGMENT=1`) gives 0.576 with medium_launcher falling to 0.08 | 110 ms of a 300 ms budget used on an RTX 5090; laptop 85 ms at 1280 |
| A5 | DINOv2 prototype verifier (DE-ViT style): one prototype per class from the sprites, re-rank YOLO boxes by similarity, never delete | +0.02 like the CNN verifier, maybe more on sibling confusion | 3 | TODO | github.com/mlzxy/devit (MIT, weights). The CNN verifier relabelling HURT (0.329 to 0.258): re-rank only |
| A6 | ZoomDet learned warp inside the view | +8 mAP on SeaDronesSee in the paper | 6 + training | REJECTED for this deadline | arXiv 2602.07512; too new to trust for one attempt |
| A7 | Soft blending of sprites | | | REJECTED | 0.316 to 0.268: the real frames are hard-pasted renders |
| A8 | Free rotation of tall objects | | | REJECTED | costs 0.06; lean limits stay |
| A9 | Train on the real validation frames | would push the board to about 0.9 | 2 | REJECTED by Elias's rule | no evaluation gain, reveals our strength |
| A10 | Second small_launcher sprites from the unlabelled ones | | 1 | BLOCKED | GrabCut returns slivers on 25 px objects; needs a manual mask or a bigger box recipe |
| A12 | Shadow-free walker sprites: the 18 ta-ta sprites include the cast shadow, so the detector's walker box is 31x56 on a 19x33 body; re-cut with `--grow 0.05 --no-growth --suffix=-tight` gives 12 to 16 x 29 px bodies (`elias/out/extra_sprites_tata_tight.jpg`) | ta-ta boxes match the organiser convention if it excludes shadows | 0.5 + 1.6 h GPU | TODO, decide after the ta-ta rerun | which convention the truth uses is unknown; the replay with three box sizes scored AP 0.74 |
| A11 | Jammer, condor, spacecraft never measured on a real unseen instance | unknown | | BLOCKED | absent from validation; only Helsinki (training set) has them |
| A13 | Terrain-aware pasting: the generator refuses paste positions whose surroundings are water or forest (`SYNTH_TERRAIN=1`, `elias/data/terrain.py`) | fewer false positives on dark blobs in trees (the medium_launcher loss); the detector stops learning objects in forest | 0.5 + 25 min GPU | IMPLEMENTED 15:15; F8_terrain_m1280 TRAINED 15:49 on pod 1: checks 0.879 0.896 0.914 0.912 0.916 (F5 0.915); portal measurement queued after F7 | `elias/backdrop_study.py`: random ground 24 % forest and 15 % water; objects 6 % tree cover and 0 % on water (the water hits are a beach next to water); `elias/out/backdrop_sheet.jpg` |

## B. Answer policy (what we emit)

| # | item | expected | hours | status | evidence |
|---|---|---|---:|---|---|
| B1 | Sibling-class hedging: for launcher, plane and tower detections also emit the sibling classes at lower confidence | confusion stops being a miss; absent classes cost nothing, low-confidence extras cost almost nothing | 1 | IMPLEMENTED; first test on the regressed F4 gave no gain (medium_launcher 0.083 with hedging against 0.11 without; medium_plane 0.0, run possibly void by CPU load). Harness on F3 17:10: factor 0.3 gives 0.604 against 0.600, all of it mine_roller (0.41 to 0.46, the tank sibling), small_launcher -0.01; factor 0.6 pending | scorer analysis in `research/02-night-report.md`; test with one-class runs for medium_launcher, medium_plane, large_tower |
| B2 | Confidence floor and thresholds (`DRONE_CONF=0.05 BIRTH 0.25 UPDATE 0.15`) re-tuned for the new model | +0.10 was the gain of lowering them the first time | 1 | DONE 17:07, keep the defaults: harness 0.600 at conf 0.03, 0.05 and 0.10; birth 0.20 gives 0.597, 0.35 gives 0.582; update 0.10 gives 0.597, 0.25 gives 0.584 | harness = team labels, offline clock; a gain over 0.01 earns one concealed third on the portal |
| B3 | Emit forecast boxes for retired tracks at a low confidence instead of dropping them | small | 1 | NEUTRAL 17:24 on the harness: 0.600 at 6 and at 12 ticks (0.3), exactly the baseline, although 201 extra boxes reached the answers; the retired tracks are objects already gone or false ones. Stays off | low-confidence extras are nearly free |
| B5 | Context prior at runtime: boxes whose surroundings are water or forest keep their place at confidence x `ELIAS_CONTEXT` (never deleted) | precision on the classes with forest false positives | 0.5 | NEUTRAL 17:17 on the harness: 0.601 at 0.5 and at 0.3 against 0.599 through the same hook; the launcher false alarms sit on grass and bare ground ('other' 12, 'grass' 10, 'sand' 1 of 23 checked), which the prior does not touch | same study |
| B6 | Alternative-label emission: a whole detection of another class over an existing track is also emitted under its own label at confidence x `DRONE_ALT_SCALE` (the track keeps its birth class; before, the detection was dropped as a conflict) | medium_plane (found in 52 % of its harness frames, another class emitted there in 41 %), mine_roller (63 % / 37 %), large_launcher (43 % / 54 %) | 0.5 | IMPLEMENTED 17:12 in `tracking/revisit.py`; harness runs at 0.5, 0.8, 1.0 queued (`harness_campaign7.log`) | the split per class (found / IoU 0.5 / other class there) is in the session log; medium_plane portal AP 0.50 matches a 50 % class lock |
| B7 | Vote-based relabel: after k consecutive whole detections of one other class over a track, the track takes that class and box (`DRONE_RELABEL_VOTES`) | the same class lock as B6, carried into the forecast frames | 0.5 | IMPLEMENTED 17:15; harness runs at 2 and 3 votes, alone and with B6 at 0.8, queued (`harness_campaign8.log`) | the night's verifier relabelling hurt (0.329 to 0.258) because it relabelled on one CNN opinion; this needs k consecutive detector calls |
| B4 | Per-class routing across several models: each class answered by the model that measured best on it (`ELIAS_ROUTE` in `elias/ensemble.py`, `DRONE_DETECTOR=elias.ensemble:build`) | takes every per-class win, e.g. F4's small_tower 0.93 | 1 + measurements | IMPLEMENTED 14:40, untested; needs per-class portal numbers for every candidate model | Elias's reframe: bandwidth is not a constraint on a pod, about 25 ms per model on a 4090 |

## C. Camera policy

| # | item | expected | hours | status | evidence |
|---|---|---|---:|---|---|
| C0 | Per-class box extents: `DRONE_CLASS_EXTENT` detector for ta-ta (leans) and medium_launcher (validation instance smaller than Helsinki's); the size prior stays for the rest | ta-ta from 0 | 0 | measured on F4: ta-ta still 0.0 with detector extents, because the learned box included the shadow (31x56); medium_launcher 0.11. Keep the switch, re-measure on F5 which has shadow-free walker sprites |
| C0b | medium_launcher box size: on the harness the deployed model's launcher box is 52 x 52 on a 30 x 45 labelled object (centred within 4 px), so 18 of 31 right-class answers land at IoU 0.3 to 0.5 and count as misses; the same shape check shows jet_plane 85 x 88 against 75 x 71 and medium_plane 60 x 55 against 45 x 58 (small_tower, tank, hangar sizes are the team's off-convention labels, ignore) | medium_launcher from 0.14 towards 0.5 if the organiser box is the tight vehicle | 0.5 | HARNESS WIN 17:26: `DRONE_BOX_SCALE='{"medium_launcher": 0.75}'` lifts the harness from 0.600 to 0.640, all of it medium_launcher (0.288 to 0.733), every other class unchanged; 0.65, 0.85, the per-axis [0.62, 0.88] and the other classes are running; the portal one-class run from the laptop is queued (`portal_session.log`) | the shape table is in the session log; the 36 'background' launcher answers at x about 3000 are the unlabelled second launcher (keep-out zone), not false positives |
| C1 | Zoom to L2 on cue: sweep at L1; when a detection is small or ambiguous, spend the next frame at L2 on its forecast position, then return | simulator 0.82 to 0.93 against 0.75 for the sweep at a 16 px recogniser; the only route to ta-ta (0 of the L1 looks, 22 to 26 L2 hits) | 4 to 6 | IMPLEMENTED 16:35 (`DRONE_CUE_EVERY/_PX/_CONF/_COOLDOWN/_KIND/_CLASSES`, `cue_track` in `tracking/workflow.py`), REJECTED for this deadline on the local harness (team labels, F3 or the routed F3+F5 mix, offline clock): baseline 0.600; cue every 4 on any unconfirmed, small or weak track 0.548 (53 cues); every 2 0.461; unconfirmed tracks only every 6 0.576 (21 cues); ta-ta transients only every 4 0.563 with just 6 cues. The loss sits in small_launcher (0.55 to 0.20 to 0.31) and small_tower: each native detour costs the band two to three frames and the small launchers in the band lose their births (12 against 16). The ta-ta gain cannot show on the harness (no walker labels) and is at most +0.3 on one class (+0.02 overall), below the measured cost | `elias/policy_sim.py` policies `cued`, `cued_lcr` promised 0.82 to 0.93 against 0.75 with an idealised recogniser; the real tracker does not relabel a track from an L2 look (a different-class detection over a track is dropped as conflicting), so a cue can only birth, confirm or refine boxes |
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
| D7 | Stale-frame guard: a request older than the newest processed frame is answered empty without touching the tracker | avoids 'out-of-order frame' failures that reset the whole workflow after three | 0.5 | IMPLEMENTED 2026-09-19 13:50 in example.py, untested on the portal | seen in the voided F4x ta-ta run: four such failures |
| D3 | Rehearsal on the pod: one concealed half run of the final weights from the pod before the attempt | 0.5 | TODO Sunday | `elias/pod_serve.sh`, `elias/pod_portal.sh` |
| D4 | Pick the final weights and settings, write them in `elias/README.md` deploy block | 0.5 | TODO | |
| D5 | Elias queues the evaluation attempt; nobody else, no agent | | TODO Sunday | hook `.claude/hooks/block-evaluation.py` |
| D6 | Terminate every agent pod afterwards, check `list-pods` | | TODO | |

## Pods live on 2026-09-19 afternoon (terminate when done, D6)

- y8fvk8xgo7qhuy elias-drone-retrain-f5b, 47.47.180.99:14002, trained F5 and F8; kept for a possible retrain (data generated, share tunnel)
- sue6qz4ml2mzxc elias-drone-serve-eu, 149.36.0.35 ssh 18676, endpoint 9053 -> public 18677; every measurement and the Sunday attempt are served from here
- terminated: z02n16fwh2u1ob (F6, 16:30) and 4cezfts69abgzf (F7, 16:15) after their probes; the weights are in elias/out/weights/ and on Oslo

## Same-day measurement rule (2026-09-19 16:30)

The deployed F3 measured small_launcher 0.34 from Oslo against 0.56 in the night run, with byte-identical code, the band at the top, no stale frames, and large_launcher unchanged (0.74 against 0.72). Whatever moved (portal truth, host, queue) is outside our code, so every model comparison uses same-day Oslo runs only. F5 thirds from Oslo: 0.315 + 0.196 + 0.103 = 0.614. F3 thirds from Oslo: 0.262 + 0.302 + 0.114 = 0.678 (night pod 0.694). F5 wins the first third by 0.053 (the walkers) and loses the middle third by 0.106 on classes not yet measured one by one, so the deployment is F3 for every class plus F5 for ta-ta, extended only where a same-day one-class run beats F3 by more than 0.03 (`elias/route_from_log.py` builds ELIAS_ROUTE from the log; `elias/out/logs/measure_route.sh` measures the routed thirds).

## Laptop harness, full pipeline per checkpoint (team labels, 11 classes, offline clock, 2026-09-19 17:00)

| checkpoint | mAP | where it differs from F3 (0.600) |
|---|---:|---|
| F3 `both_m1280.pt` | 0.600 | hangar 0.95, heli 0.88, jet 0.74, s_tower 0.74, l_tower 0.64, s_launcher 0.57, m_plane 0.49, tank 0.47, l_launcher 0.43, mine 0.41, m_launcher 0.29 |
| F5 corrected data | 0.532 | jet 0.83, l_tower 0.72, m_plane 0.54, m_launcher 0.35 up; s_tower 0.33, s_launcher 0.32, tank 0.41, mine 0.24 down |
| F8 terrain pasting | 0.542 | tank 0.52, mine 0.44 up; s_tower 0.35, s_launcher 0.34, hangar 0.85 down |
| F6 yolo26l | 0.551 | m_launcher 0.45 up; s_launcher 0.28, tank 0.36 down |

The harness's small_tower and tank labels are off the organisers' convention (the portal says F5 beats F3 on both), so
cross-checkpoint routing still comes from same-day portal runs; the harness ranks policies on one checkpoint.

## Notes from Elias and Oscar (2026-09-19 15:00)

- Mixture of experts is the deployment model: one detector per class is fine, bandwidth is not a constraint on a pod. B4 implements it.
- Helicopter: measured 0.96 AP on the portal for the deployed model last night (F3 per-class table in `02-night-report.md`), so on our pipeline it is not a weak class; Oscar's concern may be his own pipeline or the evaluation flight's different look (the Helsinki-trained window classifier scored 0 on validation helicopters before sprites from both scenes were used).

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
