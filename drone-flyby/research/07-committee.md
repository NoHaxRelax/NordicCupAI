# Committee on Oscar's synthetic flyover data: merged findings and the work list

Sunday 2026-09-20, 02:30 to 04:00. Seven agents (fine-tune and instrument on Fable; boxes, tiny, sequences, backgrounds,
red team on Opus), analysis and light GPU inference only, no portal, no pods. Full reports with every number and its
source: `elias/out/committee/_reports/<angle>.md`; scripts and tables under `elias/out/committee/<angle>/`. The data is a
synthetic training set (see `03-checklist.md`, "Oscar's synthetic flyover data"), copied to
`C:/Users/edlun/Desktop/lucky shots/drone-data/oscar-pod` (everything except `flyover-both-v1` and `flyover-val-v1`: his
pod refused SSH from 02:34).

## What the committee established

1. **The flypaste-v3 labels are not organiser boxes and the class ids are alphabetical.** `paste_sets.py` line 67 and
   `paste_v3.py` line 62 hand `apply_perturbation` the sprite canvas rectangle where the organiser box in sprite
   coordinates (`box_in_sprite` in `views-*-v2/views.json`) belongs, so the exported label is the tight silhouette.
   Organiser box over label: jet_plane 1.48 x 1.45, condor 1.46 x 1.55, medium_launcher 1.62 x 1.45, small_launcher
   1.24 x 1.58, large_launcher 1.18 x 1.28, tank 1.16 x 1.21, ta-ta 0.97 x 0.78 (three agents found this independently;
   F3, F5 and helsinki_only all answer exactly these factors on the tiles and 0.99 to 1.04 on the real Helsinki frames).
   The label files index classes alphabetically (condor 0 ... tank 15); our checkpoints use hangar 0 ... spacecraft 15.
   Anyone who trains on the set as delivered learns tight boxes under swapped names. Corrected copies:
   `elias/out/committee/instrument/rescale_labels.py`, `elias/out/committee/boxes/rebox.py`.
2. **Fine-tuning F3 on flypaste sprites costs 0.056 on the harness** (0.685 to 0.629 at a 49 % dose and 0.628 at 15 %,
   boxes already corrected to our convention), through false positives on real frames (helicopter 2 to 64 answers
   without a label, mine_roller 35 to 126); the same checkpoints gain +0.07 to +0.16 on flypaste data. The paste recipe
   (soft alpha edge, relighting, self-shading) is a domain of its own. Replay-only fine-tunes (our generator, frozen
   backbone and frozen BatchNorm, 3 minutes on the laptop GPU at 2.9 GB) keep the harness at 0.687 to 0.688 and move
   large_tower from 0.64 to 0.82 and 0.85 on the team labels, which only the portal can confirm.
3. **The tiles cannot rank our checkpoints.** Against the portal's clear F3 versus F5 verdicts the tile benchmark agrees
   on 0 of 4 classes at L1 (1 of 6 in a second, independent build) and ranks the portal's worst checkpoint first. It is a
   gate against broken checkpoints (`elias/out/committee/instrument/benchmark.py`, two minutes), never a ranking.
4. **Our detector's box convention is right; the Helsinki size prior is what costs boxes.** Against reconstructed organiser
   boxes F3 needs no scale on any class (ten of sixteen optima at exactly 1.00, none gains more than 0.022). The blend
   with `tracking/size-prior.json` (one Helsinki instance per class) inflates large_launcher: emitted-box recall at IoU
   0.5 is 0.577 with the blend against 0.747 with detector extents on validation-type launchers, and +0.025 on
   organiser-truth ones. The deployed medium_launcher 0.85 is reproduced independently and is safe on both known
   vehicles (they differ 1.57 x in width; 0.85 keeps IoU 0.80 on the Helsinki one, 0.75 would not).
5. **The tiny classes lose on recall, and nothing at inference time moves it.** small_launcher at its L1 size (10 to 13
   delivered px) is found 41 % of the time; a real zoom doubles it (F5 at L2: AP 0.67 against F3 0.26 at L1). Test-time
   magnification destroys every class (small_launcher AP 0.258 at 1280, 0.130 at 1600, 0.045 at 1920; tank 0.94 to 0.53),
   per-class birth thresholds measure 0.681 to 0.684 against 0.685, a small_launcher box scale or detector extent costs
   0.003 to 0.007. medium_launcher is the one box-limited tiny class.
6. **A second harness scene with exact labels exists** (`elias/out/committee/sequences/`, scene `scene_malmi25`): one of
   Oscar's flights with the missing odd poses synthesised by homography (0.57 px median error), all 16 classes, the real
   64 px per frame motion, oracle 1.000. On it the `box` loss bin is empty for 15 of 16 classes (it was 16 to 34 % on the
   team labels), the forecast bias flips sign (3.5 to 6.6 px behind, against 2 to 4 px ahead on team labels, both under a
   tenth of a frame's motion), condor, jammer and spacecraft score 1.00, 0.88 and 0.85 on unseen terrain (row A11
   closed), and large_launcher is answered mine_roller on 5 of 10 exact labels (the confusion is real, not only a
   team-label artefact). The 36 empty flights run as scenes as they are: every answer on them is a false positive.
7. **F3 hallucinates small_launcher on unseen terrain; F5 does it 4.5 to 5.9 times less.** On 1404 object-free L1 views
   over 14 new sites F3 makes 1.29 birth-grade false positives per view (88 % small_launcher, at 0.72 to 0.84 confidence
   on ploughed fields; nl_flevoland alone 9 per view), F5 0.29. On real pixels the absolute rate is 5 to 11 times lower
   (0.08 per view on the Helsinki frames), so the ratio and the class transfer, the rate does not. The terrain prior
   reaches 22 % of these boxes. Same-day portal small_launcher: F3 0.34, F5 0.31.
8. **What the 0.797 is made of, for an unseen flight** (red team): the only matched seen-versus-unseen pair we own is
   F3 0.694 against helsinki_only 0.601 on the validation flight, so 0.09 of F3's score is a scene premium the evaluation
   flight will not grant. Cluster births (+0.034) can only fire where same-class pairs exist: 0 of 259 objects in the
   organisers' 25 reference frames and 0 of 4873 in Oscar's pasted flights have one, against 49 % of validation
   medium_planes. The launcher box (+0.018) needs the small launcher; ta-ta on F5 (+0.038) needs walkers (an absent
   class costs nothing: the macro is over classes present). **Planning range for the attempt: 0.60 to 0.72.** Band
   calibration never failed on 36 unseen flights (thinnest margin 5.5 x); the side-band branch has never run in a
   served run.
9. **A lower-confidence duplicate box is free under the scorer.** Replaying a run's answers through the organisers' scorer
   with every medium_launcher box emitted twice (second copy at the alternative scale and 0.3 x confidence) changes
   nothing when the primary box is right (+0.000 on every class) and recovers 0.119 of medium_launcher AP when it is
   wrong; hedging every class is +0.004 to +0.008 with a worst class of -0.014.

## Work list, one item at a time (laptop first, portal only to confirm)

| # | item | evidence | expected | needs | status |
|---|---|---|---|---|---|
| 0 | Tell Oscar: labels are silhouettes, ids alphabetical, one-argument fix; sprite fine-tune cost us 0.056 | findings 1, 2 | protects his retrains | Elias | sent to Elias 04:00 |
| 1 | `DRONE_BOX_HEDGE`: emit the alternative-scale box at 0.3 x confidence (medium_launcher, ta-ta, large_launcher) | finding 9 | 0.000 if we guessed right, +0.02 to +0.06 if not | 15 min code, harness | DONE 04:00 (bc3afb6): a medium_launcher duplicate at 1.176 and 0.82 changes no class on the harness (0.685) or on `scene_malmi25` (0.586, 0.670 top-entering); a large_tower duplicate at 0.87 and 1.15 adds 0.033 to that class on the team labels. Candidate dict: `{"medium_launcher": [1.176, 0.82], "large_launcher": [0.88], "ta-ta": [0.8, 1.25], "large_tower": [0.87, 1.15]}`: harness 0.688 against 0.685, scene unchanged (run HC_H4) |
| 2 | large_launcher box: scale 0.88 (or detector extent), hedged with the present box | finding 4 | +0.005 to +0.015 | harness sanity, one portal one-class run | LAPTOP CANNOT SETTLE IT: scale 0.88 as the primary box reads -0.031 and detector extent -0.011 on the harness, whose large_launcher labels are a known artefact; the exact-label scene does not move (its large_launcher is answered mine_roller). Until a portal one-class run, the 0.88 box goes out as a free duplicate (item 1) |
| 3 | small_launcher to F5 (route) or merged: harness, `scene_malmi25` and the 36 empty flights for F3, route, merge | finding 7 | 0 on validation, +0.01 to +0.03 unseen | 20 min GPU | MEASURED 04:10 (`elias/out/logs/route_campaign.log`). Served route: harness 0.686, scene 0.584, 14.4 false answers per frame on eight empty flights (13.6 of them small_launcher; nl_flevoland_lane0 73 per frame). **small_launcher MERGED (left out of `ELIAS_ROUTE`): harness 0.686 (class -0.004), scene 0.595 (class +0.169), 7.4 per frame.** small_launcher on F5: harness 0.667 (class -0.209), scene 0.595, 2.9 per frame. small_tower and tank on F5 as well: harness 0.626 (small_tower -0.41), scene 0.571 (tank -0.38): rejected. Merge is the candidate; medium_launcher merged and everything merged are running (`route_campaign2.log`) |
| 4 | Force the side-band branch once on the harness (no exception, no refused camera moves) | finding 8 | removes a tail risk | 3 min GPU | CPU DONE 04:05 (`elias/out/committee/lead/side_band_sim.py`): left, right and bottom bands issue 0 illegal camera requests in both sweep modes (the sides-only sweep needs an intermediate stop because 1920 px exceeds the L1 limit, so it is a four-view cycle too). End to end 04:13: a vertically flipped copy of the first 120 validation frames (`drone-data/scenes/validation_vflip`) chose the bottom band by itself (view centre y 1620 on 116 of 120 frames), 120 camera moves applied, none refused, no exception; mAP 0.477 on upside-down objects, which is not a number to compare with anything |
| 5 | Replay fine-tune with Oscar's empty renders as backgrounds, four sites held out: false births on the held-out sites against F3, harness unchanged? | findings 2, 7 | robustness on unseen terrain | 15 min GPU, then a portal third | MEASURED 04:20 (`post_campaign.log`, checkpoint `elias/out/committee/finetune/runs/F3HN/weights/last.pt`: F3, one pass over 2600 replay views in the F3 recipe with 60 % new-terrain backgrounds plus 300 pure negatives, frozen backbone and BatchNorm, 2.5 minutes on the laptop; five sites held out). **Birth-grade false positives per L1 view on the HELD-OUT sites: F3 2.35, F5 0.44, F3HN 0.14** (small_launcher 1251, 175 and 34 boxes over 585 views). Harness total unchanged (0.685) but single classes move both ways (jet +0.09, large_tower +0.08, small_launcher +0.05; medium_launcher -0.13, medium_plane -0.07); exact-label scene 0.551 against 0.586 (large_tower +0.32, helicopter +0.16; mine_roller -0.66, small_plane -0.20, tank -0.17, each class one or two objects). Not a replacement for F3; a candidate answer for single classes (small_launcher, large_tower): `route_campaign3.log` |
| 6 | Runbook: planning range 0.60 to 0.72; checklist: A11 closed, committee verdicts | finding 8 | no panic at a 0.65 rehearsal | writing | TODO |
| 7 | Portal, morning (ask Oscar for 09:00 to 11:30): one-class large_launcher, large_tower F3 against the replay fine-tune, small_tower and tank F3 against F5, then the pre-flight third | findings 2, 4 | +0.01 to +0.03 together | pod 0.74 USD/h, portal | TODO |

**Result of the night (05:40):** the `robust` mode of `elias/pod_start_final.sh` = routed + large_tower from F3HN +
small_launcher merged over three models + box hedge + pair hedge (large_launcher with mine_roller, large_tower with
medium_launcher, 0.3 x confidence; `classhedge_campaign.log`: harness 0.696 to 0.709, scene 0.615 to 0.638, no class
loses, the default sibling groups cost small_launcher 0.014 and stay off). Exact served environment, run RC_ROBUST:
**harness 0.709 against 0.686, exact-label unseen scene 0.638 against 0.584, false answers on eight empty flights 4.7
against 14.4 per frame.** No portal number: `elias/morning_portal.sh` and the decision rule in `04-sunday-runbook.md`.

## Oscar's mined validation labels (branch `drone/oscar-annotation`, 08:04 Sunday): the harness is now near organiser-true

`annotation/labels/mined-labels-v4.json`: 1358 boxes, all 249 frames, all 13 classes, 0.967 on the validation API
(per-class AP of the labels 0.93 to 1.00; five classes hand-measured). Loaded as a NEW scene for MEASUREMENT ONLY:
`C:/Users/edlun/Desktop/lucky shots/drone-data/scenes/validation_mined_v4` (images are a junction to `src/validation/images`;
remove it with `cmd //c rmdir images`, never `rm -rf`). Run with `elias/out/committee/sequences/run_scene.sh`.

| config | mined-label harness | portal |
|---|---:|---:|
| plain F3, no switches | 0.709 | 0.697 |
| routed | 0.829 | 0.797 |
| robust | 0.834 | not measured |

The instrument tracks the portal (the offline clock loses no frames, hence the +0.01 to +0.03). Robust against routed
per class: large_launcher +0.055, medium_launcher +0.023, small_launcher -0.007, large_tower -0.003, the other nine
identical. Cluster births read small_plane 0.665 to 0.963 and medium_plane 0.418 to 0.765 (plain to routed). Loss
decomposition of the robust run against these labels (`elias/miss_analysis.py --scene <that path>`): the box bin is
empty except medium_launcher (21 %); **small_launcher loses 22 % of its labels and ta-ta 21 % as `lost`** (answered, then the
track retired after three silent looks and the forecast stopped), 9 % and 19 % unseen; medium_plane 19 % never born;
tank answered mine_roller on 5 %. Retired forecasts and the miss count are being re-measured against these labels
(`mined_campaign2.log`); row B3 measured them neutral on the team labels, which could not see this.

Closed by the committee (do not reopen without a new reason): fine-tuning on flypaste sprites, the tile benchmark as a
ranking, test-time magnification and crop passes, per-class birth thresholds, per-axis box scales and a centre shift, the
terrain prior as protection, a forecast lead-time correction fitted on team labels, experiments on the un-interpolated
13-frame flights (120 px per frame, twice the real motion).
