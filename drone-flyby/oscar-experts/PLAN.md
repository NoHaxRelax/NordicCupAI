# Overnight plan, 19 Sep 2026 (Oscar asleep; Claude works alone)

Re-read this file at every loop wake-up. Update the status column as stages complete.

## Rules from Oscar (binding)
- Training-split data only for everything until the experts are good; then a little validation, carefully
  (his validation labels are quick, incomplete, boxes may be off).
- Never run an organizer EVALUATION attempt. `drone/portal.py` refuses non-validation routes. Validation
  attempts: as many as needed, one at a time.
- Nothing runs on the laptop: only file edits, git, ssh/scp, and the small portal client. All compute on
  Runpod. Stop pods I am not using; boot what I need within the $100 night budget.
- Experts are candidate generators: recall first, never rule out on appearance, but not hundreds of
  candidates either. A trained verifier rules out.
- Overfitting to the exact 3D models is intended; tolerate perspective and mild lighting change.
  Perturb sprites mildly (stretch, lighting, partial shadow) when making synthetic data.
- Keep it explainable and simple; refine the simple setup rather than piling on.
- Synthetic data: the Higgsfield "sprite blender" agent creates composites from chosen sprites and
  frames on request (Oscar). Existing sets: `data/drone/synthetic-sprites-20260918-v1` (531),
  `synthetic-validation-sprites-20260918-v1`, `artifacts/drone-localized-20260918-v1/composites/*`.
- Earlier YOLO26x (`mypc-yolo26x-20260917-v2`, 118 MB, SHA 14f34b0b…) was strong on the small launcher;
  consider reusing it for that class. Checkpoint lives on mypc, not here.
- Push milestones to team repo branch `drone/oscar-experts` (sparse clone in the scratchpad,
  based on `drone/oscar-live-tracker`), folder `drone-flyby/oscar-experts/`.

- Iteration speed (Oscar 01:20): waiting on inference is waste. First parallelise on Runpod (more pods or
  more processes); if waits still dominate, rewrite hot paths in C++/Rust (experts are untrained, so they are
  candidates). Currently OpenCV matchTemplate carries the load (already C++); a full 16-class wave takes
  ~15 min eight-wide, so not yet.

- Local tests (Oscar 03:00): run a rotating stratified SAMPLE of tiles, not everything, and change the sample
  each iteration so nothing tunes to fixed frames. `evaluate_all.py --sample N --seed K` + proposal cache.

## Stages, in Oscar's order
| # | Stage | Status | Where |
|---|---|---|---|
| 0 | Feasibility: API key present, pod 1 running, evaluation guard, validation attempt submitted through the pod tunnel (uuid 3c693aeb, 00:22) | done 00:25 | `drone/portal.py`, pod `ypuawkayl3px8t` |
| 1 | Experts for all 16 classes, tested on training tiles | waves 1-2 done; gated pass done 03:35 (RESULTS.md). Pass 3 running 03:50: full training set, 12 shards, current code -> gates v3 fitted on its own candidates -> `verify_pass.sh` on a rotating sample with the verifier | `/workspace/experts/runs/train-2213` |
| 1b | Iterate on weak classes | done for launchers (blob expert: 6/6 at L1/L2 vs 2/69 before). Key finding 01:05: per-class logistic gate over recorded features (`fit_gates.py`) gives out-of-fold recall 98-100% while keeping <1% of background for jammer, small plane, small tower, spacecraft, large tower (helicopter 14%) | `fit_gates.py`, `gates-v1.json` |
| 2 | Verifier v0 trained 02:40 (`verifier-2304/model/best.pt`): 21,915 crops, held-out tiles 99.7%; evaluate-only synthetic set with dev backgrounds: accuracy 95.1%, background rejection 98.4%, weakest jet .70, ta-ta .78, medium plane .78 | `run_verifier.sh`, `eval_verifier.py` | `crops.py`, `train_verifier.py` |
| 3 | Boxes: expert pose + organizer offset rotated with the pose; tall objects via tracker placement | measured 02:20: median IoU .81-.94 for 11 classes, ta-ta .64 (see RESULTS.md) | `live_detector.py` |
| 4 | Camera strategies: local replays on the validation scene with `run_local_eval.py` (l1 sweep vs l2_top vs revisit), then real validation attempts | chained by `after_pass3.sh` (04:25): verify3 -> four replays A-D with gates v3 + verifier v0 -> verifier v1. Then `stage4_status.sh`, pick the best recording, `stage4_submit.sh` + `validate_endpoint.sh` for the real validation score | pod `/workspace/live/drone-flyby`, `validate_endpoint.sh` |
| 5 | Use API-verified validation labels and Higgsfield synthetic data to tailor experts/verifier; watch overfitting | pending | |
| 6 | Push branch, write handoff, update memory | milestone 1 pushed 01:45 (`508ae5f` on `drone/oscar-experts`) | `push_branch.sh` |

## Pod and paths
- Pod 2 `c28v8wlbm25ots` (oscar-claude-experts-2, A100 EUR-IS-1, $1.59/h, created 05:01) for replays and speed work; cloned from pod 1 with `clone_pod.sh` (pod 1 holds a transfer key). Stop it when idle.
- Pod 1 `ypuawkayl3px8t` A100, `root@157.157.221.29:17494`, helper `pod.sh`. Code+data `/workspace/experts/project`,
  runs `/workspace/experts/runs`. Live endpoint stack `/workspace/live/drone-flyby` (scene `validation` linked to
  the 249 reconstructed frames). Two idle servers from the earlier session (api.py 9305/9306 + cloudflared) hold
  28 GB GPU memory; kill them when the GPU is needed.
- Uploads so far: expert bank, grid384 tiles, v4 sprite library, reviewed bank, grid-comparison/256 manifest.
  `upload2.sh` done 00:30: synthetic sets + reference frames on the pod. `synthetic-validation-sprites` is evaluate-only (dev backgrounds).
- Laptop uplink ~0.5 MB/s: avoid large uploads.

## Speed work (03:10)
- `gpu_proposer.py`: one scene FFT per view for all classes; exact match with the CPU proposer; 3 s for 12
  classes on a 1920x1080 view vs 7 s per class on CPU. `evaluate_all.py`: one process, shared proposer,
  threaded fine pose, on-disk proposal cache keyed by kernel signature, rotating stratified sample.
- Remaining cost: fine pose ~1-5 s per class per view on CPU threads; condor (4 s) and hangar (3 s) are the slowest experts.
  Next cycle, BEFORE a gate-fitting pass: condor max_candidates 16->8, hangar one template per zoom and fewer angle offsets,
  vectorise PartModel.score across poses. Never change candidate features between fitting gates and applying them.

## Root causes found 05:20-05:50
- Proposal heading was the fitted bar angle of the rotated mask, not the rotation applied; square objects flipped
  90 degrees and every box landed off the object at L0/L1 (pass 4 collapse). Now heading = applied rotation.
- Template selection by largest mask dropped the reference tank sprite from the sweep; zoom-matched sprites can
  be poor views (frame-24 tank .39 vs frame-4 .86). Now: same zoom first, then other zooms' reference sprites.
- Efficiency work is owned by the speed session (nordic-ai-cup-2026-07) from 05:35; its exact-output batches
  land in artifacts/drone-speed-20260919/READY/ and are applied between passes.

## Root cause of passes 4-6 (06:30): shared proposer used the first class's scene settings
- `SharedProposer.propose_all` blurred/downscaled the scene with `rows[0]`'s settings (condor: downscale .5, blur 1.0)
  and correlated every class's full-resolution kernels against it. Once condor joined the shared proposer (pass 4)
  every generic class lost its true peak (tank .77 -> .69, true location gone from the top 24). Single-class spot
  checks could not see it. Fixed: one scene per (downscale, blur) group; verified single == all16 == CPU on three
  tank tiles. Proposal cache now keyed on PROPOSER_VERSION; stale caches on both pods deleted.
- Lesson: any shared/batched replacement must be checked against the per-class path on the SAME multi-class run.
- `evaluate_all.run_expert` now catches expert exceptions per (class, tile) and records them instead of killing a shard.
- Passes 4, 5 (pod 2) and 6 (pod 1, one shard crashed on an IndexError in colour_ncc) are invalid; pass 7 = first valid
  full pass with the fixed proposer + cross-zoom templates.

## Template choice is not the mine-roller problem (06:55)
- Four template orderings (same-zoom / cross-zoom for proposer and fine pose) give identical recall on a random
  sample of 7 classes x 18 tiles. The mine-roller "2/6" spot checks sampled tiles of track mine-roller-a-005-009,
  which Oscar's label review excludes (14 of 19 mine-roller records per zoom); the passes skip those tiles.
  Nothing to fix there; the expert does put a box on that instance but at IoU .37-.39 (different view, box too large).
- Fair-share proposal capping (`GenericExpert._fair_share`: merge per template, interleave by rank, cap) is under
  test as V4; kept only if it changes recall.

## 07:55 status
- Pass 7 valid (tables in RESULTS.md); gates v7 at runs/pass7/gates.json (pods 1 and 2). Milestone 5 pushed (8acde64).
- Pass 8 running on pod 1 (fair-share capping, large-launcher sprites, size refinement) -> gates v8.
- Verifier v1 training on pod 2 from pass 7 candidates (55.7k real crops, 98% background, + 2.1k synthetic).
- Stage 4 replays A-D running on pod 2 with gates v7 + verifier v0 (offline local evaluator, timeout lifted, so the
  proxy mAP measures coverage/accuracy, not speed). Rerun with gates v8 + verifier v1 once both exist.
- Box audit on pass 7 (box_stats.py): only medium_plane L0 would gain from a size calibration (.38 -> .88 with width
  x.68); other misses near targets are pose/position errors. Re-audit on pass 8 (size refinement) before deciding.

- 08:05 replays OOM'd (42 GB per live-view process: 48-kernel FFT chunks + kernel cache at 2160-px scene size).
  `SharedProposer` now sizes the chunk by scene area and caches only tile-sized scenes; a replay process is ~2 GB.
  Four replays + verifier trainer run together on pod 2. Speed session informed (twice).

- 08:25 stage 4 blocked on speed: the organizer's local evaluator is clock-driven (frame i exists at start + i x 333 ms;
  `--eval-timeout-s` only lifts the per-request timeout), so at 40-100 s per view almost every frame is skipped and the
  tracker raised "out-of-order frame". Replays A-D stopped. Strategy replays resume when a view runs in a few seconds
  (speed batch 1 applied 08:25: live 127 s -> 45 s per 4 views, identical outputs; batch 2 targets the fine pose).
- Verifier v1 (pass 7 candidates): 99.8% held-out tiles, 96.4% / 99.3% rejection on the synthetic validation set.

## 09:30 status: A/B passes running
- Pass 9 (pod 1): final code (speed batch 1, fair-share, launcher sprites 6, size refinement margin .06), old bank.
- Pass 10 (pod 2): same code + `add_track_sprites.py` auto sprites for the uncovered tracks (24 rows kept: launcher
  c-047-078 f47, medium-plane d/e f66, tank-015-036 f15/f36, tank-d f30/f61, jet-plane-c f66; helicopter autos dropped
  as junk, large-tower-038-067 mask failed) + template caps 6 for tank/medium_plane/large_tower/helicopter/jet_plane +
  ta-ta without refinement. Bank backup on pod 2: /workspace/experts/bank-backup-prepass10. Review sheet for Oscar:
  artifacts/drone-experts-overnight-20260919/auto-sprites-review-sheet.png.
- Whichever wins becomes the deployed bank; then gates + verify with matching code, then verifier v2 on its candidates.

## 10:45 validation plan (Oscar via the babysitter session: test regularly against the validation API; evaluation never)
- Correction: the organizer's local evaluator is clock-driven only with `--realtime`; the default is lockstep, so an
  offline replay records every frame. The earlier frame-order errors came from stale evaluators on the same ports.
- Recordings V-A (l1 sweep + overview), V-B (l2 top), V-C (l1 + revisit 3), V-D (l2 + revisit 3) running on pod 2
  with the final bank, gates v9, verifier v1, speed batches 1+2. Each recording is then served by `replay_server.py`
  behind a quick tunnel (`stage4_submit.sh NAME logs/NAME PORT` on pod 2) and submitted from the laptop with
  `validate_endpoint.sh NAME` (POD_PORT=12630 default; receipts under validation-attempts/). One active attempt per
  team: check `portal.py status` first. Track scores per configuration in RESULTS.md.
- Pass 11 (final bank + final code, native) on pod 1 after pass 11d (delivered-resolution recall check) finishes;
  gates v11 -> verifier v2 on pass 11 candidates -> re-record with v11/v2 and resubmit.

- 11:30 speed batch 3 applied: process pool for the live per-class stage (DRONE_EXPERT_PROCS=16; recordings restarted
  on it) and the pose-cache reproducibility fix (warp at the rounded key; changes 2 of 2,675 rows; gates refit at the
  next pass). Recordings V-A..D were crawling (6-13 frames in 20 min) while sharing pod 2 with pass 11d.

- 12:10 recordings: four at once on pod 2 crawl (L0 frames 48-66 s under contention; the l1-sweep strategies are half
  L0 frames). A and C stopped; B (l2 top) and D (l2 top + revisit 3, the shape of the team's 0.51 run) continue alone
  and get submitted first. A/C rerun later, on delivered resolution if pass 11d shows L0/L1 recall holds.
- Pass 11 = deployed config, milestone 6 pushed (98fedfb). verify11 + verifier v2 running on pod 1.

## 13:55 validation loop running
- API score 0.239 for recording D (l2 top + revisit 3, gates v9, verifier v1); recording B submitted; improved
  recordings W-A (pod 1) and W-D (pod 2) in progress with gates-t2, verifier v2, helicopter scales (1, 1.5, 2) and the
  batch-4 GPU windows; they get submitted when done.
- Bank: large-tower-038-067 added as a box-mask sprite (auto mask failed; background included, verifier prunes);
  bank now 89 sprites on both pods. Validation-scene small towers and the mine roller (excluded track) have no
  training-split sprite: not addressable under the training-only rule.
- Biggest remaining levers: camera coverage (l1 sweep + overview vs l2 top), false boxes for launcher/ta-ta/condor,
  and per-class resolution for speed (table in RESULTS.md).

- 14:30 submissions now go through the Runpod HTTP proxy (port 19123 on both pods, `validate_proxy.sh`); quick
  tunnels dropped traffic (attempt B: one request, 0.0056). Attempt D (tunnel, frames 1-8 lost): 0.239.

- 15:00 speed batch 5 applied (pinned pool, vectorised fair-share, DRONE_EXPERT_ROUTING per-class resolution; example
  routing file: native L0 for ta-ta, small_plane, jet_plane at artifacts/.../routing-L0-native-planes.json). Live idle A100:
  L2 2.1 s, L1 6.5 s, L0 5.7-7.9 s. First improved recording W-D scored lower on the proxy (.133 vs .158): the
  per-sprite gates or verifier v2 halve tank recall on validation-scene tanks; attribution recordings running.

- 15:40 speed batch 6 applied (hangar GPU windows: L1 live 6.5 -> 4.6 s). Validation transport is still the blocker:
  through the Runpod proxy the organizer's client delivered one request in three of four attempts (W-D twice, B once)
  while a real-time evaluator from pod 1 and laptop bursts arrive fine; scores from those attempts (0.066, 0.019) are
  not the recordings'. Only two attempts served a full run so far: D via tunnel 0.239, B via proxy 0.221.

- 16:25 speed batch 7 applied (pipelined proposer bit-identical: tile 0.41 -> 0.24 s, L1 3.2 -> 2.1 s; proposer->expert
  streaming; GPU-window failures now fall back to the CPU per call, so concurrent GPU recordings are safe again).
  API: W-A 0.150 (201/249 frames delivered); attribution recordings rerunning one per pod (X-D-v9v1 pod 2, X-D-gt2v1 pod 1).

- 16:40 NVIDIA MPS enabled on both pods (speed batch 8, `/workspace/experts/mps_on.sh`, idempotent; stop with
  `echo quit | nvidia-cuda-mps-control`). GPU processes started from now share the GPU concurrently (single recording
  L1 6.0 -> 2.8 s). Caveat: a hard GPU fault in one client can abort the others.

- 16:55 speed batch 9 applied (pose-cache caps 6000/4000 via DRONE_EXPERT_POSED_CACHE; identical rows; L2 0.71 s).
  Deployed pair for submissions is gates v9 + verifier v1 (attribution: per-sprite gates -.008, verifier v2 -.017 on
  the proxy). Y-A (l1 sweep, that pair, batch 7-9 code, MPS) recording on pod 1 at ~2.7 s per frame.

- 17:50 submissions go through POD 1's proxy only (ypuawkayl3px8t-19123): pod 2's proxy delivered one request in
  four of six attempts even with no-store/close headers, pod 1's served every attempt (W-A 202, Y-A 222 frames).
  Recordings made on pod 2 are copied to pod 1 (local.jsonl) and served there. Y-C submitted that way.

## 18:55 status
- Deployed pipeline: pass 11 experts + final bank, gates v9, verifier v1, helicopter scales, L0 at half scale, speed
  batches 1-9, MPS. Best local proxy 0.284 = L1-only sweep (no L0 overviews); its API score 0.235 with 191/249
  frames delivered (organizer pacing loses 12-25% of frames on recorded replays; about half the attempts stop after
  frame 1 on the organizer's side, so `validate_retry.sh` resubmits until a full run).
- API table: V-D 0.239, Y-A2 0.235, V-B 0.221, W-D 0.215, Y-A 0.203, W-A 0.150. Milestone 8 pushed (ffa2f4f).
- Running: Y-C2 (L1-only + revisits) and Y-A5 (min conf .15) on pod 1, Y-A6 (L1-only, half-height band) on pod 2.
- Neutral so far: helicopter scale sweep, birth confidence .3. Harmful on the validation scene: per-sprite gates,
  verifier v2, L0 overview frames.
- Open for Oscar: the organizer's camera limits (L1 step 1102 px, L2 551 px, no L0->L2) need a legal sweep in the
  live policy; sprite coverage of validation-scene tracks (large launcher c-047-078, small towers) is the biggest
  detector gap and cannot be closed from training data alone.

- 19:40 API noise: the same recording (Y-A2) scored 0.235 then 0.209, tracking delivered frames (191 vs 180). Treat
  API differences under 0.03 as noise; rank configurations on the local proxy and confirm the top one with repeats.
  Tracker knobs (birth, update, confidence floor) and the band height are settled: L1-only top-band sweep, 0.284.

## Open items / decisions to revisit
- Helicopter template is the unreviewed v4 mask (`review_status=claude-auto`).
- Condor full-pixel branch regressed to 12/42 after the part model took heading; comparison branch only.
- Hangar second viewpoint (validation track hangar-053-082, 30 reviewed frames) not yet added.
- Ask Oscar in the morning: draw helicopter and hangar-053 outlines in the editor.
