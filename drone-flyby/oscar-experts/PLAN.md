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

## Open items / decisions to revisit
- Helicopter template is the unreviewed v4 mask (`review_status=claude-auto`).
- Condor full-pixel branch regressed to 12/42 after the part model took heading; comparison branch only.
- Hangar second viewpoint (validation track hangar-053-082, 30 reviewed frames) not yet added.
- Ask Oscar in the morning: draw helicopter and hangar-053 outlines in the editor.
