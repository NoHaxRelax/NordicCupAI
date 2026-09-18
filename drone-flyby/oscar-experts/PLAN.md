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
| 1 | Experts for all 16 classes, tested on training tiles | run `train-2231` (00:31): 11 classes done; launchers re-run with the new blob expert; tank/jet/condor/large_launcher finishing. `finish_stage1.sh` chains summary -> gates -> verifier | `/workspace/experts/runs/train-2213` |
| 1b | Iterate on weak classes | done for launchers (blob expert: 6/6 at L1/L2 vs 2/69 before). Key finding 01:05: per-class logistic gate over recorded features (`fit_gates.py`) gives out-of-fold recall 98-100% while keeping <1% of background for jammer, small plane, small tower, spacecraft, large tower (helicopter 14%) | `fit_gates.py`, `gates-v1.json` |
| 2 | Verifier: harvest crops from expert runs + synthetic composites, fine-tune ResNet-18, presence+class; evaluate on the evaluate-only synthetic-validation set | chained after stage 1 (`run_verifier.sh`) | `crops.py`, `train_verifier.py` |
| 3 | Boxes: expert pose + organizer offset rotated with the pose; tall objects via tracker placement | measured 02:20: median IoU .81-.94 for 11 classes, ta-ta .64 (see RESULTS.md) | `live_detector.py` |
| 4 | Camera strategies: local replays on the validation scene with `run_local_eval.py` (l1 sweep vs l2_top vs revisit), then real validation attempts | `stage4_replays.sh` staged; runs after gates v2 (+verifier) exist | pod `/workspace/live/drone-flyby`, `validate_endpoint.sh` |
| 5 | Use API-verified validation labels and Higgsfield synthetic data to tailor experts/verifier; watch overfitting | pending | |
| 6 | Push branch, write handoff, update memory | milestone 1 pushed 01:45 (`508ae5f` on `drone/oscar-experts`) | `push_branch.sh` |

## Pod and paths
- Pod 1 `ypuawkayl3px8t` A100, `root@157.157.221.29:17494`, helper `pod.sh`. Code+data `/workspace/experts/project`,
  runs `/workspace/experts/runs`. Live endpoint stack `/workspace/live/drone-flyby` (scene `validation` linked to
  the 249 reconstructed frames). Two idle servers from the earlier session (api.py 9305/9306 + cloudflared) hold
  28 GB GPU memory; kill them when the GPU is needed.
- Uploads so far: expert bank, grid384 tiles, v4 sprite library, reviewed bank, grid-comparison/256 manifest.
  `upload2.sh` done 00:30: synthetic sets + reference frames on the pod. `synthetic-validation-sprites` is evaluate-only (dev backgrounds).
- Laptop uplink ~0.5 MB/s: avoid large uploads.

## Open items / decisions to revisit
- Helicopter template is the unreviewed v4 mask (`review_status=claude-auto`).
- Condor full-pixel branch regressed to 12/42 after the part model took heading; comparison branch only.
- Hangar second viewpoint (validation track hangar-053-082, 30 reviewed frames) not yet added.
- Ask Oscar in the morning: draw helicopter and hangar-053 outlines in the editor.
