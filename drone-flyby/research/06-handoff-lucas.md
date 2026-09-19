# Handoff to Lucas: the drone-flyby work on `drone/elias-verifier`

First written Saturday 2026-09-19 about 17:00, updated about midnight. Deadline Sunday 16:00 CEST, one evaluation attempt, only a human queues it
(a PreToolUse hook blocks the agent; nobody writes `.claude/EVAL_UNLOCK`). Top five hand in code and models by 20:00.

## Where things stand

- Oscar's pipeline, served from his Swedish pod, scores 0.705 to 0.727 on full public validations this afternoon
  (the portal history shows every run and the leaderboard shows the team best, 0.7269). Superseded Sunday 00:50: the routed config from the Oslo pod measures 0.797 as concealed thirds against 0.697 for plain
  F3 on the same host and hour (`research/03-checklist.md`, "The served config"), so the attempt is our endpoint and his is
  the fallback (`04-sunday-runbook.md`).
- This branch: the deployed checkpoint `elias/release/both_m1280.pt` (F3) measured 0.694 on the night pod and 0.679 same-day from an Oslo pod (thirds 0.262 + 0.302 + 0.114). A corrected-data retrain (F5, `elias/release/F5_fixed_m1280.pt`)
  measured 0.614 but is the only checkpoint that sees the 13th class, ta-ta (0.50). The planned deployment routes F3
  for every class and F5 for ta-ta (expected about 0.716, not yet measured: the routed thirds were queued when Oscar's
  runs took the portal slot). `research/05-handover-to-oscar.md` lists what plugs into his server.
- Two switches confirmed on the organiser truth Saturday night (laptop-served one-class runs, same hour, everything
  else equal): `DRONE_BOX_SCALE='{"medium_launcher": 0.85}'` moves medium_launcher 0.127 to 0.360 and
  `DRONE_CLUSTER_BIRTHS=1` moves medium_plane 0.484 to 0.923. Together about +0.05 on the total. Both are in the runbook's serving line: a dozen lines in `example.py` (`_scaled`) and three in `tracking/revisit.py` (the ambiguity rule).
  They apply to Oscar's tracker as well (his branch carries the same ambiguity rule without the flag).
- Saturday's four pods were terminated in the afternoon (Elias's instruction). A fresh Oslo pod, `jxpw8hwrqyjf5d` (ssh `root@149.36.0.150 -p 40957`, public port 40958, `elias/out/logs/pod2.env`), was created at 23:45 for clean thirds of the served config (`elias/out/logs/measure_pod2.log`); the agent terminates it when the chain ends unless the result makes it Sunday's serving pod (the runbook says which). Weights are on the laptop (`elias/out/weights/`, `elias/release/` under LFS) and pushed.
- The laptop harness (`bash elias/run_harness.sh WEIGHTS TAG --imgsz 1280 --conf 0.05 --birth-confidence 0.25
  --update-confidence 0.15`, team labels, offline clock, deterministic) ran eleven campaigns Saturday
  (`elias/out/logs/harness_campaign*.log`, per-class tables in `elias/out/harness/<TAG>/run.log`). F3 baseline 0.600;
  cluster births 0.641, launcher 0.85 0.643, both plus hedge 0.3 0.688. Every threshold, input size, augmentation,
  extent, revisit and cue variant is at or below the baseline; the numbers are in the checklist rows.

## Rules that are not written in the code

- Validation frames are a held-out measurement. Never train on them with the organiser truth or the team labels as
  targets (sprites and backgrounds cut from them are fine and are what F3 uses). Elias's rule.
- Measure without exposing the score: `DRONE_ANSWER_WINDOWS="0:83"` (thirds add up to the full score within 0.015,
  the sum slightly optimistic) and `DRONE_ANSWER_CLASSES=<class>` (score x 13 = that class's AP). Full public runs
  show the real number to every team.
- Compare only same-day, same-host runs. The unchanged F3 moved from 0.56 to 0.34 on small_launcher between the
  night and the afternoon with byte-identical code; large_launcher held. Something on the portal or host side moves.
- Laptop-served runs score about a tenth below pod-served runs (0.627 against 0.694 for F3). Never run GrabCut,
  generation or rendering on the laptop while it serves a portal run: frames time out and the run is void.
- US pods have 250 ms connect to the Helsinki server and lose frames; serve from EUR-NO-1 (Oslo), 40 to 70 ms.
- Never edit a shell script that a running `nohup` chain is executing (bash reads by byte offset); copy it first.
- One validation per team at a time. A queue request made during a teammate's run is answered with that run;
  `elias/portal.py` polls every 8 s and retries 14 times, and the team key stays in the gitignored
  `.claude/nordic-api-key` of the main checkout.

## Map of the branch (`drone-flyby/`)

| what | where |
|---|---|
| endpoint, tracker, camera policy | `api.py`, `example.py` (most switches in the docstring at the top; hedge, box scale, cluster births and the answer windows and classes are commented where they are read, lines 100 to 150; `DRONE_AUTO_BAND` in `tracking/workflow.py`), `tracking/` (44 unit tests: `python -m unittest tracking.test_placement tracking.test_revisit tracking.test_tracker`) |
| our additions to the served pipeline | `DRONE_AUTO_BAND` (band at the entering edge from the calibrated motion), `DRONE_CLASS_EXTENT`, `DRONE_HEDGE_FACTOR`, `DRONE_BOX_SCALE`, stale-frame guard, `DRONE_CUE_*` (zoom on cue, rejected), `DRONE_ANSWER_WINDOWS/CLASSES` |
| multi-model routing | `elias/ensemble.py` (`DRONE_DETECTOR=elias.ensemble:build`, `ELIAS_WEIGHTS`, `ELIAS_ROUTE`, `ELIAS_ROUTE_L2`, `ELIAS_CONTEXT`; a `module:factory` entry wraps a foreign detector), `elias/route_from_log.py` |
| serving on a pod | `elias/pod_serve_setup.sh HOST PORT` (code and deps), `elias/pod_serve.sh WEIGHTS [WINDOWS]` (tunnel, or `DIRECT_URL` for the pod's own port), `elias/watchdog.py` |
| one concealed portal run from a pod | `POD_HOST=... POD_PORT=... ANSWER_CLASSES=... bash elias/pod_portal.sh /root/out/x.pt TAG "0:83"`; the RESULT line is printed and saved in `elias/out/portal/TAG/portal.txt` with the run JSONL beside it; only the chain scripts copy it into a log (`measure_f5.log` in the afternoon, `measure_pod2.log` at night), and `route_from_log.py` reads `measure_f5.log` only |
| chains that ran today | `elias/out/logs/measure_local.sh`, `measure_route.sh`, `measure_extras.sh`, `measure_direct.sh` (all wait for a marker line in `measure_f5.log`; killed at 16:53, restartable) |
| local harness (fast lab) | `bash elias/run_harness.sh WEIGHTS TAG --imgsz 1280 --conf 0.05 --birth-confidence 0.25 --update-confidence 0.15` (team labels, 11 classes, offline clock; output `elias/out/harness/TAG/`) |
| detector probe per class and view | `python elias/class_probe.py --weights x.pt --scene validation --step 2 --json out.json`; today's results in `elias/out/probe/` |
| training | `elias/pod_upload_drone3.sh HOST PORT FRAMES_DIR [UCM_DIR]` after `elias/pod_serve_setup.sh HOST PORT` has sent the code (JPEG frames and aerial tiles, then starts `elias/pod_final2.sh`: generate 12000 views with `elias/data/synth_yolo.py`, train yolo26m at 1280, batch 16, 5 epochs, 27 min on a 4090); variants by `MODEL`, `BATCH`, `NAME`, `SYNTH_TERRAIN=1`; the UC Merced tiles are 2100 JPEGs on the laptop only (default `UCM_DIR` inside the upload script, a session scratchpad path; copy them somewhere durable before that session is cleaned) |
| sprites and data | `elias/sprites/bank.json` (227 entries; `excluded_tracks` vetoes; walker sprites `-tight`), `elias/data/validation_hidden2.json` (19 unlabelled tracks as keep-outs), `elias/data/extra_sprites.py`, `elias/data/terrain.py` |
| finding the unlabelled objects | `elias/find_unlabelled2.py`, `elias/candidate_sheets.py`, `elias/replay_endpoint.py` + `elias/replay_portal.sh` (fixed boxes through the portal, no GPU) |
| visualizer | `python elias/visualize_run.py --log RUN.jsonl --tag NAME` renders real frames, labels, the delivered view in red and our answers to MP4 and a self-contained HTML page (`elias/out/viz/`) |
| documents | `research/03-checklist.md` (every idea, status, number), `02-night-report.md` (ceiling: 1.00 to 0.92 with a perfect detector on the sweep, to about 0.85 for what the sweep cannot reach), `04-sunday-runbook.md`, `05-handover-to-oscar.md` |

## Numbers to carry in your head

Per-class AP of F3 (night pod): jet_plane 0.97, helicopter 0.96, hangar 0.95, mine_roller 0.93, tank 0.87, small_tower
0.80, large_launcher 0.72, large_tower 0.70, small_plane 0.66, small_launcher 0.56, medium_plane 0.50, medium_launcher
0.14, ta-ta 0. Same-day from Oslo: small_launcher 0.34, large_launcher 0.74. F5 from Oslo: ta-ta 0.50, small_tower
0.89, tank 0.93, medium_plane 0.53, large_launcher 0.58, small_launcher 0.31, medium_launcher 0.09.

Detector-only probe on validation (right class on the object in a best-case L1 view centred on it; files in `elias/out/probe/`): F3, F5, F6 (yolo26l) and F7 (yolo26m-p2) are alike except medium_plane (F3 0.94, F5 0.64; in the deployed top-band view 0.89 and 0.78) and medium_launcher (both find the class, F5 with the better box and confidence). F6 and F7 win
nowhere; F8 (terrain-aware pasting) equals F5 on the in-scene check.

Local harness, F3: mAP 0.600 on the team labels. Zoom on cue: 0.548 at every 4 frames, 0.461 at every 2, 0.563 with
six ta-ta cues; rejected.

## What to do next, in order

0. Tell Oscar about the two confirmed switches (section 5b and 5c of his handover) if Elias has not; they are worth
   more on his 0.727 pipeline than anything else on this branch.
1. When the portal is free, measure the routed mix with the two switches on (`CLUSTER_BIRTHS=1 BOX_SCALE='{"medium_launcher": 0.85}'`
   in front of `pod_portal.sh`, the short names: the pod scripts export the `DRONE_` ones). Saturday night's chain does exactly this: `elias/out/logs/measure_pod3.sh` (tags `P3_PLAIN`, `P3_DEPLOY`, `P3_FINAL`, log `measure_pod2.log`); copy it for a new pod and change `elias/out/logs/pod2.env`. The afternoon's `measure_route.sh` hardcodes a terminated pod and lacks the switches; do not reuse it. A run takes 2 to 6 minutes depending on the portal queue. If it beats Oscar's same-day number, the runbook is ready; if not, hand him the
   routing (his server takes our hook: `research/05-handover-to-oscar.md` section 3).
2. Read `elias/out/logs/harness_campaign.log` and the per-class tables in `elias/out/harness/*/run.log`: any threshold,
   input size or hedge that beats 0.600 by more than 0.01 on the harness deserves one concealed third on the portal.
3. medium_launcher was the largest per-class hole (0.14) and the cause was box size, not false positives: 0.36 with the 0.85 box scale. The terrain prior (`ELIAS_CONTEXT=0.5`) and F8 left the class at 0.29 and 0.25 on the harness (0.29 baseline); neither has a portal run.
4. The small_launcher drop (0.56 to 0.34, same code) is unexplained. One-class run of F3 from a fresh Oslo pod, then
   from the laptop, tells host from portal.
5. Only then the bigger items in the checklist: DINOv2 re-ranking (A5), mid-life refresh (C2), density crop (C3). Forecast boxes for retired tracks (B3) were measured neutral on Saturday.

Every result goes into the checklist row with its number; rejected ideas stay so nobody repeats them.
