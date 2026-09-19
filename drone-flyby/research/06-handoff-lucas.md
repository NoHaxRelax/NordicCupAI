# Handoff to Lucas: the drone-flyby work on `drone/elias-verifier`

Written Saturday 2026-09-19 about 17:00. Deadline Sunday 16:00 CEST, one evaluation attempt, only a human queues it
(a PreToolUse hook blocks the agent; nobody writes `.claude/EVAL_UNLOCK`). Top five hand in code and models by 20:00.

## Where things stand

- Oscar's pipeline, served from his Swedish pod, scores 0.705 to 0.727 on full public validations this afternoon
  (the portal history shows every run and the leaderboard shows the team best, 0.7269). Submit his unless something
  beats it on the same day.
- This branch: the deployed checkpoint `elias/release/both_m1280.pt` (F3) measured 0.694 on the night pod and 0.678
  same-day from an Oslo pod (thirds 0.262 + 0.302 + 0.114). A corrected-data retrain (F5, `elias/release/F5_fixed_m1280.pt`)
  measured 0.614 but is the only checkpoint that sees the 13th class, ta-ta (0.50). The planned deployment routes F3
  for every class and F5 for ta-ta (expected about 0.716, not yet measured: the routed thirds were queued when Oscar's
  runs took the portal slot). `research/05-handover-to-oscar.md` lists what plugs into his server.
- No pods are running. All four of Saturday's pods were terminated at 16:55 (Elias's instruction). Weights are on the
  laptop (`elias/out/weights/`, `elias/release/` under LFS) and pushed.
- The laptop harness campaign is running as you read this (`elias/out/logs/harness_campaign.log`): every checkpoint
  through the full pipeline per class, then confidence floor, birth and update thresholds, 1024 and 1536 input, and
  sibling hedging on F3. About 1.5 min per run, done by about 17:20.

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
| endpoint, tracker, camera policy | `api.py`, `example.py` (all switches documented at the top), `tracking/` (44 unit tests: `python -m unittest tracking.test_placement tracking.test_revisit tracking.test_tracker`) |
| our additions to the served pipeline | `DRONE_AUTO_BAND` (band at the entering edge from the calibrated motion), `DRONE_CLASS_EXTENT`, `DRONE_HEDGE_FACTOR`, `DRONE_BOX_SCALE`, stale-frame guard, `DRONE_CUE_*` (zoom on cue, rejected), `DRONE_ANSWER_WINDOWS/CLASSES` |
| multi-model routing | `elias/ensemble.py` (`DRONE_DETECTOR=elias.ensemble:build`, `ELIAS_WEIGHTS`, `ELIAS_ROUTE`, `ELIAS_ROUTE_L2`, `ELIAS_CONTEXT`; a `module:factory` entry wraps a foreign detector), `elias/route_from_log.py` |
| serving on a pod | `elias/pod_serve_setup.sh HOST PORT` (code and deps), `elias/pod_serve.sh WEIGHTS [WINDOWS]` (tunnel, or `DIRECT_URL` for the pod's own port), `elias/watchdog.py` |
| one concealed portal run from a pod | `POD_HOST=... POD_PORT=... ANSWER_CLASSES=... bash elias/pod_portal.sh /root/out/x.pt TAG "0:83"`; results append to `elias/out/logs/measure_f5.log`, transcripts and run JSONL under `elias/out/portal/TAG/` |
| chains that ran today | `elias/out/logs/measure_local.sh`, `measure_route.sh`, `measure_extras.sh`, `measure_direct.sh` (all wait for a marker line in `measure_f5.log`; killed at 16:53, restartable) |
| local harness (fast lab) | `bash elias/run_harness.sh WEIGHTS TAG --imgsz 1280 --conf 0.05 --birth-confidence 0.25 --update-confidence 0.15` (team labels, 11 classes, offline clock; output `elias/out/harness/TAG/`) |
| detector probe per class and view | `python elias/class_probe.py --weights x.pt --scene validation --step 2 --json out.json`; today's results in `elias/out/probe/` |
| training | `elias/pod_upload_drone3.sh HOST PORT` (JPEG frames, tiles, code, then starts `elias/pod_final2.sh`: generate 12000 views with `elias/data/synth_yolo.py`, train yolo26m at 1280, batch 16, 5 epochs, 27 min on a 4090); variants by `MODEL`, `BATCH`, `NAME`, `SYNTH_TERRAIN=1`; UC Merced tiles come from the HF parquet mirror (`elias/data/`) |
| sprites and data | `elias/sprites/bank.json` (227 entries; `excluded_tracks` vetoes; walker sprites `-tight`), `elias/data/validation_hidden2.json` (19 unlabelled tracks as keep-outs), `elias/data/extra_sprites.py`, `elias/data/terrain.py` |
| finding the unlabelled objects | `elias/find_unlabelled2.py`, `elias/candidate_sheets.py`, `elias/replay_endpoint.py` + `elias/replay_portal.sh` (fixed boxes through the portal, no GPU) |
| visualizer | `python elias/visualize_run.py RUN.jsonl` renders real frames, labels, the delivered view in red and our answers to MP4 and a self-contained HTML page (`elias/out/viz/`) |
| documents | `research/03-checklist.md` (every idea, status, number), `02-night-report.md` (ceiling: 1.00 to 0.92 with a perfect detector on the sweep, to about 0.85 for what the sweep cannot reach), `04-sunday-runbook.md`, `05-handover-to-oscar.md` |

## Numbers to carry in your head

Per-class AP of F3 (night pod): jet_plane 0.97, helicopter 0.96, hangar 0.95, mine_roller 0.93, tank 0.87, small_tower
0.80, large_launcher 0.72, large_tower 0.70, small_plane 0.66, small_launcher 0.56, medium_plane 0.50, medium_launcher
0.14, ta-ta 0. Same-day from Oslo: small_launcher 0.34, large_launcher 0.74. F5 from Oslo: ta-ta 0.50, small_tower
0.89, tank 0.93, medium_plane 0.53, large_launcher 0.58, small_launcher 0.31, medium_launcher 0.09.

Detector-only probe on validation (right class on the object at L1, the view the sweep delivers): F3, F5, F6 (yolo26l)
and F7 (yolo26m-p2) are alike except medium_plane (F3 0.94, F5 0.64) and medium_launcher (F5 better). F6 and F7 win
nowhere; F8 (terrain-aware pasting) equals F5 on the in-scene check.

Local harness, F3: mAP 0.600 on the team labels. Zoom on cue: 0.548 at every 4 frames, 0.461 at every 2, 0.563 with
six ta-ta cues; rejected.

## What to do next, in order

1. When the portal is free, measure the routed mix: `nohup bash elias/out/logs/measure_route.sh "ALL_DONE (F3_both_m1280)" &`
   after recreating a serving pod (runbook step 2 and `pod_serve_setup.sh`; the weights go to `/root/out/`).
   Three runs, about 12 minutes. If it beats Oscar's same-day number, the runbook is ready; if not, hand him the
   routing (his server takes our hook: `research/05-handover-to-oscar.md` section 3).
2. Read `elias/out/logs/harness_campaign.log` and the per-class tables in `elias/out/harness/*/run.log`: any threshold,
   input size or hedge that beats 0.600 by more than 0.01 on the harness deserves one concealed third on the portal.
3. medium_launcher is the largest per-class hole (0.14): the loss is confident false positives on dark blobs.
   `ELIAS_CONTEXT=0.5` (terrain prior) and F8 are the untested fixes; one one-class run each.
4. The small_launcher drop (0.56 to 0.34, same code) is unexplained. One-class run of F3 from a fresh Oslo pod, then
   from the laptop, tells host from portal.
5. Only then the bigger items in the checklist: DINOv2 re-ranking (A5), forecast boxes for retired tracks (B3),
   density crop (C3).

Every result goes into the checklist row with its number; rejected ideas stay so nobody repeats them.
