# Sunday 2026-09-20: the drone attempt, as a list

Deadline **16:00 CEST Sunday 20 September**. One evaluation attempt, zero used. Only Elias queues it (PreToolUse
hook in `.claude/settings.json`; the agent never writes `.claude/EVAL_UNLOCK`). Top five: code and models by 20:00.

> **FROZEN Sunday 11:15. The endpoint for the attempt is up: `http://149.36.0.173:36863/predict`, mode `robust`.**
> Same host and hour from the Oslo pod, six clean concealed thirds (0 or 1 frame lost each): **robust 0.814** (0.329 +
> 0.368 + 0.117) against **routed 0.804** (0.322 + 0.367 + 0.115). One class per run: large_launcher 0.734 routed, 0.774
> robust; small_launcher 0.546 routed, 0.545 merged, 0.498 from F3HN alone (`robust_hn` dropped). Final start done with
> `pod_start_final.sh ... robust`: no answer window, no class filter; the watchdog was tested by killing `api.py` (back in
> 12 s with the same environment). **Do not run `pod_portal.sh`, `morning_portal.sh` or anything else against this pod
> until the attempt is over.** Oscar's endpoint (0.727 Saturday) is the fallback.

> **Pods.** One pod of ours is running: `vwi15pxhoz4mf3` (elias-claude-serve-no4, RTX 5090, EUR-NO-1 Oslo, 0.99 USD/h):
> ssh `root@149.36.0.173 -p 36862`, public port 36863, 65 ms to the competition server. Oslo had no RTX 4090 on Sunday
> morning; an Iceland 4090 (125 ms) and a Stockholm A40 (122 ms compute, 10 and 17 frames lost, one 3.3 s timeout) were
> tried and terminated. **Evaluation day: the portal is slow and the path drops frames intermittently** (3 of 10 Oslo runs
> before 11:00 lost 77 to 165 frames with normal server compute; the six runs after 10:57 lost 0 or 1). Terminate the pod
> after the attempt. Every other pod on the account is Oscar's or Lucas's.

## What to expect from the attempt (red team, Sunday 04:00)

The 0.797 is a validation number and part of it will not transfer to an unseen flight. Plan with **0.60 to 0.72**; a
rehearsal or a result in that range is the expected outcome, not a fault, and no reason for a last-minute change.

| part of the 0.797 | on validation | on an unseen flight | why |
|---|---:|---:|---|
| F3 base | 0.697 | 0.60 to 0.70 | F3 was trained on validation sprites and backgrounds; the checkpoint that never saw them scores 0.601 on the same flight |
| cluster births | +0.034 | 0 to +0.034 | fires only where two same-class objects stand close: 49 % of validation medium_planes, 0 of 259 objects in the organisers' reference frames |
| launcher box 0.85 | +0.018 | -0.01 to +0.018 | needs the small launcher vehicle; safe on the Helsinki one (IoU 0.80), and the box hedge covers the other direction |
| ta-ta on F5 | +0.038 | 0 to +0.038 | needs walkers; an absent class costs nothing (macro over classes present) |

Details and the evidence: `07-committee.md`.

## What gets served

`routed`: `DRONE_DETECTOR=elias.ensemble:build`, the deployed F3 checkpoint for every class and F5 for ta-ta (the only
class where a same-day one-class run beat F3 by more than 0.03; F3 scores 0 on it, F5 0.50). `deploy`: F3 alone. Both
with **`DRONE_CLUSTER_BIRTHS=1` and `DRONE_BOX_SCALE='{"medium_launcher": 0.85}'`** (organiser truth Saturday 23:00,
one class per run, same hour: medium_plane 0.484 to 0.923, medium_launcher 0.127 to 0.360). Everything else as
measured: 1280 input, `DRONE_CONF=0.05`, birth 0.25, update 0.15, miss rule `seen`, L1 left-centre-right-centre band
oriented from the calibrated flight direction (`DRONE_AUTO_BAND=1`), detector extents for ta-ta and medium_launcher,
hedging off (0.3 gave +0.004 on the harness, unmeasured on the portal), sides-only sweep off (+0.006 on the harness,
mixed per class), zoom on cue off (rejected).

| config | 0:83 | 83:166 | 166:end | sum | source |
|---|---:|---:|---:|---:|---|
| F3 plain, Saturday afternoon | 0.262 | 0.302 | 0.114 | 0.679 | `measure_f5.log` |
| F3 plain, Sunday 00:00 to 00:50 | 0.265 | 0.322 | 0.110 | 0.697 | `measure_pod2.log` P2_PLAIN / P3_PLAIN |
| deploy (F3 + cluster births + launcher box) | 0.292 | rerun queued | 0.116 | | P3_DEPLOY |
| **routed (deploy + ta-ta on F5)** | **0.320** | **0.362** | **0.116** | **0.797** | P3_FINAL |

## The robust candidate (Sunday 05:00, laptop instruments only, portal confirmation pending)

`robust` = `routed` plus three changes aimed at the unseen flight (`07-committee.md`): large_tower answered by
`F3HN_m1280.pt` (a 2.5 minute replay fine-tune of F3 with unseen-terrain backgrounds and pure negatives), small_launcher
merged over the three models, the box hedge (a second box at the alternative size at 0.3 x confidence for medium_launcher, large_launcher, ta-ta and large_tower)
and the pair hedge (large_launcher and mine_roller, large_tower and medium_launcher answered under each other's name at
0.3 x confidence: the exact-label scene shows large_launcher answered mine_roller on half its labels). On the harness only
large_tower (0.64 to 0.72), small_launcher (0.55 to 0.59) and large_launcher (0.43 to 0.57) move; the other eight classes
are identical to the digit. Three models cost 16 ms per frame on the laptop GPU (median 94 against 78 ms). `robust_hn` answers small_launcher from F3HN alone.

| config | validation harness (team labels) | exact-label unseen scene | false answers per frame on 8 empty flights (worst flight) |
|---|---:|---:|---:|
| routed (served Saturday, portal 0.797) | 0.686 | 0.584 | 14.4 (73) |
| robust without the pair hedge | 0.696 | 0.615 | 4.6 (20) |
| **robust** (as `pod_start_final.sh ... robust` serves it, run RC_ROBUST 05:40) | **0.709** | **0.638** | 4.7 (20) |
| robust_hn | 0.696 | 0.605 | 1.6 (4.4) |

**Morning decision, one command:** `bash elias/morning_portal.sh HOST SSH_PORT PUBLIC_PORT` runs the one-class pairs
(large_tower, small_launcher, large_launcher box 0.88) and then interleaved thirds of `routed` against `robust`, all
concealed, about 14 runs (40 to 70 minutes with a free portal; ask Oscar to pause his probe loop). Serve `robust` if
its thirds sum is not more than 0.005 below `routed` on the same host and hour (its purpose is the unseen flight, not
the validation number); otherwise serve `routed`. If the portal is not free by 10:30, serve `routed` with the hedge
off: it is the only config with a portal number.

## Pre-flight, BEFORE the final start

1. Code on the pod equals the checkout: `bash elias/pod_serve_setup.sh HOST SSH_PORT` re-sends it (30 s). Weights:
   `ssh -p SSH_PORT root@HOST 'md5sum /root/out/*.pt'` shows `8a7c74ad...` for `F3_both_m1280.last.pt` (= `elias/release/both_m1280.pt`)
   `34deb003...` for `F5_fixed_m1280.last.pt` (= `elias/release/F5_fixed_m1280.pt`) and `62bb19c0...` for `F3HN_m1280.pt`
   (= `elias/release/F3HN_m1280.pt`, needed by the robust modes).
2. **One concealed third through the public port** (proves the port from the Helsinki server, not from the laptop):

       POD_HOST=HOST POD_PORT=SSH_PORT DIRECT_URL=http://HOST:PUBLIC_PORT IMGSZ=1280 \
       CLASS_EXTENT='{"medium_launcher":"detector","ta-ta":"detector"}' CLUSTER_BIRTHS=1 BOX_SCALE='{"medium_launcher": 0.85}' \
       bash elias/pod_portal.sh /root/out/F3_both_m1280.last.pt PREFLIGHT_0_83 0:83

   (add `DETECTOR=elias.ensemble:build ELIAS_WEIGHTS=/root/out/F3_both_m1280.last.pt,/root/out/F5_fixed_m1280.last.pt ELIAS_ROUTE='<the route object: every class 0, "ta-ta": 1, as in elias/pod_start_final.sh>'`
   for the routed config). Expect the P3 number of the same config and window within 0.02. **This run leaves the
   server answering frames 0 to 82 only. Step 3 replaces it; never queue the attempt on a server started by
   `pod_portal.sh`.**

## Final start

3. `bash elias/pod_start_final.sh HOST SSH_PORT PUBLIC_PORT routed` (or `robust`, `robust_hn`, `deploy`: the mode the morning decision picked). It stops everything on the pod,
   starts the server with no answer window and no class filter on the pod's own port, warms it up, prints the
   environment, refuses to continue if a window or class filter is set, starts the watchdog and prints the URL to
   queue. The watchdog (`elias/watchdog.py`) probes `/` every 5 s and `/predict` every 60 s and restarts `api.py`
   after three failures by sourcing `/root/logs/serve.env`.
4. From the laptop: `curl -s http://HOST:PUBLIC_PORT/api` answers with the service name and the uptime. The detector is
   not in that answer: `ssh -p SSH_PORT root@HOST 'grep -E "DRONE_DETECTOR|ELIAS_" /root/logs/serve.env'` (the start script prints the same lines).
5. Elias: RunPod balance covers the day; Oscar's and Lucas's pods untouched; nothing heavy on the laptop.
6. Freeze: no restarts, no config changes, no `pod_portal.sh`. The uptime in `/api` must keep growing and
   `/root/logs/watchdog.log` must show no `restarting api.py` line.

## The attempt

7. **Elias** queues the evaluation attempt in the browser with `http://HOST:PUBLIC_PORT/predict`. Queue it by
   **14:00**, not 15:55: the queue is shared with every team (four deep at midnight on Saturday).
8. The browser is the only view of the attempt. `python elias/portal.py status` lists validation runs only; use it
   beforehand to check that no validation of ours is in flight.
9. After: `scp` `/root/logs/serve/*.jsonl`, `/root/logs/archive`, `/root/logs/api.log` and `/root/logs/watchdog.log`
   to `elias/out/attempt/`, render it (`python elias/visualize_run.py --log elias/out/attempt/<sequence>.jsonl --tag attempt`),
   then **terminate** the serving pod.
10. If top five: the code is on branch `drone/elias-verifier` (weights in `elias/release/` under LFS); package before
    20:00.

## If the endpoint will not come up

- `tail -n 40 /root/logs/api.log`: a CUDA or import error means the code or weights on the pod differ from the
  checkout; re-send (`pod_serve_setup.sh`) and run step 3 again.
- The public port refuses connections: RunPod may have remapped it; `get-pod` (RunPod MCP) shows the current public
  port for private 9053. Fallback: a Cloudflare tunnel (`pod_serve.sh` without `DIRECT_URL` prints the URL).
- The pod is gone: create a Secure Cloud RTX 4090 in EUR-NO-1, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`,
  ports `22/tcp 9053/tcp`, SSH on; `bash elias/pod_serve_setup.sh HOST SSH_PORT`; `scp -P SSH_PORT elias/release/both_m1280.pt root@HOST:/root/out/F3_both_m1280.last.pt`
  `scp -P SSH_PORT elias/release/F5_fixed_m1280.pt root@HOST:/root/out/F5_fixed_m1280.last.pt` and
  `scp -P SSH_PORT elias/release/F3HN_m1280.pt root@HOST:/root/out/F3HN_m1280.pt`; then from step 1.
  Saturday night this took two minutes from create to first run.
