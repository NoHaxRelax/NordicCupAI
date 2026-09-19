# Sunday 2026-09-20: the drone attempt, as a list

Deadline **16:00 CEST Sunday 20 September**. One evaluation attempt, zero used. Only Elias queues it (PreToolUse
hook in `.claude/settings.json`; the agent never writes `.claude/EVAL_UNLOCK`). Top five: code and models by 20:00.

> **Status Saturday midnight.** Oscar's pipeline scored 0.705 to 0.727 on full public runs from his Swedish pod on
> Saturday afternoon (leaderboard best 0.7269). This branch: plain F3 0.679 same-day from Oslo; two switches confirmed
> on the organiser truth Saturday night (cluster births, launcher box 0.85, about +0.05 together); the full served
> config is being measured as thirds from a fresh Oslo pod (`elias/out/logs/measure_pod2.log`, tags `P3_PLAIN`,
> `P3_DEPLOY`, `P3_FINAL`; first third plain 0.265). **Fill in below when they land.** Whichever endpoint has the
> higher same-day number is the attempt; the pieces Oscar can take from here are in `05-handover-to-oscar.md`.

> **Pods.** Saturday's four pods were terminated in the afternoon. The measuring pod of Saturday night is
> `jxpw8hwrqyjf5d` (elias-claude-serve-no2, RTX 4090, EUR-NO-1, 0.74 USD/h): ssh `root@149.36.0.150 -p 40957`,
> endpoint 9053 on public TCP port **40958** (`elias/out/logs/pod2.env`). If it was terminated after the measurements,
> create a new one (last section) and use its HOST, SSH_PORT and PUBLIC_PORT below. Round trip Oslo to the Helsinki
> server 40 to 70 ms; a US pod had 250 ms and lost frames. Every other pod on the account is Oscar's or Lucas's.

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

## Pre-flight, BEFORE the final start

1. Code on the pod equals the checkout: `bash elias/pod_serve_setup.sh HOST SSH_PORT` re-sends it (30 s). Weights:
   `ssh -p SSH_PORT root@HOST 'md5sum /root/out/*.pt'` shows `8a7c74ad...` for `F3_both_m1280.last.pt` (= `elias/release/both_m1280.pt`)
   and `34deb003...` for `F5_fixed_m1280.last.pt` (= `elias/release/F5_fixed_m1280.pt`).
2. **One concealed third through the public port** (proves the port from the Helsinki server, not from the laptop):

       POD_HOST=HOST POD_PORT=SSH_PORT DIRECT_URL=http://HOST:PUBLIC_PORT IMGSZ=1280 \
       CLASS_EXTENT='{"medium_launcher":"detector","ta-ta":"detector"}' CLUSTER_BIRTHS=1 BOX_SCALE='{"medium_launcher": 0.85}' \
       bash elias/pod_portal.sh /root/out/F3_both_m1280.last.pt PREFLIGHT_0_83 0:83

   (add `DETECTOR=elias.ensemble:build ELIAS_WEIGHTS=/root/out/F3_both_m1280.last.pt,/root/out/F5_fixed_m1280.last.pt ELIAS_ROUTE='<the route object: every class 0, "ta-ta": 1, as in elias/pod_start_final.sh>'`
   for the routed config). Expect the P3 number of the same config and window within 0.02. **This run leaves the
   server answering frames 0 to 82 only. Step 3 replaces it; never queue the attempt on a server started by
   `pod_portal.sh`.**

## Final start

3. `bash elias/pod_start_final.sh HOST SSH_PORT PUBLIC_PORT routed` (or `deploy`). It stops everything on the pod,
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
  and `scp -P SSH_PORT elias/release/F5_fixed_m1280.pt root@HOST:/root/out/F5_fixed_m1280.last.pt`; then from step 1.
  Saturday night this took two minutes from create to first run.
