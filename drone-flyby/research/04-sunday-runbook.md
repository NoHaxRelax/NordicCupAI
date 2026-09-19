# Sunday 2026-09-20: the drone attempt, as a list

Deadline **16:00 CEST Sunday 20 September**. One evaluation attempt, zero used. Only Elias queues it (PreToolUse
hook in `.claude/settings.json`; the agent never writes `.claude/EVAL_UNLOCK`). Top five: code and models by 20:00.

> **Status Saturday 17:10.** Oscar's pipeline scores 0.705 to 0.727 on full public runs from his Swedish pod; this
> branch's best same-day number is 0.678 and the routed mix below is unmeasured. Unless a same-day run shows
> otherwise, the attempt is Oscar's endpoint and this runbook is the fallback; the pieces he can take from here are in
> `05-handover-to-oscar.md`. All four of Saturday's pods were terminated at 16:55, so steps 1 to 3 start from a fresh
> pod (see "If the endpoint will not come up", last bullet).

> **The serving pod exists.** `sue6qz4ml2mzxc` (elias-drone-serve-eu, RTX 4090, EUR-NO-1, 0.74 USD/h) has served
> every measurement since Saturday afternoon: ssh `root@149.36.0.35 -p 18676`, endpoint 9053 mapped to the public
> TCP port **18677** on the same address. Round trip to the Helsinki server 40 to 70 ms (a US pod: 250 ms, and the
> competitor reports drops from about 180 ms). Keep it running overnight (about 18 USD); creating a fresh Oslo pod is
> not guaranteed to find capacity, and the code, weights and logs are already on it.

## What gets served

The routed mix: `DETECTOR=elias.ensemble:build`, the deployed F3 checkpoint for every class and F5 for the classes
where a same-day one-class run beat F3 by more than 0.03 (ta-ta for certain; the table is printed by
`python elias/route_from_log.py` from `elias/out/logs/measure_f5.log`). Same-day Oslo thirds: F3 alone
0.262 + 0.302 + 0.114 = 0.678; the routed mix: **fill in from the ROUTE lines of measure_f5.log**. Everything else
as measured: 1280 input, `DRONE_CONF=0.05`, birth 0.25, update 0.15, miss rule `seen`, L1 left-centre-right-centre
band oriented from the calibrated flight direction (`DRONE_AUTO_BAND=1`), detector extents for ta-ta and
medium_launcher, hedging and box scale off unless the EXTRA runs said otherwise, zoom on cue off (rejected).

## Bring-up (already done, verify only)

1. `ssh -p 18676 root@149.36.0.35 'cd /root/work/drone-flyby && md5sum example.py api.py tracking/*.py elias/ensemble.py elias/pod_serve.sh'`
   against the laptop checkout at the deploy commit (CRLF differs, content must not). `ls -la /root/out/` shows
   `F3_both_m1280.last.pt` (44125273 bytes) and `F5_fixed_m1280.last.pt` (44125273 bytes).
2. Start the endpoint on its own public port, no tunnel (from the laptop; the key never leaves it):

       POD_HOST=149.36.0.35 POD_PORT=18676 DETECTOR=elias.ensemble:build \
       ELIAS_WEIGHTS=/root/out/F3_both_m1280.last.pt,/root/out/F5_fixed_m1280.last.pt \
       ELIAS_ROUTE='<the JSON from route_from_log.py>' CLASS_EXTENT='{"medium_launcher":"detector","ta-ta":"detector"}' \
       DIRECT_URL=http://149.36.0.35:18677 ssh -p 18676 root@149.36.0.35 'cd /root/work/drone-flyby && \
         IMGSZ=1280 DETECTOR=... ELIAS_WEIGHTS=... ELIAS_ROUTE=... CLASS_EXTENT=... DIRECT_URL=... bash elias/pod_serve.sh /root/out/F3_both_m1280.last.pt'

   (`elias/pod_portal.sh` does exactly this for a measurement run; for the attempt start the server the same way and
   do not queue anything.) `cat /root/logs/serve.url` prints `URL http://149.36.0.35:18677`.
3. Start the watchdog on the pod: `cd /root/work/drone-flyby && setsid nohup python elias/watchdog.py > /root/logs/watchdog.log 2>&1 &`
   (probes `/` every 5 s and `/predict` every 60 s, restarts api.py after three failures with `/root/logs/serve.env`).

## Pre-flight

4. From the laptop: `curl -s http://149.36.0.35:18677/` answers; `curl -s http://149.36.0.35:18677/api` names
   `drone-flyby-usecase` and the ensemble detector.
5. **One concealed validation through the direct port**, a third only (`DRONE_ANSWER_WINDOWS=0:83` via
   `pod_portal.sh ... ROUTE_direct_0_83 0:83` with `DIRECT_URL` set): expect the same number as the tunnel run of the
   same window, within 0.01. It proves the public port from the Helsinki server, not from the laptop.
6. Elias: RunPod balance covers the day; Oscar's and Lucas's pods untouched; nothing heavy on the laptop.
7. Freeze: no restarts, no config changes; `/root/logs/api.log` uptime must cover the attempt.

## The attempt

8. **Elias** queues the evaluation attempt in the browser with `http://149.36.0.35:18677/predict`. Queue it by
   **14:00**, not 15:55: the queue is shared with every team and unmeasured.
9. Watch the portal page; `python elias/portal.py status` (from the drone checkout; the status endpoint returned 404 on
   Saturday afternoon, so the browser is the reference).
10. After: `scp` `/root/logs/serve/*.jsonl`, `/root/logs/archive`, `/root/logs/api.log` and `/root/logs/watchdog.log`
    to `elias/out/attempt/`, render the visualizer (`python elias/visualize_run.py`), then **terminate** the two
    remaining pods: `sue6qz4ml2mzxc` (Oslo) and `y8fvk8xgo7qhuy` (pod 1, only if it was kept for a retrain).
11. If top five: the code is on branch `drone/elias-verifier` (weights in `elias/release/` under LFS); package before
    20:00.

## If the endpoint will not come up

- `tail -n 40 /root/logs/api.log`: a CUDA or import error means the code or weights on the pod differ from the
  deploy commit; re-push the files (`tr -d '\r' < file | ssh ... 'cat > path'`) and restart with the same command.
- The direct port refuses connections: RunPod may have remapped it; `get-pod` (RunPod MCP) shows the current public
  port for private 9053; fall back to the Cloudflare tunnel (`pod_serve.sh` without `DIRECT_URL`) and use that URL.
- The pod is gone: create a Secure Cloud RTX 4090 in EUR-NO-1 with ports `22/tcp 9053/tcp`, run
  `bash elias/pod_serve_setup.sh HOST PORT` (code and deps), scp the two checkpoints to `/root/out/`, then step 2.
