# Sunday 2026-09-20: the attempt, as a list

Deadline **16:00 CEST Sunday 20 September**. One evaluation attempt, zero used. Everything below is
executed, not remembered. Two scripts do the build; the rest is reading numbers and pressing one button.

> **There is no pod.** `9rf8oeyh70minl` was terminated on Friday evening. The build is from bare
> metal, and it has been done exactly this way before: `pod_bringup.sh` took **10 min 41 s** on
> 2026-09-18 (weights 3.5 min in parallel with both venvs, vLLM ready in 275 s, endpoint warm).
> With pod creation, upload, a training soak and one validation run, budget **one hour**. Start by
> **12:00** at the latest; the validation queue is shared with every other team on deadline day.

## What gets served

Qwen/Qwen3.8-27B on vLLM, faster-whisper `large-v3-turbo`, Ollama `qwen3:4b` as fallback, prompt
`units-fewshot-both`, units `clause-and`, offsets -0.20/-0.02, deadlines 40/50 s. That stack
measured **0.8261** on training and **0.8084** on validation (run 397). `pod_endpoint.sh serve` now
bakes exactly this in, and `pod_bringup.sh` refuses to finish unless `GET /api` reports it.

**Top-5 obligation:** the jury wants training code and models by **20:00 CEST the same day**.
`bash bench/hpc/jury_package.sh` builds the archive from HEAD in seconds (README included); build it
before the attempt, then add the pod URL and the final score to `README_JURY.md` after.

## Build

1. **Create the pod** (RunPod MCP `create-pod`): A100 SXM 80 GB, **`minCudaVersion: 13.0`**
   (a 12.8 host silently cannot run the cu130 vLLM), ports `8000/http 9054/http 22/tcp`,
   `startSsh`, `/workspace` persistent 120 GB, $1.59/h. Read HOST and PORT from `get-pod`.
2. **Upload, from the laptop** (verifies md5 of every serving file against HEAD; refuses on mismatch):

       bash bench/hpc/pod_upload.sh HOST PORT

3. **Bring up, on the pod** (weights + venvs + vLLM + endpoint + config check, ~11 min):

       nohup bash /workspace/medical-appointment/bench/hpc/pod_bringup.sh > /workspace/logs/bringup_outer.log 2>&1 &
       tail -f /workspace/logs/bringup_outer.log        # ends with CONFIG_OK, BRINGUP_DONE and the submit URL

   The submit URL is `https://<POD_ID>-9054.proxy.runpod.net/predict`, path included.

## Pre-flight

4. From the laptop, read-only:

       curl -s https://<POD_ID>-9054.proxy.runpod.net/api    # llm_variant units-fewshot-both, unit_split clause-and, breaker_open false
       python bench/portal_status.py                          # final attempts used 0, nothing in flight

5. **One soak through the URL, training files only** (39 conversations, ~8 min; twice the validation
   load, which is the longest run the proxy has ever carried):

       python local_evaluator.py --url https://<POD_ID>-9054.proxy.runpod.net/predict

   Go: 0.80-0.83, 0 failed, 0 timeouts, worst round trip under 25 s.
6. Elias: RunPod balance covers two hours; Oscar's pods untouched; miner Space paused.
7. **One validation run** on the exact URL. Expect **0.8084**; runs 394-397 reproduce to four
   decimals, so anything below 0.79 is a broken build, not variance. Read for go/no-go only.
8. Freeze. Nothing restarted or reconfigured from here; `GET /api` uptime must cover the validation run.

## The attempt

9. **Elias** queues the evaluation attempt from the browser, URL pasted from the bring-up log. The
   agent never does this (PreToolUse hook in `.claude/settings.json`; only a human writes
   `.claude/EVAL_UNLOCK`). **Queue it by 14:00**, not 15:55: the queue is shared and unmeasured.
10. Watch `python bench/portal_status.py --watch` and nothing else. Expect 10-15 minutes.
11. After: `scp` `/workspace/request_dump` and `/workspace/logs` down, then **terminate** the pod
    (never stop: a stopped pod is pinned to its host and may never restart). Oscar's pods: untouched.
12. If top five: finish `README_JURY.md`, send the archive from step "What gets served" before 20:00.

## If the build will not come up

- `WEIGHTS_FAILED` in the log: re-run the `hf download` line by hand with `HF_HUB_OFFLINE=0`; the
  download resumes. `VLLM_FAILED`: `tail -n 40 /workspace/logs/vllm.log`; a driver below 580 means
  the wrong host, terminate and create again with `minCudaVersion 13.0`.
- Rehearsed fallback topology: laptop `api.py` on 9054 (supervisor `scratchpad/serve_forever.sh`,
  `scratchpad/serve.env`) with `LLM_URL` at the pod's 8000 proxy, behind a Cloudflare quick tunnel
  (`scratchpad/cloudflared.exe tunnel --url http://localhost:9054`); the hostname changes on every
  tunnel restart, so submit only once it is stable.
- Last resort: laptop alone on `qwen3:4b`. Scores about 0.68; it avoids a zero, nothing more.
