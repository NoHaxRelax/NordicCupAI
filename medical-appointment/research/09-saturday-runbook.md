# Sunday 2026-09-20: the attempt, as a list

Deadline 16:00 **Sunday the 20th** (corrected: it is not Saturday). One evaluation attempt, zero
used. Everything below is executed, not remembered.

> **There is no pod.** `9rf8oeyh70minl` was terminated on Friday evening once entry 59 established
> that the remaining headroom is smaller than the measurement error. The whole endpoint therefore
> has to be rebuilt from bare metal on Sunday morning, which is about **25 minutes** of wall clock
> (12 minutes of installs plus a 52 GB weight download, which the terminated pod's disk no longer
> holds). Start early. Budget two hours before the deadline, not one.

## What gets served

- Qwen/Qwen3.8-27B on vLLM, faster-whisper `large-v3-turbo`, Ollama `qwen3:4b` as fallback.
- **Config that must be live** — this is the stack that measured 0.8261 training / 0.8084 validation:

      LLM_VARIANT=units-fewshot-both        # spelling line + acted-upon line, NOT plain units-fewshot
      UNIT_SPLIT=clause-and
      LLM_DEADLINE=40
      PREDICT_DEADLINE=50
      offsets -0.20 / -0.02

  `pod_endpoint.sh serve` still bakes in `units-fewshot`. **Override it** via `/workspace/serve.env`
  or edit the script before serving, and confirm with `GET /api` that `llm_variant` reads
  `units-fewshot-both`. Serving the wrong variant costs 0.015 and would not be visible in a
  green-looking health check.

## Build (Sunday morning)

1. Create the pod — RunPod MCP `create-pod`: A100 SXM 80 GB, **`minCudaVersion: 13.0`** (the cu130
   vLLM build needs driver >= 580; a 12.8 host silently cannot run it), ports `8000/http 9054/http
   22/tcp`, `startSsh`, persistent 120 GB at `/workspace`. About $1.59/h.
2. Upload the code with an **explicit file list**. Never ship `.claude/` — it holds the API key,
   the control token and the evaluation guardrail.
3. On the pod:

       bash /workspace/medical-appointment/bench/hpc/pod_bootstrap.sh install      # venv-vllm, ~4 min
       MAX_LEN=16384 GPU_UTIL=0.80 bash .../pod_bootstrap.sh serve                 # vLLM on :8000
       bash .../pod_endpoint.sh install                                            # venv-api, ollama, turbo weights
       nohup bash .../pod_endpoint.sh serve > /workspace/logs/endpoint_serve.log 2>&1 &
       bash .../pod_endpoint.sh status

   Upload a changed file with `sed 's/\r$//' f | ssh ... 'cat > f.up && mv f.up f'` and verify
   `md5sum` against `git show HEAD:...`. An `ssh` with `< /dev/null` on a command that reads piped
   stdin **writes an empty file** — this has bitten us.
4. The URL to submit is `https://<POD_ID>-9054.proxy.runpod.net/predict`, path included. It is
   stable for the life of the pod. **Write the new id into this file.**

## Pre-flight

5. Read-only checks from the laptop:

       curl -s https://<POD_ID>-9054.proxy.runpod.net/            # "Your endpoint is running!"
       curl -s https://<POD_ID>-9054.proxy.runpod.net/api         # llm_variant units-fewshot-both, unit_split clause-and,
                                                                  # llm_model Qwen/Qwen3.8-27B, fewshot_pool 195 positives,
                                                                  # breaker_open false
       python bench/portal_status.py                              # final attempts used 0, nothing in flight

6. One soak through the URL, **training files only** — never validation files here (7 min):

       python local_evaluator.py --url https://<POD_ID>-9054.proxy.runpod.net/predict

   Go: score 0.80-0.83, 0 failed, 0 timeouts, worst round trip under 25 s.
7. Elias: RunPod balance covers the hour; Oscar's pods accounted for and untouched; Miner Space
   paused with the control token so nothing queues during the attempt.
8. **One** validation run through the exact URL. Expect **0.8084**; runs 394-397 reproduce it
   exactly, so treat anything below 0.79 as a broken build, not as variance. The score is read for
   this go/no-go only, never for tuning.
9. Freeze. Nothing restarted, redeployed or reconfigured after this point. `GET /api` uptime must
   cover the validation run, which proves no restart happened.

## The attempt

10. **Elias** queues the evaluation attempt from the browser, URL pasted from this file. The agent
    never does this — the PreToolUse hook in `.claude/settings.json` blocks it, and only a human
    writes `.claude/EVAL_UNLOCK`.
11. Watch `python bench/portal_status.py --watch` and nothing else. Expect 10-15 minutes.
12. After: copy `/workspace/request_dump` down over SSH for the post-mortem, then **terminate** the
    pod (not stop — a stopped pod is pinned to its host and may never restart, which is how
    `v2pqefpdqwlh57` was lost). Do not touch Oscar's pods.

## If the build will not come up

- Fallback topology, rehearsed: laptop `api.py` on port 9054 (supervisor `scratchpad/serve_forever.sh`,
  config `scratchpad/serve.env`) with `LLM_URL` at the pod's 8000 proxy, behind a Cloudflare quick
  tunnel (`scratchpad/cloudflared.exe tunnel --url http://localhost:9054`). The tunnel hostname is
  the submitted URL and **changes on every cloudflared restart**, so submit only after it is stable.
- Last resort: the laptop alone with `LLM_FALLBACK_MODEL=qwen3:4b` as primary. This scores far
  below 0.80 and is a way to avoid scoring zero, not a way to compete.
