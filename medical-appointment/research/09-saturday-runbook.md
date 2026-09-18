# Saturday 2026-09-19: the attempt, as a list

Deadline 16:00. One evaluation attempt. Everything below is executed, not remembered
(research/committee-2026-09-17/02-deployment.md, pre-flight; findings 52-53 for the numbers).

## What is being served

- URL to submit: **https://v2pqefpdqwlh57-9054.proxy.runpod.net/predict** (path included).
- Pod `nordic-27b-serve` (RunPod, id `v2pqefpdqwlh57`, A100 80 GB, $1.59/h): turbo ASR, api.py,
  vLLM Qwen/Qwen3.8-27B at 80 % memory, Ollama qwen3:4b fallback. Config: `units-fewshot`,
  `UNIT_SPLIT=clause-and`, offsets -0.20/-0.02, `LLM_DEADLINE=40`, `PREDICT_DEADLINE=50`.
- Measured Friday through that URL on the 39 training files: 0.815 / 0.812 / 0.815, 8.5-9.1 s mean,
  13-17 s worst per conversation, 0 failed, 0 fallbacks.
- Alternative (rehearsed, not preferred): laptop api.py on port 9054 with LLM calls to the pod;
  needs a tunnel that does not exist today.

## Friday evening: leave it alone

Nothing runs on the pod but the endpoint. No benches, no experiments, no restarts. The laptop's
Ollama, oracle server and laptop endpoint may stay up; they do not touch the pod.

## Saturday morning

1. Status, from the laptop (all read-only):
   ```
   curl -s https://v2pqefpdqwlh57-9054.proxy.runpod.net/            # "Your endpoint is running!"
   curl -s https://v2pqefpdqwlh57-9054.proxy.runpod.net/api         # uptime, unit_split clause-and, llm_model Qwen/Qwen3.8-27B,
                                                                    # fewshot_pool 195 positives, breaker_open false, counts
   python bench/portal_status.py                                    # validations N, final attempts used 0, nothing in flight
   ```
   Go: uptime covers the night (no restart), counts show no guesses/timeouts since the last soak.
   No-go: a restart in the night means something crashed: read /workspace/logs/api.supervisor.log
   and api.log over SSH (`ssh -i ~/.ssh/id_ed25519 -p <port> root@<host>`, host/port from the
   RunPod Connect panel or `get-pod`) before anything else.
2. One soak through the URL, training files only (7 minutes; never validation files here):
   ```
   python local_evaluator.py --url https://v2pqefpdqwlh57-9054.proxy.runpod.net/predict
   ```
   Go: score 0.80-0.82, 0 failed, 0 timeouts, worst round trip under 25 s.
3. Balance and clock (Elias): RunPod balance at least $30; Oscar's pods accounted for; at least
   90 minutes before 16:00 remain when step 4 starts.
4. Miner Space paused (Elias, control token): the Space must not queue anything during 5-7.

## Two validation runs, then freeze

5. Elias queues **two validation runs** on the exact URL, back to back, no restart between.
   Read from `python bench/portal_status.py --watch`: both finish, 0 request errors, scores within
   0.02 of each other and at or above 0.70. The scores are read only for this go/no-go, not for
   any tuning. No-go: fix, then repeat both.
6. Freeze. From here nothing is restarted, redeployed or reconfigured. `GET /api` uptime must cover
   both validation runs (proves no restart happened).

## The attempt

7. Elias queues the evaluation attempt from the browser, URL pasted from this file. The agent never
   does this (hook in .claude/settings.json).
8. Watch `python bench/portal_status.py --watch` and nothing else. Expect 10-15 minutes.
9. After: `portal_status.py` shows final attempts used 1 and a score; keep `/workspace/request_dump`
   on the pod for the post-mortem (copy it down over SSH); then **stop the pod** in the RunPod UI
   (or `pod-action stop` through the RunPod MCP if it is authorised). Do not stop Oscar's pods.

## If the pod is unreachable in the morning

- Pod stopped or lost: start it in the RunPod UI, read the new SSH line, then on the pod
  `bash /workspace/medical-appointment/bench/hpc/pod_bootstrap.sh install && MAX_LEN=16384 GPU_UTIL=0.80 bash .../pod_bootstrap.sh serve`,
  `bash .../pod_endpoint.sh install`, then `setsid nohup bash .../pod_endpoint.sh serve > /workspace/logs/endpoint_serve.log 2>&1 < /dev/null &`
  (about 12 minutes; /workspace keeps weights, code and Ollama models; /opt is rebuilt). The proxy
  hostname stays the same while the pod exists. Then steps 1-2.
- Pod gone for good: create a new one (RunPod MCP `create-pod`: A100 SXM 80 GB, `minCudaVersion 13.0`,
  ports `8000/http 9054/http 22/tcp`, `startSsh`, persistent 120 GB at /workspace), upload the code
  tarball (explicit file list; never `.claude/`), run the two install scripts, and the URL changes to
  the new pod id: update this file and the submission.
- Fallback topology: laptop api.py (supervisor `scratchpad/serve_forever.sh`, config `scratchpad/serve.env`)
  with `LLM_URL` at the pod's 8000 proxy, behind a Cloudflare quick tunnel
  (`scratchpad/cloudflared.exe tunnel --url http://localhost:9054`); the tunnel hostname is the URL and
  changes on every cloudflared restart.
