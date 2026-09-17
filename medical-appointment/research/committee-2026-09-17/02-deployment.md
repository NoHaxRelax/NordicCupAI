# Committee review 02: deployment and operational risk of the final run

Reviewer lens: deployment and operations. Written 2026-09-17 night, for Friday morning.
Sources read: `README.md`, `model.py`, `example.py`, `api.py`, `dtos.py`, `utils.py`,
`local_evaluator.py`, `bench/llm/prompts.py`, `bench/llm/bench.py`, `bench/llm/serve_vllm.sh`,
`bench/hpc/RUNPOD.md`, `bench/hpc/HANDOFF.md`, `bench/hpc/sync.sh`, `bench/mine/probe_server.py`,
`bench/results/llm/*.json`, `bench/results/asr/summary.cluster.md`,
`research/07-findings-log.md` entries 10, 12-15, 20, 21, 26-28, 31-37, `research/06-edge-ideas.md`
item 48, and `.claude/hooks/block-evaluation.py`. Nothing was executed and no server was contacted.

## Verdict

The modelling case for the 27B is solid and the timing case is comfortable, but the deployment is
not close to ready and almost none of the remaining risk is in the model. What serves today
(`model.py`) talks to Ollama's native `/api/chat` with Ollama-only fields and parses Ollama's
response shape, so every line of the LLM client changes for vLLM; the prompt that won the bench
(`units-fewshot`, 0.780) lives in `bench/llm/prompts.py`, which imports `model.py` and therefore
cannot simply be imported back; and its example pool is built from the 39 `*.large-v3-turbo.json`
transcripts, which are gitignored and exist only on this laptop, so a fresh checkout on Oscar's
box degrades the prompt to an all-negative example block without raising anything. Timing is the
easy part: on the H100 the few-shot prompt costs 2.8 s of LLM time per conversation (worst 6.0 s)
against a 60 s budget, so the variant choice should be made on operational grounds, and on those
grounds few-shot beats joint-demo decisively because its blast radius on a failure is one question
rather than ten. The real threats to the single attempt are all infrastructure: an ephemeral
`trycloudflare` hostname that is also the submitted URL, a 4.95 MB body against proxy defaults of
1 MB, a shared RunPod balance that can stop the pod mid-attempt, a table-lookup probe server one
port away from the real endpoint, and a server crash that has already scored us 0.0000 once today
(entry 13). My recommendation: one stable endpoint with a hard 40 s internal deadline and the
laptop 4B as an in-process fallback, ASR on the laptop, the 27B behind it, `units-fewshot`, and a
pre-flight that is executed as a list rather than remembered.

## Findings

### 1. The served LLM client is Ollama-native and every field of it changes for vLLM

**Claim.** `model.ask_llm` posts to `{LLM_URL}/api/chat` (`model.py:263`) with `format` for the
schema, `think: False`, `keep_alive: -1` and `options: {temperature, num_ctx, num_predict}`
(`model.py:251-262`), then reads `r.json()['message']['content']` (`model.py:265`). vLLM serves
none of that: the path is `/v1/chat/completions`, the schema goes in
`response_format: {"type": "json_schema", "json_schema": {"name": ..., "schema": ...}}`, thinking
is switched off with `chat_template_kwargs: {"enable_thinking": false}`, the token cap is
`max_tokens`, `num_ctx` has no equivalent (it is the server's `--max-model-len`), `keep_alive` does
not exist, and the answer is at `choices[0].message.content`. The working reference implementation
is `bench/llm/bench.py` `Client.body` and `Client.chat`, which is what produced every 27B number.

**Evidence.** `model.py:240-266`; `bench/llm/bench.py` (`Client.body`, `Client.chat`,
`parse_json`); `bench/llm/serve_vllm.sh` header, "Flags verified against docs.vllm.ai".

**Severity.** High. Not subtle (it fails immediately), but it is five separate places to get right
and there is no test for it outside the bench.

**Action.** Extract `bench/llm/bench.py`'s `Client` into the serving path rather than rewriting it,
with an `LLM_API=ollama|openai` switch so the laptop 4B fallback keeps working through the same
code. Specific things that go wrong, each of which must be checked once by hand:
a wrong model name (vLLM 404s or 400s on an id that is not the served one, so either read
`GET /v1/models` at warm-up or pass `--served-model-name`); a 400 on `json_schema` with no
fallback (the bench falls back to `json_object` once and says so; `model.py` has no such path);
thinking left on, which returns empty `content` and a `reasoning_content` field, so
`json.loads('')` raises and every question becomes a guess; and fenced or `<think>`-wrapped
content, which `model.py`'s bare `json.loads` cannot survive but `bench.py`'s `parse_json` strips.
Port `parse_json` as it stands.

### 2. The winning prompt cannot be imported from the bench, and the bench must keep measuring what serves

**Claim.** `bench/llm/prompts.py` does `from model import ...` at module level
(`bench/llm/prompts.py:60-63`). If `model.py` imports `prompts` at module level, that is a circular
import; and `bench/llm` is not on the server's `sys.path` at all, so the import also has to be
arranged. The deliberate design note in the `prompts.py` docstring ("the bench measures exactly
what serves") points the right way: the prompt machinery belongs in the serving code.

**Evidence.** `bench/llm/prompts.py` docstring and lines 55-70; `model.py` has no reference to
`prompts`.

**Severity.** Medium, but it is on the critical path for every other change.

**Action.** Move `FewShot` (and `Joint`/`JointDemo` if the joint form is chosen) into `model.py`
next to `SYSTEM` and `SCHEMA`, and have `prompts.py` import them from there. Then one file is
deployed, the bench still exercises the serving code, and the serving path has no `bench/`
dependency. Do not solve it with a lazy import inside `ask_llm`: that hides the failure until the
first request, which during the attempt is the worst possible time to discover it.

### 3. The few-shot example pool degrades silently to an all-negative prompt when the turbo transcripts are missing

**Claim.** This is the most dangerous single line in the deployment. `FewShot.pool` reads
`data/question_train.csv` plus `transcripts/<stem>.<asr>.json` with `asr` defaulting to
`large-v3-turbo` (`bench/llm/prompts.py:369`), extracts the words inside each gold span, and then
drops any positive example whose text came out empty: `if ex['yes'] and not ex['text']: continue`
(`bench/llm/prompts.py:398`). If the turbo transcripts are absent, every positive example is
dropped and the EXAMPLES block reduces to the two negative lines "no evidence (answer no)". The
prompt then demonstrates only how to answer no, on a set that is exactly balanced. Nothing raises,
nothing logs, and the endpoint answers fast and wrong.

The transcripts are not in the repository: `medical-appointment/.gitignore` ignores `transcripts/`,
and `git ls-files transcripts/` returns 39 files, all of them `*.large-v3.json`. The 39
`*.large-v3-turbo.json` files, which are both the served ASR's transcripts and the pool's source,
are untracked local artefacts. `bench/hpc/sync.sh up` copies them to the cluster and
`bench/hpc/HANDOFF.md` item 3 records that they were copied by hand, which is exactly how this
became invisible.

`JointDemo.pool` has the same dependency with a gentler failure: `if not tf.exists(): continue`
(`bench/llm/prompts.py:516`) leaves the pool empty, `demos_for` returns no demos, and the prompt
silently becomes plain `units-joint`, which scores 0.761 instead of 0.775.

**Evidence.** `bench/llm/prompts.py:389-403` and 505-520; `medical-appointment/.gitignore`
("Transcripts and model weights you cache locally"); `git ls-files transcripts/` (39 files, tag
`large-v3` only); `bench/hpc/HANDOFF.md` "Done so far" item 3.

**Severity.** Critical. A silent quality failure on a one-shot attempt is worse than a crash.

**Action.** Three things, all cheap. (1) Ship the pool explicitly: either commit the 39
`*.large-v3-turbo.json` files (they are our own derived data and small) or add a `deploy/` manifest
and copy them with the code; the deployment needs `data/question_train.csv` and those 39 files and
nothing else from `data/`. (2) Assert at warm-up: count positive examples with non-empty text and
refuse to consider the endpoint healthy below a floor (the pool should hold about 195 positives;
require at least 150). (3) Make the degrade explicit: if the pool is short, fall back to the plain
`units` prompt (0.762 measured) and log it at ERROR, rather than serving an example block that
teaches the model to say no. Report the count in `GET /api` (finding 16).

### 4. The joint prompt loses ten questions per failure; the few-shot prompt loses one

**Claim.** On the 27B the two candidates are statistically tied and operationally far apart.
`units-fewshot` scores 0.780 (nulls on no) with ten independent requests of about 1,472 prompt
tokens and 35 completion tokens; `units-joint-demo` scores 0.775 with one request of 3,503 prompt
tokens and 294 completion tokens. The difference, 0.005, is below the 0.006 noise floor of entry 15.
But a single malformed or truncated reply costs one question under few-shot and all ten under
joint, and the README prices that at ten marks rather than one. The joint form also needs a
1,500-token completion budget where few-shot needs 200: entry 34 reading (6) records that the
joint variants failed on the cluster purely because the 200-token default cut the ten-answer JSON.
Under schema-constrained decoding a cut JSON is unparseable, so the failure is total, not partial.

**Evidence.** `bench/results/llm/qwen3.8-27b.units-fewshot.large-v3-turbo.json` (score 0.780
nulls-on-no, completion mean 34.9, max 150, 0 stopped at max_tokens 200) and
`...units-joint-demo...json` (0.775, completion mean 294.1, max 328, max_tokens 1500);
`research/07-findings-log.md` entries 34 (reading 6), 35, 15 (noise floor 0.006);
`README.md` "OBS", "One failure costs ten questions".

**Severity.** High, as a design choice made now rather than on Saturday.

**Action.** Serve `units-fewshot`. It also requires the least new code: the existing per-question
`SCHEMA`, the existing `anchor_ids`/`span_from_ids` post-processing, the existing 200-token cap
(raise to 256 for margin), and no answer-splitting logic at all. If the pending pod bench of
`units-joint-demo-fewshot` comes back more than 0.02 above few-shot, reconsider; below that, the
robustness is worth more than the point. If joint is chosen anyway, then: 1,500-token budget;
default a missing or out-of-range `q` entry to a guess with no span and count it in the log;
and on a parse failure retry the conversation once with the per-question few-shot path, which the
budget easily affords (10.2 s used of 60 s).

### 5. Nothing in the request path guarantees a reply inside 60 seconds

**Claim.** This is the only failure that can end the attempt early, and the current code has no
defence against it. `model.answer_all` transcribes with no time limit, then fans the questions out
over a thread pool with `max_workers=len(questions)` (`model.py:360`) where each request carries a
25 s socket timeout (`model.py:71`). A per-question exception is caught and turned into a guess
(`model.py:356-358`) and `example.predict` catches whatever is left (`example.py:47-56`), so a
*failed* backend produces a well-formed reply, which is exactly right. What is missing is a cap on
the *slow* path: a hung ASR call, a TCP connection that is accepted but never answered, or a vLLM
restart that leaves requests queued, all produce silence rather than a guess, and five silences in
a row end the attempt and score every remaining conversation wrong.

**Evidence.** `model.py:338-368`; `README.md` "Timing" and "Scoring" ("What ends an attempt early
is silence"); `local_evaluator.py:51,61` (`REQUEST_TIMEOUT_SECONDS = 60`,
`MAX_CONSECUTIVE_TIMEOUTS = 5`).

**Severity.** Critical. It converts every other infrastructure fault from "lose ten marks" into
"lose the tail of the attempt".

**Action.** Add a wall-clock deadline in `example.predict`: run `answer_all` in a worker thread
with a 40 s budget and, on overrun, return the all-guess response immediately (`[True] * n`,
`[None] * n`) while the thread is abandoned. Separately: split the connect and read timeouts
(connect 3 s, read 15 s for few-shot, whose measured p95 is 3.0 s per question), use one
module-level `requests.Session` with a pooled `HTTPAdapter` and no retries instead of a fresh
connection per question (ten TLS handshakes per conversation over a tunnel is pure latency), and
give ASR its own guard. The fallback response is worth about 0.2 for that conversation; silence is
worth 0 for that conversation and threatens everything after it.

### 6. Add the laptop 4B as an in-process fallback, not as a second URL

**Claim.** The rollback that everyone assumes is "submit the laptop URL instead", but the URL is
fixed at the moment the attempt is queued and cannot be changed afterwards
(`portal_status.queue_validation` posts `{'url': service_url}`, and the README says the URL is used
exactly as given). A backend-level fallback is strictly better: if the 27B does not answer within
its read timeout, ask the local Ollama `qwen3:4b` with the same `units` prompt, which is the
configuration that scored 0.6759 on validation. Then a pod that dies mid-attempt costs tIoU, not
the attempt, and no human decision is needed while the clock runs.

**Evidence.** `bench/portal_status.py:66-70`; `README.md` "Make your endpoint reachable" ("used
exactly as given, path included"); `research/07-findings-log.md` entry 26 (run F, 0.6759) and
entry 10 (turbo plus the 4B coexist on the 8 GB card; validation runs served 7 to 18 s per
conversation).

**Severity.** High value, low cost. About 30 lines given finding 1's `LLM_API` switch.

**Action.** Implement it, keep Ollama running with `keep_alive: -1` and the 4B resident during the
attempt, and log every fallback so the post-mortem shows how many conversations used it. Verify
once with a validation run in which the pod's vLLM is deliberately stopped.

### 7. Timing: the few-shot prompt is safe on both cards, joint-demo is safe only on a measured card

**Claim.** The measured H100 numbers leave a large margin, and the arithmetic matters because the
whole-attempt budget is an average, not a per-request allowance.

| variant (Qwen3.8-27B, H100, vLLM 0.29) | LLM per conversation, mean | worst | prompt tokens | completion tokens, max |
|---|---:|---:|---:|---:|
| units-fewshot | 2.76 s | 5.99 s | 1,472 | 150 |
| units | 2.16 s | 3.52 s | 961 | 34 (of 200) |
| units-joint-demo | 10.23 s | 11.45 s | 3,503 | 328 (of 1500) |
| units-joint | 17.22 s | 19.63 s | 1,192 | 587 (of 1500) |

ASR adds on top of that. `large-v3-turbo` measured RTF is 0.043 on the cluster sweep, so a 2-minute
median conversation is about 5 s and the 3.5-minute worst case about 9 s on that hardware; on the
laptop the served end-to-end round trip was 7 to 18 s per conversation with the 4B, which brackets
it. So on an H100: few-shot lands at roughly 8 s per conversation typical and under 15 s worst,
that is 13 percent of a single request's budget and about 5 minutes of the 38-minute whole-attempt
budget for 38 conversations. Joint-demo lands at about 15 s typical, 10 minutes of the attempt
budget, still comfortable. Both are safe on an H100 by a factor of four or more.

On an A100 nothing is measured. From the hardware alone, expect prefill-bound work to take two to
three times longer (A100 BF16 dense throughput is roughly a third of H100's) and decode roughly
1.7 times longer (2.0 against 3.35 TB/s). That puts few-shot at 6 to 9 s of LLM time, still safe,
and joint-demo at 20 to 30 s, which combined with ASR is inside 60 s per request but only just, and
eats half the whole-attempt average. `units-joint-demo-all` (38 worked conversations, about 60k
prompt tokens, `--max-model-len 98304` per `bench/hpc/RUNPOD.md`) has never been run at all; a 60k
prefill is plausibly 10 to 20 s on an H100 and 30 to 60 s on an A100, and the demo set shifts per
conversation so prefix caching will not save it.

**Evidence.** The four result files under `bench/results/llm/` cited above (their "Round trip" and
"per conversation wall" lines); `bench/results/asr/large-v3-turbo.json` (`rtf` 0.0428,
`rtf_per_file_mean` 0.0478, `audio_seconds` 4766.7 over 39 files);
`research/07-findings-log.md` entries 10, 20, 26; `bench/hpc/RUNPOD.md` step 3;
`README.md` "Timing".

**Severity.** Medium, and it resolves to a clear rule.

**Action.** Serve `units-fewshot` on either card. Do not serve `units-joint-demo` on an A100 without
measuring it first (`bench/llm/bench.py --variant units-joint-demo --limit 5` reports the per
conversation wall directly). Do not serve `units-joint-demo-all` at all this weekend: it is
unmeasured, it needs a 98k context, and its prefill is the one thing that can push a single request
past 60 s.

### 8. Cold start: the expensive one is vLLM's boot, not the first request

**Claim.** Two different cold starts are being conflated. The first inference after `/health`
answers costs seconds, not minutes, and the measured evidence is in the bench worst cases: few-shot
worst 5.99 s against a 2.76 s mean, joint-demo worst 11.45 s against 10.23 s, which is the first
conversation paying for schema grammar compilation and a cold prefix cache. Our own `warm_up`
already covers the ASR side properly: it loads faster-whisper and runs one real transcription of
`data/audio/conversation_sample_4.mp3` (`model.py:371-384`). The expensive cold start is vLLM
itself, about 8 minutes on the pod per `bench/hpc/RUNPOD.md`, of which 52 GB of weights off a
network volume is the bulk, plus torch.compile artefacts on a first start for a given config.

Two traps around it. First, `serve_vllm.sh` defaults `XDG_CACHE_HOME` and `VLLM_CACHE_ROOT` to
`/dtu/blackhole/...`, which does not exist on RunPod, and the pod's container disk is erased on
every stop (`RUNPOD.md`), so unless those point at `/workspace` every pod restart recompiles.
`HF_HOME` has the same default, which on the pod would mean re-downloading 52 GB into a disk that
is about to be erased. Second, `warm_up` runs at import of `example.py`, which runs at import of
`api.py`, before uvicorn binds the port (`example.py:19`, `api.py:19,58`), and `warm_llm` swallows
its own failure (`model.py:269-274`). So the process can be up and the port closed for the length
of the warm-up, and a connection refused during the attempt is silence, which accumulates towards
the five.

**Evidence.** `bench/results/llm/qwen3.8-27b.units-fewshot.large-v3-turbo.json` ("per conversation
wall 2.76 s mean, 5.99 worst"); `model.py:371-384`, `269-274`; `example.py:19`; `api.py:19,58`;
`bench/llm/serve_vllm.sh` (`HF_HOME`, `XDG_CACHE_HOME`, `VLLM_CACHE_ROOT` defaults);
`bench/hpc/RUNPOD.md` ("about 8 minutes to a serving vLLM", "container disk is erased on every
stop"); `README.md` "There is no separate warm-up period".

**Action.** Never restart vLLM inside the attempt window: bring it up, verify it, and leave it.
Set `HF_HOME=/workspace/hf` and `XDG_CACHE_HOME=/workspace/cache` on the pod so a restart is
minutes rather than an hour. Extend `warm_up` to exercise the *served* prompt (few-shot block,
real schema, real pool) so the grammar compiles and the pool loads before the first request, not
during it. Make the warm-up retry `/health` in the background instead of blocking the bind, and
never queue anything until `GET /` on the public URL answers.

### 9. The 27B bench numbers were measured with large-v3's edge offsets, not the served turbo ones

**Claim.** Every 27B result reports `offsets start -0.14s end +0.12s`, which are `large-v3`'s fitted
values. `bench/llm/bench.py` selects the transcript tag with `--asr` but the offsets come from
`model.py`'s `_FITTED[ASR_MODEL]` with `ASR_MODEL` defaulting to `large-v3` (`model.py:66,77-83`),
and the cluster jobs never exported it. The served configuration uses turbo's own rule
(-0.20 start, -0.02 end), which is what bought 0.051 on validation in entry 20. The ASR sweep
quantifies the gap at the ceiling: for `large-v3-turbo`, oracle tIoU is 0.848 with its own fitted
shift and 0.814 with the served large-v3 shift.

**Evidence.** The `offsets` line in every `bench/results/llm/qwen3.8-27b.*.json` report block;
`model.py:66,77-83`; `bench/results/asr/summary.cluster.md` section A (`large-v3-turbo`: +fit shift
0.848, +served shift 0.814); `research/07-findings-log.md` entries 17, 20.

**Severity.** Medium. The variant *ranking* is probably unaffected because every row shares the
mistake, but 0.780 is not the number the served configuration would produce, and the difference is
of the same order as the gaps being argued over.

**Action.** Rerun the top two variants once with `ASR_MODEL=large-v3-turbo` exported. It costs
108 s of H100 time for few-shot (measured total wall) and it is the only way the bench number and
the served number mean the same thing. Do it in the same pod session as the three pending benches.

### 10. Failure mode: the tunnel hostname is ephemeral and it is also the submitted URL

**Claim.** The public path is a Cloudflare quick tunnel (`https://xxx.trycloudflare.com`, the form
used throughout `bench/mine/count_run.py` and `bench/mine/span_probe.py`). A quick tunnel's
hostname is random and is reassigned on every `cloudflared` restart, and the attempt carries the
URL as a parameter, so if `cloudflared` dies during the attempt the endpoint is unreachable at the
submitted address for the rest of it. That is silence, five in a row, and every remaining
conversation scored wrong. Worse, nowhere in the repository is it written down how the endpoint is
made reachable; I found only `localhost` URLs and the `xxx.trycloudflare.com` placeholder.

**Likelihood.** Moderate over a 10 to 15 minute attempt, high over a day. Quick tunnels are
best-effort with no SLA; they survived 358 probe runs over an hour today, which is encouraging but
not a guarantee.
**Consequence.** Catastrophic: the tail of the attempt, up to the whole score.
**Mitigation.** A named tunnel on a domain Elias controls, so the hostname survives a
`cloudflared` restart, or the pod's own address if the pod is the endpoint. Keep `cloudflared`
under the same restart supervisor as the server. Have the exact URL in a file and paste it, never
retype it, and include `/predict`.
**Test.** Queue a validation run against the final URL from the final process, then kill
`cloudflared`, let it restart, and check whether the hostname changed. If it changed, a named
tunnel is mandatory, not optional.

**Evidence.** `bench/mine/count_run.py:10`, `bench/mine/span_probe.py:29-30`;
`bench/portal_status.py:66-70`; `README.md` "Make your endpoint reachable";
`research/07-findings-log.md` entry 28 (358 validation runs at about 9 s each).

### 11. Failure mode: a proxy in front of the endpoint rejects a 5 MB body

**Claim.** The largest training MP3 is 3.71 MB, which is 4.95 MB base64, against nginx's default
`client_max_body_size` of 1 MB and default `proxy_read_timeout` of 60 s. Any proxy hop (nginx on
Oscar's box, RunPod's HTTP proxy, a tunnel) is a candidate. The README warns about exactly this.

**Likelihood.** Low if there is no nginx, high if anybody adds one. RunPod's HTTP proxy limits are
not documented anywhere in this repo, which means they are unknown, which is the same thing as a
risk.
**Consequence.** Total. A 413 on every request is a non-2xx on every conversation, so 38
conversations scored wrong, and it would look like a model problem in the portal.
**Mitigation.** Prefer a path with no HTTP proxy: the tunnel straight to uvicorn, or RunPod's
direct TCP port mapping rather than its HTTP proxy. If a proxy is unavoidable, set
`client_max_body_size 16m` and `proxy_read_timeout 90s`.
**Test.** Before queueing anything, POST the largest training body to the public URL from a device
that is not on the same network (phone hotspot), and read the status code and the round trip. One
curl, two minutes, protects the whole attempt.

**Evidence.** `research/06-edge-ideas.md` item 48 (3.71 MB / 4.95 MB, nginx defaults, body handling
9.7 ms); `README.md` "Bodies are megabytes, not kilobytes"; `bench/hpc/RUNPOD.md` step 1 (ports
8000 and 9054 must be added as HTTP ports; only 8888 and TCP 22 are exposed now).

### 12. Failure mode: vLLM dies or OOMs under load, especially sharing a card with faster-whisper

**Claim.** `serve_vllm.sh` carries its own warning that a dense 27B BF16 checkpoint at
`--gpu-memory-utilization 0.92` with ten concurrent workers "was never load-tested" and can OOM
mid-run rather than at load. That warning is now partly discharged: entries 34 and 35 ran 390
questions at ten workers against exactly this configuration with zero request errors. What is
still untested is faster-whisper sharing the card. At 0.92 utilisation on an 80 GB H100, vLLM
reserves about 73.6 GB and leaves about 6.4 GB for everything else; turbo in float16 needs roughly
1.6 GB plus cuDNN workspace, so it fits, but with no margin for a fragmented allocator.

**Likelihood.** Low for the LLM alone on an H100, moderate if ASR moves onto the same card at 0.92.
**Consequence.** Bounded, and this is the good news: a 500 or a connection error from vLLM is
caught per question and becomes a guess (`model.py:356-358`, `example.py:53-55`), so the
conversation scores about 0.2 and the timeout counter stays at zero. The attempt survives. Only
a hang, not a crash, is dangerous, which is what finding 5 addresses.
**Mitigation.** Run ASR on the laptop and keep the pod's GPU for vLLM alone; this is the topology
I recommend anyway, because the laptop ASR stack is the one with six clean validation runs behind
it. If ASR must share the card, set `GPU_UTIL=0.85`. Keep `--max-num-seqs 64` (hybrid Mamba models
reject the default 1024).
**Test.** Run `local_evaluator.py` end to end against the assembled endpoint over the full 39
training conversations twice in a row, watching `nvidia-smi`, and read the "Round trip" and
"timeouts" lines. That is 78 conversations of real HTTP load, more than the attempt itself.

**Evidence.** `bench/llm/serve_vllm.sh` (the dense-27B advisory block, `--max-num-seqs 64`);
`research/07-findings-log.md` entries 34, 35 (0 request errors at ten workers);
`model.py:356-358`; `example.py:47-56`; `bench/hpc/RUNPOD.md` step 3.

### 13. Failure mode: a malformed or wrong-length body

**Claim.** The defences here are in good shape and should be left alone. `example.predict` checks
the lengths itself and falls back to `[True] * n, [None] * n` on any exception
(`example.py:51-56`), `dtos.ASRQuestionResponseDto` validates that the three lists line up, and
`api.py:39` calls `utils.validate_response` before returning. One sharp edge: `validate_response`
runs *after* `predict` has already built a reply, and it raises, which makes uvicorn return a 500.
That is the intended "fail loudly" design and it is right for validation, but during the one
evaluation attempt a 500 throws away a reply that was worth about five marks in expectation.

**Likelihood.** Low. The only realistic trigger is a NaN or an inverted interval reaching the DTO,
and `span_from_ids` clamps both (`model.py:328-331`).
**Consequence.** Ten marks for that conversation, and the counter resets, so the attempt survives
(README: "a 500 ... proves you are alive").
**Mitigation.** Keep `validate_response` but catch it in `api.py` during the attempt and, on
failure, return the all-guess body instead of a 500, logging loudly.
**Test.** `python local_evaluator.py --oracle` must print 1.000 (harness sanity), and point
`LLM_URL` at a stub that returns truncated JSON, an empty body and a 500, confirming the endpoint
still answers 200 with ten booleans inside the deadline.

**Evidence.** `example.py:44-62`; `dtos.py` (`evidence_matches_answers`); `utils.validate_response`;
`api.py:32-41`; `README.md` "Scoring" and "OBS".

### 14. Failure mode: five consecutive timeouts, and what has actually happened to us

**Claim.** The only 0.0000 in our history came from this family. Entry 13 records the 15:45
validation attempt: the server had crashed two minutes earlier, a Windows fail-fast in
`ucrtbase.dll`, and every request came back 502, score 0.0000. The response was a supervisor loop
around `api.py`, which lives in `scratchpad/serve_forever.sh` and is not in the repository at all.
So the single mitigation for the single worst thing that has actually happened to this deployment
is an untracked file in a scratch directory.

**Likelihood.** Moderate. It has happened once in one day of serving.
**Consequence.** With 502s, every conversation is scored wrong but the attempt is not aborted (a
502 is a reply). With a dead process and a closed port, it is silence, five in a row, and the
attempt ends where it stood.
**Mitigation.** Commit the supervisor into the repository (`bench/hpc/serve_forever.sh` or
`deploy/`), supervise `cloudflared` the same way, and give the restart path a bounded warm-up so a
restarted process answers fast rather than binding late. Plus finding 5's deadline, which keeps a
slow process from ever looking silent.
**Test.** Kill `python api.py` in the middle of a validation run and confirm from
`bench/portal_status.py --watch` that the run still completes with a plausible score. This is the
most valuable single validation run available this weekend: unlimited attempts, and it rehearses
the thing that has already cost us a run.

**Evidence.** `research/07-findings-log.md` entry 13; the memory note on `serve_forever.sh`;
`README.md` "Five timeouts in a row ends the attempt".

### 15. Failure mode: the whole-attempt average, and the pod restart that fits inside it

**Claim.** The whole-attempt budget for 38 conversations is 2,280 s. At the few-shot topology's
roughly 8 s per conversation, the attempt consumes about 5 minutes of it, leaving over half an hour
of slack; at joint-demo's 15 s, about 10 minutes. That slack is large enough to absorb one full
vLLM restart (8 minutes per `RUNPOD.md`) without breaching the average, *provided* the requests
during the restart return guesses quickly instead of blocking. That is the second reason for
finding 5's deadline: it converts the restart from an attempt-ending event into a bounded loss of
a few conversations.

**Likelihood.** Low that the average is breached by our own latency; the risk is entirely in
stalls.
**Consequence.** If the average is breached, the queued conversations are never sent and are scored
wrong, with no silence needed.
**Mitigation.** The 40 s deadline, plus a rule: no configuration changes, no restarts, no
`nvidia-smi`-heavy debugging on the serving box once the attempt is queued.
**Test.** The two back-to-back `local_evaluator.py` runs of finding 12 report the mean and worst
round trip directly; read the worst, not the mean.

**Evidence.** `README.md` "Timing"; `local_evaluator.py:56`
(`ATTEMPT_BUDGET_SECONDS_PER_CONVERSATION`); the per-variant walls in finding 7;
`bench/hpc/RUNPOD.md` restart procedure.

### 16. The probe endpoint and the real endpoint differ by one port, and the probe would score the evaluation set at the floor

**Claim.** `bench/mine/probe_server.py` serves `/predict` on port 9055 from
`bench/mine/current_answers.json`, a filename-to-answers table for the 19 validation conversations,
with null spans, and answers all-no for unknown files. `api.py` serves `/predict` on 9054. If the
tunnel points at 9055 during the evaluation attempt, all 38 evaluation conversations are unknown
files, every answer is no, every span is null, and the score is 0.4 times 0.5, that is 0.200, the
all-constant floor, from an endpoint that returns 200 instantly and looks perfectly healthy. This
is the most catastrophic confusion currently available and it is one character in a
`cloudflared --url` argument.

**Evidence.** `bench/mine/probe_server.py` (`PORT` default 9055, unknown files get all-no,
`GET /` returns "probe endpoint running"); `api.py:22-23,52-54` (port 9054, `GET /` returns
"Your endpoint is running!"); `research/07-findings-log.md` entry 31 (the mined 1.0 came from that
table).

**Severity.** Critical, and trivially preventable.

**Action.** Kill `probe_server.py` and anything else listening on 9055 before the attempt, and
check identity rather than liveness: `GET /` must return "Your endpoint is running!". Better, spend
ten minutes extending `GET /api` in `api.py` to report the served identity: git commit, uptime,
`ASR_MODEL`, `START_OFFSET`/`END_OFFSET`, `LLM_URL`, `LLM_MODEL`, prompt variant, few-shot pool
size, `SPAN_ON_NO`, `TRANSCRIPT_CACHE`, and whether the LLM answered a probe. Then the entire
"is the right thing deployed" family of risks collapses into one curl, which is the difference
between a checklist that is executed and one that is remembered.

### 17. `TRANSCRIPT_CACHE=1` in production would serve a cached transcript for a conversation we have never heard

**Claim.** `model._cached_transcript` returns `transcripts/<stem>.<ASR_MODEL>.json` whenever
`TRANSCRIPT_CACHE=1`, keyed only on the incoming `audio_filename`
(`model.py:129-143`). It defaults to off, which is correct, but it is a development convenience one
environment variable away from serving the wrong audio's transcript. The risk is currently
theoretical because the hidden sets appear to use the 56 sample ids we do not have
(`research/06-edge-ideas.md` item 48: 39 of 1..95 used, 56 unused, 19 plus 38 hidden), but it
depends on an inference about the organisers' numbering, not on anything we control.

**Evidence.** `model.py:68,129-143`; `research/06-edge-ideas.md` item 48.

**Severity.** Low likelihood, total consequence. Worth one line in the pre-flight.

**Action.** Assert `TRANSCRIPT_CACHE` is off in the serving process and show it in `GET /api`.

### 18. The pod cannot be stopped from code, the balance is shared, and a billing stop mid-attempt is unrecoverable

**Claim.** Elias's RunPod login has no API key, so nothing can start or stop the pod
programmatically; everything is manual through the UI or SSH. The H100 bills $3.49/h against a
$150 balance shared with Oscar's three `nordic-a100-*-overnight` pods. If that balance reaches zero
during the attempt, RunPod stops the pods, the endpoint disappears, and the attempt ends in
silence with no way to intervene in time. Left running unattended from Friday evening, one H100
alone is about 43 hours of balance, and Oscar's A100s are drawing on the same pot at the same time.

**Evidence.** `bench/hpc/RUNPOD.md` ("No API key", "$3.49/h", "Oscar's own pods ... bill the same
$150 balance", "Without an API key nothing can stop it from the inside"); the memory note (balance
$150 at 2026-09-17 22:50).

**Severity.** Critical, and invisible until it happens.

**Action.** Read the balance immediately before queueing and require a floor of at least $30, which
is about eight hours of H100. Ask Oscar tonight for two things: an API key (Settings, API Keys, his
account only), and an agreement that his overnight pods are stopped or capped during our Saturday
window. Set an alarm: the pod must not be left running after the attempt, because nothing will stop
it for us. If the balance cannot be secured, prefer Oscar's A100s with the few-shot prompt, or fall
back to the laptop, both of which have no billing cliff.

### 19. The only copies of the pod's serving scripts are on the pod

**Claim.** `bench/hpc/RUNPOD.md` refers to `/workspace/serve27b.sh`, `/workspace/bench27b.sh` and
`/workspace/run_benches.sh`. None of them exist in the repository, and `bench/hpc/` holds only
`env.sh`, `sync.sh` and the two LSF files. The network volume is the single copy, and the venv on
it is already documented as broken (`/workspace/venv-vllm`, "treat as broken"), so the volume is
not a place to keep the only copy of anything.

**Evidence.** `bench/hpc/RUNPOD.md` lines 5-11, step 3; `ls bench/hpc/`.

**Severity.** Medium.

**Action.** First thing in the next pod session, before any benching, copy those three scripts back
into `bench/hpc/` and commit them. They also encode the settings that finding 8 depends on
(`HF_HOME`, cache roots, `--max-model-len`, `--max-num-seqs`), so having them under review is worth
more than having them work.

### 20. Calendar risk: the cluster closes Friday 20:00 and Saturday has no rehearsal time

**Claim.** The DTU service window runs Friday 2026-09-18 20:00 to Monday 08:00, so the H100 queue
is unavailable for the entire final day, and the deadline is Saturday 16:00. That leaves the RunPod
pod as the only GPU for both the three pending benches and the deployment. Those are competing uses
of the same card and the same balance.

**Evidence.** `bench/hpc/HANDOFF.md` "Hard constraints"; `bench/hpc/RUNPOD.md` (three pending
benches: `units-fewshot`, `units-joint-demo-fewshot`, `units-joint-demo-all`);
`research/07-findings-log.md` entry 35 ("to be tested on the RunPod H100 rather than the cluster
queue").

**Severity.** High, as a scheduling decision to take Friday morning rather than Saturday.

**Action.** Split Friday and Saturday cleanly. Friday is for measurement: restart the pod, run the
three pending benches plus the corrected-offset rerun of finding 9, and decide the variant by
Friday evening. Saturday is for deployment only: no new variants, no new models, one code change
set landed before noon, two clean validation runs, then the attempt. If the variant decision is not
made by Friday 22:00, serve `units-fewshot` and stop arguing; it is within noise of everything else
and it is the safest thing to operate.

### 21. Two smaller things that will bite

**Claim (a).** The `Dockerfile` cannot build the current server: it copies only `dtos.py utils.py
example.py api.py`, while `example.py` imports `model`, and `model.py` needs `data/` and
`transcripts/` for the few-shot pool. Anyone who reaches for containerisation on Saturday under
time pressure will lose half an hour to an `ImportError`.

**Claim (b).** The evaluation attempt is queued by hand, by Elias, because
`.claude/hooks/block-evaluation.py` denies any agent call that mentions the evaluation queue, and
the API key lives in his browser. That guardrail is correct and should stay, but it means the final
action is a human typing a URL, with no automation to verify it. The portal also allows only one
attempt in flight at a time, and the Hugging Face miner Space can queue validations on its own
(`MINER_AUTOSTART`, the `/control/<token>/{pause|resume|status}` endpoints); it is documented as off,
but "documented as off" is not "checked as off".

**Evidence.** `Dockerfile`; `.claude/hooks/block-evaluation.py` and `.claude/settings.json`;
`bench/mine/space/app.py` (miner thread plus `/control`); `README.md` "Validation and evaluation"
(one attempt going at a time); the memory note on the Space.

**Severity.** Low and medium respectively, both cheap to close.

**Action.** Either fix the `Dockerfile` (add `model.py`, `data/question_train.csv`, the 39 turbo
transcripts) or delete it from the deployment plan so nobody reaches for it. And check the Space's
`/control/<token>/status` before queueing anything serious, as the memory file already instructs.

## Pre-flight checklist for the single evaluation attempt

Run in this order. Every step has a check you can read, and the go/no-go is stated. Steps 1 to 6
are Friday; 7 onwards is Saturday morning. Nothing in this list is run by an agent: the queueing
steps are Elias's, by the guardrail.

1. **Freeze the variant.** Restart the pod per `bench/hpc/RUNPOD.md`, copy `serve27b.sh`,
   `bench27b.sh`, `run_benches.sh` back into `bench/hpc/` and commit (finding 19), then run the
   three pending benches plus one rerun of the leader with `ASR_MODEL=large-v3-turbo` exported
   (finding 9). Go: a variant is chosen and its per-conversation wall on the card you will actually
   serve from is known. No-go: default to `units-fewshot` and move on.
2. **Land the code changes as one reviewed set** (findings 1 to 6): the OpenAI client with the
   `json_object` fallback and `parse_json`, the prompt classes moved into `model.py`, the pool
   assertion with a loud fallback to plain `units`, the 40 s deadline in `example.predict`, the
   pooled `Session` with split connect and read timeouts, the 4B in-process fallback, and the
   extended `GET /api` (finding 16). Go: `python local_evaluator.py --oracle` prints 1.000 and
   `python local_evaluator.py` over the 39 training conversations scores at or above 0.70 with zero
   timeouts. No-go on anything below 0.68: that is a regression against today's served 0.6759 and
   the cause must be found before deploying.
3. **Ship the pool.** Confirm `data/question_train.csv` and the 39 `*.large-v3-turbo.json`
   transcripts are present on whichever box runs `model.py`, and that `GET /api` reports a
   few-shot pool of at least 150 positive examples (finding 3). Go: the number is there and
   plausible. No-go: do not serve the few-shot prompt with a short pool; serve plain `units`.
4. **Stand up the serving topology and leave it alone.** ASR on the laptop, vLLM on the pod with
   `HF_HOME` and `XDG_CACHE_HOME` under `/workspace`, `--max-num-seqs 64`, `GPU_UTIL` 0.92 if the
   card is vLLM-only and 0.85 if it is shared (findings 8, 12). Go: `/health` answers and a warm-up
   request returns a valid JSON answer in under 10 s.
5. **Fix the public path.** One stable hostname, supervised alongside the server; write the exact
   URL, path included, into a file (finding 10). Go: `GET <url>/` returns "Your endpoint is
   running!" (not "probe endpoint running", finding 16) from a device off the local network.
6. **Test the body size from outside.** `curl -X POST` the largest training body (4.95 MB base64)
   at the public URL from a phone hotspot and read the status and the round trip (finding 11).
   Go: 200 with ten booleans, round trip under 20 s. No-go on a 413 or a 502: remove the proxy hop
   or raise its limit, then repeat.
7. **Soak it.** `python local_evaluator.py --url <public url>` over the 39 training conversations,
   twice back to back, watching `nvidia-smi` on the pod (finding 12). Go: zero timeouts, worst
   round trip under 25 s, score within 0.02 of step 2's local number. No-go on any timeout: the
   deadline of finding 5 is not doing its job.
8. **Rehearse the two failures that have happened or can.** During one validation run, kill
   `python api.py` and let the supervisor restart it; during another, stop vLLM on the pod
   (findings 6, 14). Go: both runs complete, the first with a score near normal, the second with a
   score near the 4B's 0.6759, and `bench/portal_status.py` shows no errors on either. No-go: the
   fallback and the supervisor are the two things standing between a bad minute and a lost attempt.
9. **Clear the field.** Kill `probe_server.py` and anything on 9055; confirm the miner Space is
   paused via `/control/<token>/status`; confirm no validation attempt is in flight in
   `bench/portal_status.py`; confirm `n_evaluations` is 0 (findings 16, 21).
   Go: all four clean.
10. **Check the money and the clock.** Read the RunPod balance and require at least $30 remaining;
    confirm Oscar's overnight A100s are stopped or accounted for; confirm the local time leaves at
    least 90 minutes before the 16:00 deadline (findings 18, 20). No-go on a thin balance: serve
    from the laptop instead.
11. **Two clean validation runs on the exact final URL, from the exact final process.** Back to
    back, no restarts in between, and read the score, the per-request errors and the timing from
    `bench/portal_status.py`. Go: both runs complete with no errors and both scores within 0.02 of
    each other and at or above 0.70. No-go: fix, then repeat. Two runs, not one: entry 26 shows
    validation-to-validation noise of about 0.01, so a single number cannot distinguish a lucky
    run from a stable one.
12. **Freeze.** From this point nothing is restarted, redeployed, reconfigured or debugged on the
    serving boxes. Check `GET /api` uptime and confirm it covers both validation runs, which proves
    no restart happened in between (finding 16).
13. **Queue the evaluation attempt.** Elias only, from the browser, with the URL pasted from the
    file written in step 5, `/predict` included. Then watch `bench/portal_status.py --watch` and do
    nothing else: with 38 requests at about 8 to 15 s each the attempt should finish in ten to
    fifteen minutes.
14. **After.** Confirm `n_evaluations` reads 1 and the score is recorded, keep the request dump for
    the post-mortem, then stop the pod in the RunPod UI, because nothing else can (finding 18).

### Rollback plan

Three levels, cheapest first, because the expensive one cannot be exercised mid-attempt.

1. **Per request, automatic (preferred).** The in-process 4B fallback of finding 6. If the pod does
   not answer within the read timeout, the same endpoint answers from local Ollama `qwen3:4b` with
   the `units` prompt, the configuration that scored 0.6759 on validation run F. Nothing changes on
   the outside, the URL stays valid, and the loss is tIoU on the affected conversations only. This
   is the only rollback that works after the attempt is queued.
2. **Before queueing, by URL.** If the 27B endpoint fails any go/no-go above, submit the laptop
   endpoint: `api.py` on 9054 with `ASR_MODEL=large-v3-turbo`, `LLM_URL=http://localhost:11434`,
   `LLM_MODEL=qwen3:4b`, `START_RULE=first-word-end`, `SPAN_ON_NO=1`, `TRANSCRIPT_CACHE=0`, under
   the supervisor, behind the same tunnel. Expected score about 0.6759, which is 17th on the
   validation board of entry 31 but is a real number from a proven stack, and it is worth far more
   than a 0.78 configuration that did not finish. Decision point: if step 11 has not passed by
   Saturday 13:00, take this.
3. **Degraded but alive.** If both answering backends are unreachable, the current code already
   returns ten guesses with no spans, worth about 0.200 per conversation, and keeps the attempt
   running (finding 5). That is the floor to protect, and it is protected by the deadline, the
   supervisor and the stable hostname, not by anything about the model.

### The three things I would do first, if only three get done

1. The 40 s deadline plus the 4B in-process fallback (findings 5 and 6). It bounds every other
   failure in this document.
2. Ship the few-shot pool with an assertion and a loud fallback (finding 3). It is the one failure
   that would look like success all the way to the final score.
3. A stable public hostname, plus the body-size curl from off-network, plus killing the probe
   server (findings 10, 11, 16). Three cheap checks against three total losses.
