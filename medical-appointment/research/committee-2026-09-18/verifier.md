# Verifier report, 2026-09-18

Scope: what is about to be served (`model.py`, `example.py`, `api.py`, `bench/llm/prompts.py`, the
laptop `serve.env` + `bench/hpc/serve_forever.laptop.sh`, and the pod scripts
`bench/hpc/pod_bootstrap.sh` / `pod_endpoint.sh`), ranked by expected damage to the one evaluation
attempt. Every claim below was checked by reading the code or by an offline run on the training set;
nothing from `bench/mine/` was read, no server was touched, no endpoint was called. Scratch harness:
`<scratchpad>/verifier/` (`u1.py`, `u2.py`, `u4.py`, `u5.py`, `u6.py`, `u7.py`, `u8.py`, `u9.py`,
`u10.py`).

## Ranked findings

| # | Sev | What | Evidence | Fix |
|---|---|---|---|---|
| 1 | High | Moving the endpoint onto the pod (`pod_endpoint.sh`) puts faster-whisper **and** Ollama `qwen3:4b` on a card where vLLM already holds `--gpu-memory-utilization 0.92` of 80 GB. About 6.4 GB is left for 5-7 GB of new allocations; a CUDA OOM in `_load_asr()` is an unhandled exception at import and turns the endpoint into a 3-second restart loop. | `pod_bootstrap.sh` lines 20 and 48 (`GPU_UTIL=0.92`); findings log entry 49 (A100 SXM 80 GB, 52 GB of weights, util 0.92); `pod_endpoint.sh serve` runs `ollama_up` then api.py; `model.warm_up` calls `_load_asr()` outside every `try`, and the exception propagates through `example.py`'s module body and `api.py`'s `from example import predict`. | Before switching the submitted URL: start vLLM with `GPU_UTIL=0.80` (or `MAX_LEN=16384 GPU_UTIL=0.85`), bring vLLM up **first**, then Ollama, then api.py, and check that `nvidia-smi` shows more than 3 GB free after api.py's warm-up. Keeping the endpoint on the laptop avoids the class entirely. |
| 2 | High | The pod endpoint path is unproven end to end: a 4.72 MB POST body has never gone through the RunPod HTTP proxy, and `pod_endpoint.sh install` never runs a CUDA transcription, so a broken ctranslate2/cuDNN pairing first shows up as a crash loop at serve time. | Largest training body computed at 4.72 MB (`data/audio` max 3,710,684 B, 4,947,580 base64 chars); `install()` builds the model with `device='cpu'` and only prints versions; committee report 02 already flagged 1 MB proxy defaults. | Smoke-test on the pod before submitting: one `/predict` with the largest training MP3 through `https://<POD_ID>-9054.proxy.runpod.net/predict`, and one CUDA transcription from `venv-api` in the same shell that runs `serve`. Pin `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` to the versions a working install resolved. |
| 3 | Medium | A conversation that hits the 55 s wall is **not** cancelled. `example._pool` has `max_workers=4`, so four overrunning conversations saturate the pool and every later conversation is answered with `[True]*n, [None]*n` without the model ever running. Five such in a row ends the attempt. The abandoned threads also re-enter `model._asr`, one shared `WhisperModel` with no lock. | `example.py`: `_pool = ThreadPoolExecutor(max_workers=4, ...)` and `future.result(timeout=PREDICT_DEADLINE)` which abandons the result, not the work; `model._asr` is a module global created once in `_load_asr`. | Lower `PREDICT_DEADLINE` to about 48 s (the longest legitimate internal path is `LLM_DEADLINE + _FALLBACK_GRACE` = 52 s from `t0`, so 48 s only cuts genuinely stuck ASR), and/or serialise `answer_all` behind a lock so a stuck conversation cannot contend with the next one. |
| 4 | Medium | **Warm-up produces no log at all.** `api.py` calls `logging.basicConfig` *after* `from example import predict`, so every `logger.info` inside `model.warm_up()` is dropped by a root logger with no handlers. The runbook's instruction to check `api.log` for `LLM backend vllm at ...` and `LLM warm in ...` can never succeed. | `api.py` import order (basicConfig on line 25, `from example import predict` on line 19); `grep -c "few-shot pool\|LLM warm\|ASR warm\|LLM backend" api.log api.log.err` returns 0 in both, while post-startup `INFO:model:conversation_...: 251 words, 53 units, ASR 7.5s` lines are present in the same file. | Pre-flight from `GET /api` instead of the log: `model.status()` reports `llm_model`, `llm_variant`, `unit_split`, `span_on_no`, `fewshot_pool.{examples,positives}`, `breaker_open` and the counters. A one-line fix (move `basicConfig` above the imports) exists but is a code change on the day. |
| 5 | Medium | Silent downgrade to the 4B: `ask_primary` skips the pod whenever `remaining - 12 < 2`, i.e. whenever ASR took more than 26 s, and neither that path nor the breaker-open path increments `primary_failed` or sets `last_primary_error`. A long evaluation audio answers a whole conversation with the 4B and `/api` shows only `fallback_ok` rising. | `model.ask_primary`: both `raise TimeoutError` branches sit before the `try` that does `_count('primary_failed')`. Training audio is 74-232 s and live ASR was 1.6-8.6 s (`api.log.err`), so the margin holds for training-length files only. | Accept, but watch `counts.fallback_ok` in `/api` during the attempt; `ask` does log `primary LLM failed (...)` at WARNING, which does reach `api.log.err`. If evaluation audio is much longer than training audio, raise `LLM_DEADLINE` (the 60 s budget leaves room; 40 s is conservative). |
| 6 | Medium | First-conversation risk on the HTTPS proxy: `warm_llm` opens exactly one pooled connection, then the first real conversation fires ten concurrent requests, i.e. nine fresh TLS handshakes, against a 3.05 s connect timeout. One slow handshake raises `ConnectionError`, which `_Breaker.trip` reads as "pod unreachable" and sends **every** question for the next 20 s to the 4B. | `model._session` pooling; `_chat_openai` uses `timeout=(3.05, max(1.0, timeout))`; `_Breaker.trip` fires on `requests.ConnectionError`/`Timeout` with `hold=20.0`; `serve.env` uses `https://...proxy.runpod.net/v1` (round trip about 1.0 s, entry 49). | Warm with ten parallel requests instead of one, or raise the connect timeout to about 8 s. Not observed in the 39-conversation soak (0 fallbacks), so this is a tail risk, not a defect. |
| 7 | Low | The `abs(i - qi) <= 2` window in `anchor_ids` is now a small net negative for the served model under clause units: removing it is **+0.0024 +- 0.0029** score on both 3.8-27B clause-and runs. It is a net positive for the 3.6-27B (-0.0086 +- 0.0042 to remove), so the constant is model-dependent and inside noise. `locate_quote` never contributes an id the model did not already cite (0 of 390). | `u8.py`: served `win=2` 0.8125 / 0.8179, no window 0.8150 / 0.8203 (3.8-27B H100/A100); 0.8139 to 0.8052 (3.6-27B); `u2.py`: "qi not among cited: 0". | Leave it. The gain is below the run-to-run noise (0.0053, finding 9) and changing it would serve an untested configuration. |
| 8 | Low | The served prompt tells the model the evidence is "the single utterance that states the fact" while the transcript it sees is cut into clause pieces. `units-fewshot-cl` exists to fix exactly that and measured +0.0019 +- 0.0046, i.e. nothing. | `prompts._FEWSHOT_NOTE` vs `_CLAUSE_NOTE`; replay of `qwen3.8-27b.units-fewshot-cl.large-v3-turbo.clause-and.json` = 0.8144 against 0.8125. | Nothing to do; recorded so nobody "fixes" the prompt tonight. |
| 9 | Low (statistics) | Entry 48's `+0.0151 +- 0.0089` reproduces exactly, but the bar understates the uncertainty: the two arms are **different runs** (sentence = 2026-09-17 cluster job, `--max-tokens 200`; clause-and = 2026-09-18 job, `--max-tokens 1500`), and the same configuration re-run on different hardware shifts by **+0.0053 +- 0.0026 (z = 2.05)**. Selection optimism from picking clause-and among the modes is negligible: **+0.0011**. | `u6.py` cluster bootstrap over the 39 conversations; `u7.py` best-of-three optimism, with clause-and the argmax in 90.9 % of resamples. | Quote the clause-and gain as "+0.015, roughly +-0.010 once run-to-run drift is included; sign supported by three runs, two models, two prompts and a pre-registered oracle ordering". Do not quote 0.8125 / 0.8179 as a predicted evaluation score. |
| 10 | Low | The two supervisors diverge: the laptop script `unset`s every override before sourcing `serve.env`, the pod one does not, so a variable removed from `/workspace/serve.env` keeps its exported value across restarts. Neither sets `HF_HUB_OFFLINE=1`, so `WhisperModel(...)` contacts huggingface.co on every restart although the weights are cached. | `serve_forever.laptop.sh` has an explicit `unset` block; `pod_endpoint.sh serve` only exports. `HF_HUB_OFFLINE` is set in `pod_bootstrap.sh` (vLLM) and in `clause_bench.lsf`, nowhere in either serve loop. | Export `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` in both serve loops; mirror the `unset` block into `pod_endpoint.sh`. |
| 11 | Low | `pod_bootstrap.sh` defaults (`MAX_LEN=73728`) do not match the server that produced 0.8179 (`--max-model-len 16384`, entry 49). A pod restart run "as documented" serves a different configuration from the measured one. | `pod_bootstrap.sh` line 19 against findings log entry 49. | `MAX_LEN=16384 bash pod_bootstrap.sh serve`, or change the default. It also frees KV cache, which helps finding 1. |

## Findings in detail

### 1. VRAM contention if the whole endpoint moves onto the pod

`bench/hpc/pod_bootstrap.sh`:

```
GPU_UTIL=${GPU_UTIL:-0.92}
...
nohup "$VENV/bin/vllm" serve "$MODEL" --port "$PORT" --host 0.0.0.0 \
    --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" --gpu-memory-utilization "$GPU_UTIL" \
```

Findings log entry 49: one A100 SXM 80 GB, 52 GB of weights, `--gpu-memory-utilization 0.92`.
0.92 x 80 GB = 73.6 GB, leaving about 6.4 GB. `pod_endpoint.sh serve` then adds, on the same card,
`ollama serve` with `OLLAMA_NUM_PARALLEL=4 OLLAMA_CONTEXT_LENGTH=6144` holding `qwen3:4b` (about
3-4 GB with that KV budget) and faster-whisper `large-v3-turbo` in float16 (about 1.6 GB of weights
plus cuDNN workspace). That is 5-7 GB against 6.4 GB.

The failure is not graceful:

```python
def warm_up() -> None:
    if not TRANSCRIPT_CACHE:
        model = _load_asr()          # no try/except anywhere above this
```

`_load_asr` raising propagates out of `warm_up()`, out of `example.py`'s module body
(`model.warm_up()` at import), out of `api.py`'s `from example import predict`, and uvicorn never
starts. The supervisor then restarts every 3 s forever with only a traceback in the log. The same
holds on the laptop.

Reproduction: none run (it would mean touching the pod). The arithmetic comes from the committed
flags and entry 49's own numbers.

### 2. Untested body size and ASR install on the pod

The largest supplied conversation is 3,710,684 bytes of MP3, i.e. 4,947,580 base64 characters, a
JSON body of 4.72 MB (computed over `data/audio/*.mp3`; durations 74-232 s). The laptop endpoint is
reached directly (its access log shows `192.38.81.6` and `205.169.39.42` doing `GET /`), so nothing
proxies it today. The pod path introduces `https://<POD_ID>-9054.proxy.runpod.net`, which has never
carried a body that size in this project.

`pod_endpoint.sh install` ends with

```python
m = WhisperModel('large-v3-turbo', device='cpu', compute_type='int8')   # weights only; CUDA is exercised by api.py's warm-up
```

so `LD_LIBRARY_PATH`, cuBLAS and cuDNN are first exercised by the process that must not fail.
`libpath()` also lists `nvidia/cuda_runtime/lib` and `nvidia/cuda_nvrtc/lib`, which `install` never
installs. Harmless in itself (non-existent entries are ignored), but it means the path list is not
evidence that the right libraries are present.

### 3. The 55 s wall does not cancel anything

`example.py`:

```python
_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='predict')
...
        future = _pool.submit(model.answer_all, audio_bytes, request.audio_filename, request.questions)
        try:
            answers, spans = future.result(timeout=PREDICT_DEADLINE)
        except FutureTimeout:
            global timed_out
            timed_out += 1
            raise TimeoutError(f'answer_all still running after {PREDICT_DEADLINE:.0f} s; answering with guesses')
```

`future.result(timeout=...)` abandons the result, not the work. The worker keeps one of the four
slots and keeps calling `model._asr.transcribe` on a shared `WhisperModel` with no lock. Four
consecutive overruns leave no slot for the fifth conversation, whose `submit` queues and whose
`result` times out without the model having run at all; `local_evaluator` mirrors the service's
"five timeouts in a row ends the attempt".

Bounds that make 48 s safe: `answer_all` sets `deadline = t0 + LLM_DEADLINE` (40 s) and
`ask_fallback` gets `deadline + _FALLBACK_GRACE - now` (12 s), so the longest legitimate path ends
52 s after `t0`. The longest observed round trip in the soak was 16.3 s.

### 4. Warm-up logging is dead

`api.py`:

```python
from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from example import predict          # line 19 - runs model.warm_up()
from utils import validate_response

HOST = '0.0.0.0'
PORT = 9054

logging.basicConfig(level=logging.INFO)   # line 25
```

Reproduction:

```
$ grep -c "few-shot pool\|LLM warm\|ASR warm\|LLM backend" <scratchpad>/api.log <scratchpad>/api.log.err
api.log:0
api.log.err:0
```

while post-startup INFO records are present in the same file
(`INFO:model:conversation_sample_33.mp3: 251 words, 53 units, ASR 7.5s`). `logger.exception` in the
warm-up *is* visible, because Python's last-resort handler emits WARNING and above; so a warm-up
failure shows and a warm-up success does not. `GET /api` carries everything the pre-flight wants,
including `fewshot_pool: {examples, positives}` and `unit_split`.

### 5. `ask_primary`'s skip paths are invisible to the counters

```python
    budget = min(LLM_TIMEOUT, remaining - (_FALLBACK_RESERVE if LLM_FALLBACK_MODEL else 0.0))
    if _breaker.open():
        raise TimeoutError('primary LLM skipped: breaker open after a transport failure')
    if budget < 2.0:
        raise TimeoutError(f'primary LLM skipped: {remaining:.1f} s left before the deadline')
    try:
        out = _chat_openai(p.system, p.user, p.schema, getattr(p, 'demos', None), budget)
        _count('primary_ok')
        return out
    except Exception as exc:
        _breaker.trip(exc)
        _count('primary_failed')
```

Both `raise` statements sit before the `try`, so `primary_failed` and `last_primary_error` keep
their old values. With `LLM_DEADLINE=40` and `_FALLBACK_RESERVE=12` the crossover is ASR > 26 s.
Live ASR on the 39 training files was 1.6-8.6 s, so the margin is about 3x on training-length audio.

### 6. One warm connection, ten cold ones

`warm_llm` sends a single request; `answer_all` then fans out to
`ThreadPoolExecutor(max_workers=len(questions))`. `_chat_openai` uses `timeout=(3.05, budget)`, and
the 3.05 s covers connect *and* TLS handshake for an `https://...proxy.runpod.net` URL. A
`requests.ConnectionError` from any of the nine new connections calls `_breaker.trip`, which holds
the breaker open for 20 s, roughly two conversations answered entirely by the local 4B. The soak
(0 fallbacks over 39 conversations) is one clean observation, not a proof.

### 7. `anchor_ids` / `locate_quote` under clause units, measured

Served rule:

```python
def anchor_ids(out: dict, units: List[Unit]) -> List[int]:
    cited = [int(i) for i in out.get('segments', []) if isinstance(i, (int, float)) and 0 <= int(i) < len(units)]
    qi = locate_quote(str(out.get('quote', '')), units)
    if qi is None:
        return cited
    keep = [qi] + [i for i in cited if abs(i - qi) <= 2 and i != qi]
    return sorted(set(keep))
```

Replaying the stored raw answers with the window removed (`u8.py`; score = 0.4 x accuracy +
0.6 x mean tIoU; cluster bootstrap over the 39 conversations):

| run | served `win=2` | no window | difference |
|---|---:|---:|---:|
| 3.8-27B few-shot clause-and (H100) | 0.8125 | 0.8150 | +0.0024 +- 0.0029 |
| 3.8-27B few-shot clause-and (A100) | 0.8179 | 0.8203 | +0.0024 +- 0.0029 |
| 3.6-27B few-shot clause-and | 0.8139 | 0.8052 | -0.0086 +- 0.0042 |
| 3.8-27B few-shot sentence | 0.7974 | 0.7971 | -0.0003 +- 0.0030 |

Supporting facts, all on the served 3.8-27B clause-and run (`u2.py`, `u8.py`):

- `locate_quote` returned `None` for 63 of 390 answers; 60 of those had an empty or sub-8-character
  quote (the "no" answers). Only **3** answers had a real quote that failed the 0.5 difflib ratio,
  so the threshold is not a problem on short clause pieces (under sentence units it was 1).
- The quote's unit was among the cited ids in **every** case where both existed, so the quote anchor
  never adds an id: `win=99` and "cited ids only" score identically (0.6984 mean tIoU).
- It does rescue 2 answers where the model cited nothing but quoted something.
- Using the quote's unit alone costs -0.068 tIoU, so the citation list, not the quote, does the work.
- The model quoted something longer than the piece it matched (a whole sentence against a clause) in
  59 of 390 answers under clause-and, against 23 under sentence units; the extra pieces come back
  through the cited ids, which is why the window costs so little.

### 9. The clause-and statistics

`u6.py` resamples the 39 conversations with replacement 4000 times and recomputes
`0.4 x accuracy + 0.6 x mean tIoU` from the replayed per-question rows:

| comparison | difference | cluster-bootstrap SE | z |
|---|---:|---:|---:|
| clause-and - sentence (3.8-27B few-shot) | +0.0151 | 0.0088 | 1.72 |
| clause-all - sentence | +0.0094 | 0.0076 | 1.23 |
| clause - sentence | +0.0050 | 0.0062 | 0.81 |
| **same config, A100 - H100** | **+0.0053** | **0.0026** | **2.05** |
| units-fewshot-cl - units-fewshot (both clause-and) | +0.0019 | 0.0045 | 0.42 |

Entry 48's `+- 0.0089` reproduces. What it does not contain: the sentence arm came from a different
job on a different day with `--max-tokens 200`, and the A100-against-H100 row shows that a pure
re-run of the identical configuration moves the score by 0.005 with high confidence. Widen the
interval accordingly; the sign is still well supported (three runs, two models, two prompt designs,
and the three modes ordering themselves exactly as the entry-47 oracle predicted before the LLM ran).

Selection optimism (`u7.py`): in each bootstrap resample, take the best of `clause`, `clause-all`,
`clause-and` against `sentence` and compare it with that mode's full-sample difference. Mean
optimism **+0.0011** (sd 0.0083); `clause-and` is the argmax in 90.9 % of resamples, `clause-all` in
5.2 %, `clause` in 3.8 %. The winner's curse is negligible because the modes are nested and ordered.

## Checked and found correct

1. **Live units are the clause-and units.** Rebuilding units from the cached turbo transcripts and
   comparing with the unit counts the live endpoint logged for all 39 training conversations
   (`u4.py`): mean |live - clause-and| = **0.46** units, mean |live words - cached words| = 1.49 of
   about 300. The live path really is running `UNIT_SPLIT=clause-and`, and live ASR matches the
   transcripts the bench measured, so the bench number transfers.
2. **The fitted edge offsets are still right under clause units.** Grid search over START_OFFSET in
   [-0.70, +0.40] and END_OFFSET in [-0.50, +0.50] on the clause-and run gives (-0.20, 0.00) at
   0.6951 against the served (-0.20, -0.02) at 0.6944; a leave-one-conversation-out refit picks
   (-0.20, 0.00) for **39 of 39** held-out conversations and gains **+0.0007** tIoU (`u5.py`).
   Splitting the start offset by "sentence-initial unit" against "interior clause piece" gains
   +0.0007 score, and only 4 of 195 spans anchor on an interior piece (`u10.py`). No action.
3. **The run-contiguity rule is at its optimum.** `i - run[-1] <= 2` (bridge one skipped unit):
   tightening to strict adjacency costs -0.0036 tIoU, loosening to 3 changes nothing, and the run is
   never truncated in 390 answers because `anchor_ids` already bounds the spread (`u5.py`, `u2.py`).
4. **Clamping.** `span_from_ids` clamps start into `[0, duration]` and end into
   `[start + 0.05, duration]`; `prompts._clamp` is identical. The only residual is a span whose start
   lands exactly on `duration`, where `end = duration + 0.05`; it cannot affect IoU against a gold
   inside the audio.
5. **Few-shot pool: no leakage, not stale.** `bench/llm/pool/large-v3-turbo.json` holds exactly 390
   entries from 39 stems, all of them `conversation_<transcript_id>` from `data/question_train.csv`,
   195 positives all carrying evidence text, and every pool question is a training question. The
   pool's `text` is built from gold-span word midpoints and does not depend on `UNIT_SPLIT`, so
   clause units did not stale it. On a file whose stem matches nothing, `set_conversation` excludes
   nothing and all 390 examples stay available, which is the intended behaviour. `warm_llm` refuses
   to start below 150 positives, and the pool file is tracked by git, so a pod copy without the
   gitignored transcripts serves the same examples.
6. **Scoring is one scorer.** `bench.py`, `replay.py` and `local_evaluator.py` all feed
   `local_evaluator.Statistics.record` and `utils.temporal_iou`, and spans are recorded whatever the
   answer was, so a span next to a "no" is credited, matching `SPAN_ON_NO=1`. `replay.py` reproduces
   the stored score to four decimals for all six clause-mode runs (0.8125 / 0.8179 / 0.8139 /
   0.8144 / 0.8068 / 0.8024) and reproduces the known-wrong stored 0.7849 of the sentence run as
   0.7974 with the right offsets. `replay.py` rebuilds units from `config.unit_split`, and `bench.py`
   writes that field from the environment that built them.
7. **Token budget.** Serving uses `LLM_MAX_TOKENS=600`, the bench used 1500. Across the three
   3.8-27B runs the **maximum** completion is 162 tokens (p99 = 64) and `stopped_at_max_tokens = 0`,
   so 600 is 3.7x the observed worst case. The thinking-off path is doubled up: the request sets
   `chat_template_kwargs {"enable_thinking": false}` (`model._chat_openai` and `bench.Client.body`
   send the identical key) and both the pod and the cluster start vLLM with
   `--default-chat-template-kwargs '{"enable_thinking": false}' --reasoning-parser qwen3`, which puts
   any stray reasoning in `reasoning_content` and leaves `content` as the JSON both readers parse.
8. **Serving prompt equals benchmarked prompt.** `model._chat_openai` builds the same body as
   `bench.Client.body` minus `logprobs`/`top_logprobs`; `units-fewshot` carries `demos=None`, so the
   message list is identical (system plus one user turn). `LLM_VARIANT=units-fewshot` is the variant
   whose stored runs score 0.8125 / 0.8179.
9. **Response shape.** `SPAN_ON_NO=1` (the default, not overridden in `serve.env`), so a "no" returns
   real floats rather than `None`, which is what the portal credits (entry 38). `validate_response`
   cannot raise on anything `predict` constructs: answers are `bool(...)`, lengths are forced to
   `len(request.questions)` by the catch-all, span values are `round(x, 2)` floats, `end >= start +
   0.05`, and start and end are always both set or both `None`. 60 of 390 "no" answers legitimately
   return `None`/`None` (empty quote and empty citation list).
10. **Environment plumbing on the laptop.** `serve_forever.laptop.sh` unsets every override and
    exports `ASR_MODEL=large-v3-turbo` before sourcing `serve.env`, so `_FITTED` resolves to
    (-0.20, +0.28, -0.02) and START_OFFSET/END_OFFSET land on (-0.20, -0.02). `serve.env` sets
    `LLM_MODEL=Qwen/Qwen3.8-27B` explicitly, so no `GET /v1/models` round trip is needed at start-up.
    `bench.py` refuses an `ASR_MODEL` that contradicts `--asr`, so the entry-38 offset bug cannot
    recur.
11. **`model.py` is Linux-safe.** The CUDA-DLL block is guarded by `_p.is_dir()`, which is false on a
    Linux `site-packages`, so `os.add_dll_directory` is never reached on the pod.
12. **Clause splitter.** Every clause piece carries at least three words (`_clause_split` merges
    right when the head is short and merges left when the tail is short); the 612 of 2529 units below
    three words are plain sentence units ("Yes.", "Okay."). Across all 97 cached transcripts only
    **3** clause-and cuts fall between a measurement and a following "and"/"or", and all three read
    correctly. The number/unit guard in `_clause_is_break` does not apply to the comma-less
    coordinating-conjunction cut, which is the one gap, and it costs nothing measurable.

## Not checked

- The pod itself: no request was sent to vLLM, Ollama, the RunPod proxy or the live endpoint on
  9054, so `GET /v1/models` returning exactly `Qwen/Qwen3.8-27B`, the proxy's body limit and read
  timeout, the actual VRAM headroom, and whether Ollama sits fully on the GPU are all unverified.
- `bench/mine/` (held-out validation) was not opened, so nothing here says anything about validation
  behaviour; entry 48's caveat that clause golds may be rarer there stands unexamined.
- Whether faster-whisper's `WhisperModel` is safe to call from several threads at once. Finding 3
  assumes it is not, which is the conservative reading.
- Behaviour when the evaluation audio is materially longer than the training audio (74-232 s): the
  26 s ASR crossover of finding 5 and the 55 s wall of finding 3 were reasoned about, not exercised.
- The `x1`, joint-demo and `words` variants and the non-27B rows: not re-verified, they are not
  served.
- End-to-end timing of a whole attempt against the service's whole-attempt budget and its
  five-consecutive-timeout abort.
