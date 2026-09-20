# Medical Appointment: the evaluation endpoint

Team **Powered by Smørrebrød**, Technical University of Denmark. Nordic AI Cup 2026, use case
`medical-appointment`.

This branch contains one thing: the `/predict` endpoint that served our single evaluation attempt on
**20 September 2026** (score **0.7814**, no errors), and what is needed to stand it up again. The
served files here are byte-identical to the ones that ran; the upload to the pod was checksum-verified
against the commit.

## What to use for evaluation

One configuration. Nothing else on this branch is an alternative.

| setting | value |
| --- | --- |
| speech recognition | `faster-whisper` 1.2.1, model `large-v3-turbo`, English, word timestamps, beam 5 |
| answering model | `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, bf16, vLLM 0.29.0 |
| decoding | temperature 0, thinking disabled, output constrained to a JSON schema |
| prompt | `LLM_VARIANT=units-fewshot-both` in `bench/llm/prompts.py` |
| transcript units | `UNIT_SPLIT=clause-and` |
| span offsets | start -0.20 s, end -0.02 s (`model._FITTED`) |
| spans on "no" answers | returned when the model cites a passage (`SPAN_ON_NO=1`, the default) |
| fallback | Ollama `qwen3:4b`, same prompt, only if the 27B fails or the time budget is nearly spent |
| time limits | 40 s for the language model per conversation, 50 s for the whole reply |
| hardware | one NVIDIA A100 SXM 80 GB, host driver 580 or newer (CUDA 13.0) |

No hosted speech or language API is in the request path. Everything runs on the one machine.

**The environment variables are the configuration.** Started without them, `model.py` falls back to a
development default (a small local Ollama model and sentence units), which is not what we served. Use
path A or the exact variables in path B below, and confirm with `GET /api` before trusting a score.

## Set it up

### A. One command on a bare RunPod pod (this is exactly what we did)

Pod: A100 SXM 80 GB, minimum host CUDA version 13.0, image
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, container disk 60 GB, `/workspace` volume 120 GB,
ports `8000/http`, `9054/http`, `22/tcp`, SSH enabled.

From a laptop with this branch checked out:

    bash bench/hpc/pod_upload.sh HOST PORT          # copies the code, verifies checksums against the commit

On the pod:

    nohup bash /workspace/medical-appointment/bench/hpc/pod_bringup.sh > /workspace/logs/bringup_outer.log 2>&1 &
    tail -f /workspace/logs/bringup_outer.log       # ends with CONFIG_OK, BRINGUP_DONE and the submit URL

`pod_bringup.sh` downloads the pinned model revision, builds two Python environments (one for vLLM,
one for the endpoint), starts vLLM, Ollama and `api.py`, and refuses to finish unless `GET /api`
reports the configuration in the table above. It took 11 min 52 s on evaluation day. The endpoint is
then at `https://<POD_ID>-9054.proxy.runpod.net/predict`.

### B. By hand on any machine with an 80 GB GPU

    # 1. the language model, in its own environment
    python -m venv venv-vllm && venv-vllm/bin/pip install "vllm==0.29.0"
    venv-vllm/bin/vllm serve Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
        --port 8000 --max-model-len 16384 --max-num-seqs 64 --gpu-memory-utilization 0.80 \
        --enable-prefix-caching --reasoning-parser qwen3 --language-model-only \
        --default-chat-template-kwargs '{"enable_thinking": false}'

    # 2. the fallback
    ollama serve & ollama pull qwen3:4b

    # 3. the endpoint
    python -m venv venv-api
    venv-api/bin/pip install "faster-whisper==1.2.1" "ctranslate2==4.8.2" fastapi "uvicorn[standard]" requests pydantic numpy
    ASR_MODEL=large-v3-turbo TRANSCRIPT_CACHE=0 LLM_BACKEND=vllm LLM_URL=http://localhost:8000/v1 \
    LLM_MODEL=Qwen/Qwen3.8-27B LLM_VARIANT=units-fewshot-both LLM_NO_THINK=vllm UNIT_SPLIT=clause-and \
    LLM_FALLBACK_URL=http://localhost:11434 LLM_FALLBACK_MODEL=qwen3:4b LLM_DEADLINE=40 PREDICT_DEADLINE=50 \
        venv-api/bin/python api.py                  # serves on port 9054

`bench/hpc/pod_bootstrap.sh` and `bench/hpc/pod_endpoint.sh` are the scripted form of steps 1 to 3.

## Check it

    curl -s http://localhost:9054/api

must report `llm_variant: units-fewshot-both`, `unit_split: clause-and`, `llm_model: Qwen/Qwen3.8-27B`,
`asr_model: large-v3-turbo`, `breaker_open: false` and a few-shot pool of 390 examples, 195 positive.

With the organisers' training audio copied to `data/audio/`:

    python local_evaluator.py --url http://localhost:9054/predict

Expected on the 39 training conversations: about **0.824** (accuracy 0.997, mean temporal IoU 0.71),
0 failed conversations, 0 timeouts, about 9 s per conversation.

## How it works

1. **Transcribe** the audio with word timestamps (`model.transcribe`).
2. **Cut the word stream into clause-sized units** (`model.make_units`, mode `clause-and`): sentence
   boundaries, pauses of 0.6 s or more, then inside a sentence after a comma or semicolon and before a
   coordinating conjunction, keeping every piece at least three words long.
3. **Ask the 27B each question** (`bench/llm/prompts.py`, variant `units-fewshot-both`). The prompt shows
   the numbered units, plus the twelve most similar positive training questions with the words of their
   annotated spans and the two most similar negatives. The model returns a verbatim quote, yes or no,
   and the ids of the units the answer rests on. Ten questions run concurrently.
4. **Build the span** from the cited units (`model.anchor_ids`, `model.span_from_ids`): the end of the
   first word of the first unit minus 0.20 s, to the end of the last word minus 0.02 s.
5. **Always reply in time.** A failed or late question goes to the fallback model; if that fails the
   question is answered "yes" with no span; a conversation still running at 50 s is answered in full
   the same way, so a reply arrives inside the 60 s limit. On evaluation day none of this was used.

## Models and data

**No model was trained or fine-tuned.** All three models are used as published.

Two small artefacts are derived from the supplied training data, and only from it:

- `bench/llm/pool/large-v3-turbo.json`, the few-shot example pool: for each of the 390 training
  questions its text, its label and the transcript words inside the annotated span. To rebuild it, put
  the training audio in `data/audio/`, then

      python transcribe_cache.py --model large-v3-turbo
      python bench/llm/prompts.py --export-pool large-v3-turbo

  Examples are never taken from the conversation being answered. The pool holds the 39 training
  conversations and nothing from the validation or evaluation sets.
- The two span offsets, fitted on the training annotations by `bench/asr/fit_edges.py`.

`data/question_train.csv` is the organisers' file, included so that both steps can be rerun.

## Evaluation day, 20 September 2026 (times in CEST)

| time | event |
| --- | --- |
| 12:07 | RunPod pod `73blsqshzga9fm` created: A100 SXM 80 GB, data centre EUR-IS-1, driver 580.159.04 |
| 12:09 | code uploaded, nine serving files checksum-verified against the commit |
| 12:21 | `CONFIG_OK units-fewshot-both / clause-and, weights 1d4bf0f2..., vllm 0.29.0`, `BRINGUP_DONE` |
| 12:22 to 12:28 | training set through the public URL: 0.824, 0 failed, 0 timeouts, slowest conversation 15.4 s |
| 12:28 to 12:41 | five validation runs: 0.8117, 0.8057, 0.8081, 0.8116, 0.8117 |
| 12:51:53 | the evaluation attempt queued against `https://73blsqshzga9fm-9054.proxy.runpod.net/predict` |
| 12:57:07 | finished: **0.7814**, error list empty |

During the evaluation the endpoint answered 38 conversations, every one with HTTP 200, mean 7.4 s and
slowest 11.7 s per conversation, with no fallback, no guess and no timeout. The yes/no answers were
identical across the five validation runs; seven of 190 spans moved by one neighbouring unit between
runs (batch-order numerics in vLLM), which is a spread of about 0.003 in score.

The logs are in `evidence/`: `bringup_outer.log` (the build and the configuration check), `api.log`
(every conversation with its timing), `api.supervisor.log` (the endpoint was started once and never
restarted), `validation_runs.json` and `evaluation_attempt.json` (the portal's replies). The pod was
terminated after the attempt; its full request dump is kept and available on request.

## Disclosure

On 17 September we recovered the labels of the validation set from the scores the portal returned for
our own validation runs, which produced a validation entry of 1.0. We disclosed this to the organisers
on the morning of 18 September. Those labels are not in the request path, are not in the few-shot pool
and no parameter was fitted on them; we used them only to measure this pipeline offline. Our honest
validation score is the 0.81 above.

## Files

| path | what it is |
| --- | --- |
| `api.py`, `example.py`, `dtos.py`, `utils.py` | the organisers' endpoint template, with `example.predict` calling our pipeline |
| `model.py` | the pipeline: transcription, units, the language-model calls, fallback, spans, deadlines |
| `bench/llm/prompts.py` | the prompt text and the few-shot selection; the served variant is `units-fewshot-both` |
| `bench/llm/pool/large-v3-turbo.json` | the few-shot example pool |
| `bench/hpc/pod_bringup.sh`, `pod_bootstrap.sh`, `pod_endpoint.sh`, `pod_upload.sh` | the scripts that built and served the pod |
| `bench/asr/fit_edges.py`, `span_ceiling.py`, `transcribe_cache.py` | how the offsets and the pool's transcripts are derived |
| `local_evaluator.py`, `requirements.txt`, `data/question_train.csv` | the organisers' scorer, its requirements and the training questions |
| `evidence/` | logs and portal replies from evaluation day |

`bench/llm/prompts.py` also defines the other prompt variants we benchmarked during development. They
are not used by the endpoint; the variant is selected by `LLM_VARIANT` and checked at start-up.

The complete development history, with every experiment and measurement behind these choices, is on
the branch `medical-appointment` of the same repository (the served commit is `c53c621`).
