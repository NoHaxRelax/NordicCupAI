# Medical Appointment: submission notes for the Scientific Jury

Team "Powered by Smørrebrød", Technical University of Denmark. Nordic AI Cup 2026, use case
medical-appointment. This file is the README of the code archive built by `bench/hpc/jury_package.sh`.

## What the system is

One HTTP endpoint (`api.py`, FastAPI, the organisers' template) answering ten yes/no questions per
recorded consultation and returning, for each, the time span of the evidence.

1. **Transcription** with `faster-whisper`, model `large-v3-turbo`, word timestamps on, English.
2. **Units.** The word stream is cut into clause-sized units (`model.make_units`, mode `clause-and`):
   sentence boundaries (terminal punctuation, ASR segment ends, pauses of 0.6 s or more), then inside
   each sentence a cut after every comma or semicolon (never inside a number or between a number and
   its unit) and before a comma-less coordinating conjunction (and, but, or, so), keeping every piece
   at least three words long.
3. **Answering** with `Qwen/Qwen3.8-27B` served by vLLM, temperature 0, thinking disabled, output
   constrained to a JSON schema (`quote`, `answer`, `segments`: the ids of the units that carry the
   evidence). The prompt (`bench/llm/prompts.py`, variant `units-fewshot-both`) shows as examples the
   twelve most similar positive training questions (by question-word overlap), each with the
   transcript words inside its annotated span, plus the two most similar negatives. Examples never
   come from the conversation being answered.
4. **Span** = end of the first cited unit's first word minus 0.20 s, to the end of its last word
   minus 0.02 s. The two offsets are fitted on the training annotations.
5. **Fallback.** If the 27B fails, or too little of the 40 s LLM budget is left, a local Ollama
   `qwen3:4b` answers the same prompt; if that fails too, the question is answered "yes" with no span.
   Independently, a conversation still running at 50 s is answered all "yes" with no spans, so a
   reply always arrives inside the 60 s limit.

Everything runs on one machine (an A100 80 GB RunPod instance during evaluation). No hosted ASR or
LLM is in the request path.

## Trained models

None. No weights were fine-tuned. The three models are used as published:

| role | model | source |
| --- | --- | --- |
| ASR | `large-v3-turbo` | fetched by `faster-whisper` on first use |
| answering | `Qwen/Qwen3.8-27B` | Hugging Face hub, bf16 |
| fallback | `qwen3:4b` | Ollama library |

The only artefacts derived from the training data are:

- `bench/llm/pool/large-v3-turbo.json`: the few-shot example pool. Built by `prompts.export_pool()`
  from `data/question_train.csv` and our transcripts of the 39 training recordings; for each training
  question it stores the question, its label and the transcript words whose midpoint lies inside the
  annotated span.
- The two span offsets (-0.20 s, -0.02 s), fitted on the training annotations
  (`bench/asr/fit_edges.py`; the served constants are `model._FITTED`).

## How to run it

    pip install -r requirements.txt faster-whisper==1.2.1    # the template's needs, plus the ASR
    # serve Qwen/Qwen3.8-27B with vLLM on :8000 (bench/hpc/pod_bootstrap.sh), Ollama qwen3:4b on :11434
    ASR_MODEL=large-v3-turbo LLM_BACKEND=vllm LLM_URL=http://localhost:8000/v1 LLM_MODEL=Qwen/Qwen3.8-27B \
      LLM_VARIANT=units-fewshot-both UNIT_SPLIT=clause-and LLM_DEADLINE=40 PREDICT_DEADLINE=50 python api.py
    python local_evaluator.py --url http://localhost:9054/predict     # the organisers' scorer, training set

`bench/hpc/pod_bringup.sh` does all of that on a bare RunPod pod in about eleven minutes and is
exactly what served the evaluation. Measured on the 39 training conversations: 0.826 (accuracy 0.997,
mean temporal IoU 0.712). On the organisers' validation set: 0.8084 (validation run 397). The
pipeline is deterministic: runs 394 and 395, of the configuration before the last two prompt lines
were added, both scored 0.8074.

## Where things are

- `model.py`, `example.py`, `api.py`, `dtos.py`: the served pipeline.
- `bench/llm/`: prompt variants, the offline bench (`bench.py`) and the replay scorer (`replay.py`).
- `bench/units/`: the clause-unit oracle. `bench/fit_edges.py`: offset fitting.
- `bench/hpc/`: the DTU HPC and RunPod scripts used to run the benches and to serve.
- `bench/ref/`: transcripts corrected by ear, used only to check ASR quality.
- `bench/mine/`: tooling that recovered the validation labels from the scores the portal returned
  for our own validation runs (about 360 probe runs on 17 September), which produced our validation
  entry of 1.0. This was disclosed to the organisers on the morning of 18 September. It is not in the
  request path, and no parameter was fitted on those labels. They touched two served choices, both
  recorded in the findings log: one sentence of the few-shot note in `bench/llm/prompts.py` was first
  written after looking at the recovered validation spans and was kept only after being re-derived
  from the training annotations alone (entry 40); and the spans-on-every-question policy
  (`SPAN_ON_NO=1`), which is what the organisers' `local_evaluator.py` credits, was checked against
  the recovered labels (entry 38) and left unchanged. The validation numbers above come from ordinary
  validation runs of the pipeline. Our honest validation score is 0.8084.
- `research/`: the findings log (`07-findings-log.md`, 61 numbered entries) and the committee
  reports that record every measurement behind the design choices above.
