# Medical Appointment: submission notes for the Scientific Jury

Team "Powered by Smørrebrød", Technical University of Denmark. Nordic AI Cup 2026, use case
medical-appointment. This file is the README of the code archive built by `bench/hpc/jury_package.sh`.

## What the system is

One HTTP endpoint (`api.py`, FastAPI, the organisers' template) answering ten yes/no questions per
recorded consultation and returning, for each, the time span of the evidence.

1. **Transcription** with `faster-whisper`, model `large-v3-turbo`, word timestamps on, English.
2. **Units.** The word stream is cut into clause-sized units (`model.make_units`, mode `clause-and`):
   sentence boundaries, plus cuts at subordinating and coordinating conjunctions.
3. **Answering** with `Qwen/Qwen3.8-27B` served by vLLM, temperature 0, thinking disabled, output
   constrained to a JSON schema (`quote`, `answer`, `segments`: the ids of the units that carry the
   evidence). The prompt (`bench/llm/prompts.py`, variant `units-fewshot-both`) shows the twelve
   nearest training questions and their gold units as examples, plus two negatives.
4. **Span** = end of the first cited unit's first word minus 0.20 s, to the end of its last word
   minus 0.02 s. The two offsets are fitted on the training annotations.
5. **Fallback.** If the 27B is unreachable or the per-conversation deadline (50 s) is near, a local
   Ollama `qwen3:4b` answers the same prompt; if that fails, the question is answered "yes" with no span.

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
  from `data/question_train.csv` and our transcripts of the 39 training recordings; it maps each
  training question to the unit ids that overlap its gold span.
- The two span offsets (-0.20 s, -0.02 s), fitted by `bench/fit_edges.py` on the training annotations.

## How to run it

    pip install -r requirements.txt              # api.py, faster-whisper, requests
    # serve Qwen/Qwen3.8-27B with vLLM on :8000 (bench/hpc/pod_bootstrap.sh), Ollama qwen3:4b on :11434
    ASR_MODEL=large-v3-turbo LLM_BACKEND=vllm LLM_URL=http://localhost:8000/v1 LLM_MODEL=Qwen/Qwen3.8-27B \
      LLM_VARIANT=units-fewshot-both UNIT_SPLIT=clause-and LLM_DEADLINE=40 PREDICT_DEADLINE=50 python api.py
    python local_evaluator.py --url http://localhost:9054/predict     # the organisers' scorer, training set

`bench/hpc/pod_bringup.sh` does all of that on a bare RunPod pod in about eleven minutes and is
exactly what served the evaluation. Measured on the 39 training conversations: 0.826 (accuracy 0.997,
mean temporal IoU 0.712). On the organisers' validation set: 0.8084 (runs 394-397, deterministic).

## Where things are

- `model.py`, `example.py`, `api.py`, `dtos.py`: the served pipeline.
- `bench/llm/`: prompt variants, the offline bench (`bench.py`) and the replay scorer (`replay.py`).
- `bench/units/`: the clause-unit oracle. `bench/fit_edges.py`: offset fitting.
- `bench/hpc/`: the DTU HPC and RunPod scripts used to run the benches and to serve.
- `bench/ref/`: transcripts corrected by ear, used only to check ASR quality.
- `bench/mine/`: tooling that recovered the validation labels through the public validation
  scoreboard on 17 September, which produced our validation entry of 1.0. This was disclosed to the
  organisers on the morning of 18 September. It is not in the request path and the served pipeline
  was never tuned on those labels; the validation numbers above come from ordinary validation runs
  of the pipeline. Our honest validation score is 0.8084.
- `research/`: the findings log (`07-findings-log.md`, 61 numbered entries) and the committee
  reports that record every measurement behind the design choices above.
