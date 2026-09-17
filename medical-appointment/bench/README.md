# bench: the experiment harness for the medical-appointment case

Everything in here exists to answer two questions with numbers instead of opinions:

1. **Which ASR gives the best evidence spans?** Measured on the 39 training
   conversations as (a) the oracle-selection tIoU ceiling of sentence units,
   (b) the signed offset between the annotated span edges and the model's word
   boundaries, and (c) word error rate against a reference transcript.
2. **Which answering model, prompt and span rule score best**, offline, over the
   390 training questions, on cached transcripts, without the HTTP server.

Nothing in `bench/` is imported by the serving path (`api.py`, `model.py`).
`bench/` may import from `model.py` (units, offsets, prompt) so the bench measures
the same code that serves.

## Layout

```
bench/
  README.md                this file: the contract
  asr/
    common.py              audio loading, transcript writer, CLI helper (shared)
    run_faster_whisper.py  faster-whisper (CTranslate2) any Whisper size
    run_hf_whisper.py      HF transformers Whisper, optional speculative decoding
    run_whisperx.py        WhisperX: Whisper + wav2vec2 forced alignment
    run_parakeet.py        NVIDIA Parakeet-TDT (v2, v3) native word timestamps
    run_canary_qwen.py     NVIDIA Canary-Qwen-2.5B, text only, silver reference
    run_qwen3_asr.py       Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B
    run_granite.py         IBM granite-speech-4.1-2b-plus native timestamps
    align_mms.py           re-time any cached transcript with torchaudio MMS_FA
    compare.py             the report: ceiling, offsets, WER, per model tag
    fit_edges.py           LOCO fit of edge-transformation rules per model tag
  llm/
    bench.py               390-question offline bench vs an OpenAI-compatible server
    serve_vllm.sh          start vLLM for one model on one H100
    prompts.py             prompt/output variants (unit ids, first/last words, ...)
  hpc/
    sync.sh                rsync code+data up, results down (alias `dtu`)
    env.sh                 build the venvs on blackhole (one per dependency family)
    asr_bench.lsf          gpuh100 job: run every ASR runner, then compare.py
    llm_bench.lsf          gpuh100 job: for each model, serve vLLM, run bench.py
  ref/                     hand-corrected reference transcripts (see PROTOCOL.md)
  results/                 outputs (gitignored)
```

## Transcript schema (the one contract every runner must honour)

One JSON file per conversation per model tag, written to
`medical-appointment/transcripts/<audio stem>.<tag>.json`, for example
`transcripts/conversation_sample_4.parakeet-tdt-0.6b-v2.json`. This is the
schema `transcribe_cache.py` already writes and `span_ceiling.py`,
`model.py` and `bench/llm/bench.py` read:

```json
{
  "file": "conversation_sample_4.mp3",
  "model": "parakeet-tdt-0.6b-v2",
  "duration": 106.5,
  "seconds": 0.9,
  "segments": [
    {"start": 0.0, "end": 1.14, "text": " Morning, Dr Fabricius.",
     "words": [{"w": " Morning,", "start": 0.0, "end": 0.52, "p": 0.98}, ...]}
  ]
}
```

Rules:

- `duration` is the audio length in seconds; `seconds` is wall-clock time of the
  transcription call only (not model load), measured in the runner.
- `segments[].words[].w` keeps the model's own spelling, including punctuation
  attached to the word and a leading space if the tokenizer produces one.
  `text` is the concatenation of the segment's words.
- `start`/`end` are seconds from the start of the decoded audio. No offsets, no
  padding, no rounding beyond what the model gives. Corrections happen in
  `model.py`, never in a runner.
- A model without segments (word stream only) writes one segment per sentence,
  split at terminal punctuation, using `common.words_to_segments`.
- A model without word timestamps (text-only, reference use) writes segments
  with `words: []` and segment-level times if it has them, else `start`/`end`
  of `0`/`duration`, and sets `"word_timestamps": false` at the top level.
- `p` (word probability) is optional; omit if the model has none.
- Runners never modify `data/`. They read MP3 via `common.load_audio`, which
  decodes to 16 kHz mono float32 once, so every model sees identical samples.

Tag naming: lowercase, the model's HF name without the org, plus a suffix for a
variant: `large-v3`, `large-v3+spec-distil`, `whisperx-large-v3`,
`parakeet-tdt-0.6b-v2`, `qwen3-asr-1.7b+fa`, `granite-4.1-2b-plus`,
`<tag>+mms` for `align_mms.py` output.

## Results schema

`bench/results/asr/<tag>.json`: `{tag, n_files, rtf, wer (or null), ceiling: {segment, sentence, merge4}, offset: {start_median, start_p25, start_p75, end_median, ...}, first_word_fraction: {...}}` written by `compare.py`.

`bench/results/llm/<llm>.<variant>.<asr tag>.json`: per-question predictions plus
the same summary block `local_evaluator.py` prints (accuracy by type, mean tIoU,
missing spans, latency per conversation), written by `bench/llm/bench.py`.

## Ground truth

`data/question_train.csv`: 390 rows, 195 with `evidence_start`/`evidence_end`
(seconds, on a 0.02 s grid). Scoring helpers live in `utils.py`
(`temporal_iou`, `gold_evidence`, `group_questions_by_conversation`); use them,
do not reimplement. Score = 0.4 accuracy + 0.6 mean tIoU over the 195 annotated
yes questions (a missed positive counts 0 in that mean).

## Cluster

DTU HPC, ssh alias `dtu`, run remote commands through a login shell:
`ssh -q dtu 'bash -lc "..."'`. Project root on scratch:
`/dtu/blackhole/1e/205502/nordic` (this repo synced there; the
`medical-appointment/` folder inside it). GPU queue `gpuh100` (H100 80 GB),
group `dcc-h100-users`. `$HOME` is quota-limited: models, caches, venvs and
outputs go under blackhole (`HF_HOME=/dtu/blackhole/1e/205502/hf`).
The main branch is a uv project (pyproject.toml, uv.lock, owned by Lucas);
the bench does not touch it and builds its own venvs under blackhole.
