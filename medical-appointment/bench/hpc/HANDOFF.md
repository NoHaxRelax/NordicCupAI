# Handoff: get the LLM answering bench running on the DTU cluster

Written 2026-09-17 evening for a second Claude Code session (or a human). Read
`bench/README.md`, `bench/hpc/README.md`, `research/07-findings-log.md` entries 20-21
and 27 first. The memory file for this project (`~/.claude/projects/.../memory/`) has the
cluster paths and rules (never touch the competition's evaluation endpoint; this task never
touches the portal at all).

## Why this matters

The served pipeline answers with qwen3:4b. On the validation set it got 181 of 190
binaries (entry 27 of the findings log) and its span tIoU is about 0.49 against an oracle
ceiling of 0.85 (entry 21: a quarter of the spans point at the wrong sentence). The bench in
`bench/llm/bench.py` scores candidate answering models on the 39 training conversations
(390 questions, gold spans) so we can pick a bigger model to serve. Serving budget: 60 s per
conversation, one GPU we bring ourselves (an H100 via a tunnel is acceptable for the
evaluation run, or a quantised model on the RTX 5070 8 GB laptop).

## What happened

Job 29427875 (`bench/hpc/llm_bench.lsf`, H100, queued 15:52, ran 17:55-17:57) produced
nothing:

- `venv-vllm` holds **vLLM 0.11.0**, although `bench/hpc/env.sh` meant to install 0.29.0.
  `bench/llm/serve_vllm.sh` adds `--language-model-only --default-chat-template-kwargs
  '{"enable_thinking": false}'` for Qwen3.5/3.6/3.8, which 0.11 rejects, and Qwen3.6+ need
  vLLM >= 0.19 anyway. So Qwen/Qwen3.8-27B and Qwen/Qwen3.6-27B failed at startup.
- `openai/gpt-oss-20b` failed with `IncompleteSnapshotError`: its download into
  `/dtu/blackhole/1e/205502/hf/hub` was cut short.
- The other eight models in the default list were never prefetched (compute nodes have no
  internet; weights must be fetched on the login node first).
- Logs: `/dtu/blackhole/1e/205502/nordic/logs/llm_bench.29427875.{out,err}` and
  `.../medical-appointment/bench/results/logs/llm_bench.29427875.*.serve.log`.

## Done so far (2026-09-17 18:20)

1. `venv-vllm` now holds vLLM 0.29.0 (torch 2.13 cu130); the old venv is kept as
   `venv-vllm-0.11`. Both Qwen flags are accepted by 0.29 (checked with a dummy serve on the
   login node, which fails later only because the login node has no GPU).
2. `env.sh prefetch` re-fetched `openai/gpt-oss-20b` (complete now) and fetched
   `Qwen/Qwen3.6-35B-A3B-FP8`; log `/dtu/blackhole/1e/205502/nordic/logs/prefetch2.log`.
   Gated models (gemma, mistral) still need `HF_TOKEN`.
3. The 39 turbo transcripts of the training set were copied to
   `.../medical-appointment/transcripts/*.large-v3-turbo.json`.
4. Job 29429494 (18:18) failed instantly: renaming a venv breaks the absolute shebangs of
   its `bin/*` launchers (`bin/vllm` pointed at `venv-vllm2/bin/python3.12`). Fixed with
   sed on the shebang lines of both venvs and on `pyvenv.cfg`. Resubmitted as
   **job 29429655** at 18:48 (H100 queue; the previous wait was 28 minutes):
   `MODELS="Qwen/Qwen3.8-27B Qwen/Qwen3.6-27B openai/gpt-oss-20b Qwen/Qwen3.6-35B-A3B-FP8"
   ASR_TAGS="large-v3-turbo large-v3" bsub < bench/hpc/llm_bench.lsf`.
   Logs: `/dtu/blackhole/1e/205502/nordic/logs/llm_bench.29429655.{out,err}`.
   The ASR sweep 29428666 has been running on the A10 since 18:19.

## To do

0. New since the submission: `bench/llm/prompts.py` has `units-fewshot` and `words-fewshot`
   (findings log 30: no gain for qwen3:4b, untested on bigger models). The cluster copy was
   synced at 19:55; job 29429655 still runs the default `VARIANTS="units units-claim words"`.
   Once its results are in, run one more job on the best model only with
   `VARIANTS="units-fewshot words-fewshot"` (the job skips results that already exist).
1. Watch 29429655 (`bjobs -w`). If a model fails at startup, its serve log is under
   `bench/results/logs/llm_bench.29429655.<short>.serve.log`; fix and resubmit only the
   failed models (`MODELS="..."`), the job skips models whose results already exist.
2. Results land in `bench/results/llm/<model>.<variant>.<asr>.json`. Summarise with
   `python bench/llm/bench.py --summary` (see `bench/README.md`) and add a findings-log entry:
   accuracy, mean tIoU under the nulls-on-no policy, yes-rate, and seconds per conversation.
   The decision we need: which model, if any, beats qwen3:4b by more than the 0.006 noise
   floor on the score, and whether it fits the 60 s budget.

## Hard constraints

- **DTU service window: Friday 2026-09-18 20:00 to Monday 08:00.** Nothing runs on the
  cluster after Friday evening; the competition deadline is Saturday 16:00.
- Do not run anything against `https://cases.nordicaicup.com` from this task. The
  validation and evaluation endpoints are handled elsewhere and the evaluation attempt is
  never triggered by an agent.
- Commit and push to the private repo often (branch `medical-appointment`).
