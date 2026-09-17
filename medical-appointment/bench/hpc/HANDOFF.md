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

## Already started (check before redoing)

1. A fresh venv with the current vLLM is installing on the login node:
   `/dtu/blackhole/1e/205502/venvs/venv-vllm2`, log
   `/dtu/blackhole/1e/205502/nordic/logs/venv-vllm2.install.log`. When it is done, check
   `venv-vllm2/bin/vllm --version` and that `torch.cuda` imports (the login node has no GPU,
   so just `python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"`).
2. A prefetch of the missing weights may be running: see
   `/dtu/blackhole/1e/205502/nordic/logs/prefetch2.log`. It uses
   `LLM_MODELS="..." PREFETCH_SALM=0 bash bench/hpc/env.sh prefetch`.

## To do

1. When `venv-vllm2` is good, either swap it in (`mv venv-vllm venv-vllm-0.11 && mv
   venv-vllm2 venv-vllm`) or point the job at it (`llm_bench.lsf` lines 118-119 build the
   paths from `$VENVS/venv-vllm`). Swapping keeps every script unchanged.
2. Make sure each model in `MODELS` has a complete snapshot under HF_HOME (`env.sh
   prefetch`; a complete snapshot has no `*.incomplete` blobs). Gated models (gemma,
   mistral) need `HF_TOKEN` in the environment.
3. Resubmit: `cd /dtu/blackhole/1e/205502/nordic/medical-appointment && MODELS="Qwen/Qwen3.8-27B
   Qwen/Qwen3.6-27B openai/gpt-oss-20b Qwen/Qwen3.6-35B-A3B-FP8" ASR_TAGS="large-v3-turbo
   large-v3" bsub < bench/hpc/llm_bench.lsf`. Check the H100 queue first
   (`bqueues gpuh100`, `bjobs -u all -q gpuh100 | grep -c PEND`); the last wait was two hours.
   The A100 queue is far longer. The turbo transcripts are what we serve, so `large-v3-turbo`
   must be in `ASR_TAGS` if those transcripts exist on the cluster (`ls transcripts/*.large-v3-turbo.json`);
   otherwise sync them from the laptop with `bench/hpc/sync.sh`.
4. Results land in `bench/results/llm/<model>.<variant>.<asr>.json`. Summarise with
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
