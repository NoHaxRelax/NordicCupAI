# bench/hpc: the DTU HPC side

Four files move the bench onto the cluster and run it overnight on one H100:

| file | runs on | does |
|---|---|---|
| `sync.sh up` / `down` / `ls` | laptop | code + data + transcripts up to blackhole; `bench/results/` + `transcripts/` back down |
| `env.sh venvs` / `prefetch` / `check` / `all` | login node | five venvs under `/dtu/blackhole/1e/205502/venvs/`; every model weight into `HF_HOME`, the torchaudio aligners, nltk `punkt_tab` and a SALM warm-up into the other caches |
| `asr_bench.lsf` | gpuh100 job | every ASR runner in order, then `compare.py --ref canary-qwen-2.5b` |
| `llm_bench.lsf` | gpuh100 job | per answering model: vLLM up, `bench.py` for each ASR tag x variant, vLLM down |

Paths on the cluster: repo root `/dtu/blackhole/1e/205502/nordic`, case folder
`.../nordic/medical-appointment`, job stdout `.../nordic/logs/`, weights
`HF_HOME=/dtu/blackhole/1e/205502/hf`, the other caches next to it (`TORCH_HOME=.../torch-home`,
`NLTK_DATA=.../nltk_data`, `NEMO_CACHE_DIR=.../nemo-cache`, `TMPDIR=.../tmp`), venvs `/dtu/blackhole/1e/205502/venvs/`.

## Cluster facts (probed 2026-09-17 from `gbarlogin1` through `ssh dtu 'bash -lc ...'`)

| what | found |
|---|---|
| login node | `gbarlogin1`, AlmaLinux 9.8, gcc 11.5, git 2.52, rsync, `/usr/bin/ffmpeg`; modules `ffmpeg/4.4.4 ... 8.1.1` |
| python | system `/usr/bin/python3` = 3.9.25. Modules `python3/3.9.19 ... 3.14.2`, default `3.11.9`. **`module load python3/3.11.9` exports `PYTHONHOME` and `PYTHONPATH`** (module show), which break venvs, so `env.sh` uses the interpreters by path: `/appl9/python/3.11.13/bin/python3.11` and `/appl9/python/3.12.11/bin/python3.12` (both run under `env -i`; the prep `venv-nordic` was built on 3.11.13 the same way and ran inside jobs) |
| uv | `~/.local/bin/uv` 0.10.12; managed CPythons 3.11.15 and 3.13.12 in `~/.local/share/uv`; `~/.cache/uv` is already 19 GB in `$HOME`, so `env.sh` sets `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` on blackhole |
| cuda modules | `cuda/10.0 ... 13.3.1`, default `11.8`. Not loaded by the scripts: the pip wheels carry their own CUDA runtime, and a system CUDA on `LD_LIBRARY_PATH` can shadow the pip cuDNN 9 CTranslate2 needs |
| GPU driver | H100 hosts (`lshosts -gpu`): **610.57.04** on `NVIDIAH100PCIe` (V100 hosts: 580.178.04). Any cu12x or cu13x wheel loads |
| gpuh100 hosts | 24 H100 80 GB PCIe listed by `bhosts -gpu` (2 per host), hosts `EPYC9354`, 64 cores, 755 GB RAM, resources `sm90 gpu80gb`; queue `Open:Active`, 41 RUN / 1 PEND at probe time; `RES_REQ: same[type:model] cu[type=enclosure:maxcus=1] affinity[core]` |
| gpuh100 walltime | `bqueues -l gpuh100`: RUNLIMIT default 15 min, **maximum 1440 min = 24 h**; USERS `lsfghopperusers/ lsfghoppersdtubasen/` (group-gated; the account is in) |
| memory | DTU's `rusage[mem=X]` is **per core** (hpc.dtu.dk page_id=1416): `-n 8` with `mem=4GB` = 32 GB (ASR job), `mem=8GB` = 64 GB (LLM job) |
| blackhole | `/dtu/blackhole/1e` (`nvme02b:/nvmee/data`): 3.0 T, 757 G used, **2.2 T free** (26 %). User dir holds `nordic/` (248 M: prep survivalsim, `hpc/`, `logs/`, `runs/`), `venv-nordic` (980 M, torch 2.14 cpu), `hf/` (111 G: an old `hub/` of Qwen2.5/Llama checkpoints, reused as `HF_HOME`), `miniconda3` 27 G, `mcwm` 236 G, `codllm` 122 G, `sv-en` 46 G, `shopvoices` 14 G, `tmp/` |
| `$HOME` | `/zhome` is 94 % full cluster-wide (851 T); nothing of ours goes there |
| internet, login node | `curl -sI https://huggingface.co` -> `HTTP/2 200`; PyPI 200 |
| internet, compute nodes | **none.** The user's own `sv-en/stage_models.sh` on blackhole says "Compute nodes have no internet. A job that downloads a model at runtime fails with a network error". Hence `env.sh prefetch` on the login node and `HF_HUB_OFFLINE=1` in both jobs |
| HF token | none on the cluster (`~/.cache/huggingface/token`, `$HF_HOME/token` absent). Not needed: every model id used answered the HF API with `gated: false` |
| laptop | Git Bash has **no rsync** (WSL Ubuntu has one, but a different ssh/key setup); `sync.sh` falls back to tar-over-ssh. ssh aliases `dtu` (login.gbar.dtu.dk) and `dtu-transfer` (transfer.gbar.dtu.dk) exist |

## Tonight, in order

Timings are estimates; each step prints its own.

1. Connect the DTU VPN (Cisco Secure Client, `vpn.dtu.dk`). Off-VPN `ssh dtu` fails.
2. Laptop, Git Bash, from `medical-appointment/`:
   ```bash
   bash bench/hpc/sync.sh up          # ~100 MB: code, data/audio, question_train.csv, transcripts/ (large-v3), bench/
   ```
   Excludes `.git`, `bench/results`, `__pycache__`, `*.pyc`, `models/`, `.venv`. Never deletes on the cluster. Afterwards it strips CRLF from every `bench/**/*.sh` and `*.lsf` on the remote (the repo has `core.autocrlf=true`; a `\r` after `#!/bin/bash` or on a `#BSUB` line kills a script on Linux) and marks the shell scripts executable.
3. Cluster, login node:
   ```bash
   ssh dtu
   cd /dtu/blackhole/1e/205502/nordic/medical-appointment
   nohup bash bench/hpc/env.sh all > /dtu/blackhole/1e/205502/nordic/logs/env-all.log 2>&1 &
   tail -f /dtu/blackhole/1e/205502/nordic/logs/env-all.log        # per venv: logs/env-<venv>.log, prefetch: logs/env-prefetch.log
   ```
   `venvs`: five venvs, 20-60 min (vLLM and NeMo each pull ~10 GB of wheels).
   `prefetch`: ASR weights ~50 GB (10-30 min; includes `Qwen/Qwen3-1.7B` and `nvidia/canary-1b-flash`, which `SALM.from_pretrained('nvidia/canary-qwen-2.5b')` resolves at load), the torchaudio aligners, `run_whisperx.py --prefetch` (nltk `punkt_tab` into `NLTK_DATA`), a CPU `SALM.from_pretrained` warm-up (`PREFETCH_SALM=0` skips it), then the 11 LLM checkpoints ~550 GB (1-3 h at login-node bandwidth). Idempotent both ways: rerun after any interruption. `PREFETCH_LLM=0` skips the LLMs, `LLM_MODELS="..."` narrows them.
   ```bash
   bash bench/hpc/env.sh check        # built venvs, cached models, disk
   ```
4. Submit the ASR job as soon as `check` shows the venvs and the ASR models (the LLM prefetch can keep running):
   ```bash
   bsub < bench/hpc/asr_bench.lsf     # prints "Job <ID> is submitted to queue <gpuh100>"
   ```
   ~1-2 h wall: 15 steps, each with `STEP_TIMEOUT` 3600 s. Smoke first if in doubt: `LIMIT=2 STEPS="fw-large-v3 parakeet-v2 compare" bsub < bench/hpc/asr_bench.lsf`.
5. Submit the LLM job after the ASR job (it needs `transcripts/*.parakeet-tdt-0.6b-v2.json`; `large-v3` transcripts are already there from the laptop). LSF can chain it:
   ```bash
   bsub -w "ended(<asr job id>)" < bench/hpc/llm_bench.lsf
   ```
   or, to start now on the whisper transcripts only: `ASR_TAGS="large-v3" bsub < bench/hpc/llm_bench.lsf`.
   11 models x 2 tags x 3 variants; ~20-30 min per model (the two gpt-oss models take longer: they cannot stop reasoning, so they run with `--max-tokens 1500` and an 80 min timeout); `-W 8:00`. Models are visited in the order of `MODELS`; every finished result file (`"partial": false` in its summary) is skipped, so a resubmission (or a second job with a shorter `MODELS`) continues.
6. Watch, from the cluster:
   ```bash
   bjobs -w                                                            # PEND / RUN, host, elapsed
   tail -f bench/results/logs/asr_bench.<ID>.log                       # live copy of the job's stdout (LSF's -o file is written at the end)
   tail -f bench/results/logs/asr_bench.<ID>.<step>.log                # one runner's own output
   tail -f bench/results/logs/llm_bench.<ID>.log  bench/results/llm/vllm.<model>.log
   bhist -l <ID>                                                       # after it ends: TERM_RUNLIMIT vs exit code
   ```
   or from the laptop: `D=~/.claude/skills/dtu-hpc-shared/driver.sh; bash $D bjobs; bash $D log /dtu/blackhole/1e/205502/nordic/medical-appointment/bench/results/logs/asr_bench.<ID>.log 60`.
7. Pull results (laptop):
   ```bash
   bash bench/hpc/sync.sh ls          # transcripts per tag, result files on the cluster
   bash bench/hpc/sync.sh down        # bench/results/ (asr/*.json, summary.md, llm/*.json, logs/) and transcripts/
   ```
8. Read locally:
   ```bash
   pip install "jiwer>=4,<5" num2words          # light; only if compare.py is rerun locally
   python bench/asr/compare.py --ref canary-qwen-2.5b        # or open bench/results/asr/summary.md written on the cluster
   ls bench/results/llm/                        # <model>.<variant>.<asr tag>.json, summary block inside each
   python span_ceiling.py --model parakeet-tdt-0.6b-v2      # the original ceiling script on any new tag
   ```

## What the jobs do, exactly

`asr_bench.lsf` (`-n 8`, `rusage[mem=4GB]`, `-W 8:00`, one H100 exclusive): `nvidia-smi`, a table of which weights are cached, then in order, each with a timestamp, its venv, `timeout`, its own log and a summary line:

| step | venv | command |
|---|---|---|
| fw-large-v3, fw-large-v3-turbo, fw-distil-large-v3 | venv-asr | `run_faster_whisper.py --model ...` |
| hf-whisper, hf-whisper-spec | venv-asr | `run_hf_whisper.py`, `run_hf_whisper.py --assistant` |
| whisperx | venv-whisperx | `run_whisperx.py` |
| parakeet-v2, parakeet-v3, canary-qwen | venv-nemo | `run_parakeet.py`, `run_parakeet.py --model v3`, `run_canary_qwen.py` |
| qwen3-asr, qwen3-align-lv3 | venv-qwen | `run_qwen3_asr.py`, `run_qwen3_asr.py --align-only large-v3` |
| granite | venv-asr | `run_granite.py` |
| mms-large-v3, mms-parakeet-v2 | venv-asr | `align_mms.py --tag large-v3`, `--tag parakeet-tdt-0.6b-v2` |
| compare | venv-asr | `compare.py --ref canary-qwen-2.5b` |

`LIMIT=N` adds `--limit N`, `FORCE=1` adds `--force`, `STEPS="a b"` runs a subset. For the faster-whisper steps the job puts the pip cuBLAS/cuDNN directories on `LD_LIBRARY_PATH` (the one-liner from the faster-whisper README), which CTranslate2 needs before Python starts.

`llm_bench.lsf` (`-n 8`, `rusage[mem=8GB]`, `-W 8:00`): for each `MODEL` that is cached and has a missing or unfinished result: `bench/llm/serve_vllm.sh MODEL` with `VLLM=<venv-vllm>/bin/vllm`, `WAIT=1500`, `GPU_UTIL=0.95` (`serve_vllm.sh` itself adds `--reasoning-parser qwen3 --language-model-only` and thinking off for every `Qwen/Qwen3.[5-9]*`, so Qwen3.8-27B is covered); then for each `ASR` in `ASR_TAGS` (default `large-v3 parakeet-tdt-0.6b-v2`) and `V` in `VARIANTS` (default `units units-claim words`): `bench/llm/bench.py --asr ASR --variant V --model MODEL --no-think vllm --url http://localhost:8000/v1 --out bench/results/llm/<short>.<V>.<ASR>.json`, plus `--max-tokens 1500` and a 4800 s timeout (instead of 2400) for the two gpt-oss models (`BENCH_FOR`, `TIMEOUT_FOR`: gpt-oss cannot switch reasoning off and vLLM counts its analysis channel against `max_tokens`, so with the default 200 nearly every answer would stop at the limit with empty JSON); then `serve_vllm.sh stop`, `pkill` as a fallback, and a wait until the GPU memory is back under 2 GB. If the server dies mid-model the remaining runs for that model are skipped. A result counts as done only when its summary says `"partial": false`; bench.py's `<name>.partial.json` checkpoint of a run cut by the timeout or the walltime is not a result, and that run is redone in full.

Default model list, verified against `https://huggingface.co/api/models/<id>` (all exist, `gated: false`), in the order they run: `Qwen/Qwen3.8-27B`, `Qwen/Qwen3.6-27B`, `openai/gpt-oss-20b`, `Qwen/Qwen3.6-35B-A3B-FP8`, `google/gemma-4-31B-it`, `zai-org/GLM-4.7-Flash`, `openai/gpt-oss-120b`, `mistralai/Ministral-3-14B-Instruct-2512`, `Qwen/Qwen3.5-27B`, `Qwen/Qwen3.5-35B-A3B-FP8`, `Qwen/Qwen3-30B-A3B-Instruct-2507`. Corrections to the candidate list: `mistralai/Ministral-3-14B-Instruct` does not exist (401; the id carries the `-2512` suffix), `google/gemma-4-31b-it` is spelled `gemma-4-31B-it`; `Qwen/Qwen3.8-27B` (Aug 2026) was added because research/04 ranks it first; the two 35B-A3B MoEs run from their FP8 repos (`Qwen/Qwen3.6-35B-A3B-FP8`, `Qwen/Qwen3.5-35B-A3B-FP8`, both `gated: false`, 36 GB) because the BF16 twins are 72 GB (35.95 B params per the HF API), which leaves no room for a KV cache on 80 GB; `Qwen/Qwen3.6-27B-FP8` also exists if the BF16 27B models turn out slow.

## The venvs (`env.sh`)

| venv | python | torch | why separate |
|---|---|---|---|
| venv-asr | 3.11 | latest, cu129 | faster-whisper 1.2.1, transformers 5.17, accelerate, peft, torchaudio (MMS_FA), torchcodec, num2words, jiwer 4, librosa, soundfile, requests, pydantic, huggingface_hub (`hf` CLI) |
| venv-whisperx | 3.11 | ~=2.8.0, cu128 | whisperx 3.8.6 pins `torch~=2.8.0` and `huggingface-hub<1.0`; transformers 5 needs `huggingface-hub>=1.5`, so the two cannot share a venv (this is the one deviation from the task's venv list); plus `num2words` (`run_whisperx.py` exits without it), soundfile, librosa |
| venv-nemo | 3.12 | 2.12.0+cu126 | `nemo_toolkit[asr,speechlm2,cu12]` 3.0.0; its `cu12` extra pins exactly that torch |
| venv-qwen | 3.11 | latest, cu128 | qwen-asr 0.0.6 pins `transformers==4.57.6`, `accelerate==1.12.0`; transformers backend, the `[vllm]` extra (vllm==0.14.0) is not installed |
| venv-vllm | 3.12 | 2.13.0, cu129 | vllm 0.29.0 (wheels built with CUDA 12.9), plus requests, pydantic for bench.py |

Nothing else is pinned; every pin above is the library's own requirement read from PyPI on 2026-09-17. `env.sh` is idempotent through a `.built-<STAMP>` marker per venv (`FORCE=1` or `STAMP=...` rebuilds), installs with `uv pip install --python <venv> --torch-backend=<cuXXX>` (falling back to `--extra-index-url https://download.pytorch.org/whl/<cuXXX>`), and runs an import check on the login node before it writes the marker. Logs: `/dtu/blackhole/1e/205502/nordic/logs/env-<venv>.log`.

`env.sh prefetch` stages, with `snapshot_download(ignore_patterns=[original/*, *.gguf, *.pth, *.h5, *.msgpack, *.ot, onnx])`: the three Systran/mobiuslabs faster-whisper repos, `openai/whisper-large-v3`, `distil-whisper/distil-large-v3`, `nvidia/parakeet-tdt-0.6b-v2` and `-v3`, `nvidia/canary-qwen-2.5b` plus the two repos its `config.json` names and `SALM.from_pretrained` resolves at load (`Qwen/Qwen3-1.7B` for the LLM config through `AutoConfig`, and `nvidia/canary-1b-flash`, whose whole `.nemo` NeMo downloads for the encoder config; `*.nemo` is kept by the ignore list, and NeMo's `_get_hf_hub_pretrained_model_info` checks `try_to_load_from_cache` in `HF_HOME/hub` before any network call, so the cached `.nemo` is found offline), `Qwen/Qwen3-ASR-1.7B`, `Qwen/Qwen3-ForcedAligner-0.6B`, `ibm-granite/granite-speech-4.1-2b-plus` and `-2b`, the torchaudio `MMS_FA` and `WAV2VEC2_ASR_BASE_960H` checkpoints into `TORCH_HOME`, `run_whisperx.py --prefetch` in venv-whisperx (nltk `punkt_tab` into `NLTK_DATA`: `whisperx.align` calls `nltk.download('punkt_tab')` and raises `RuntimeError` when that returns `False` offline; the runner's `check_caches()` refuses to start unless `HF_HOME`, `TORCH_HOME` and `NLTK_DATA` all point outside `$HOME`, which both `env.sh` and `asr_bench.lsf` export), a CPU `SALM.from_pretrained('nvidia/canary-qwen-2.5b')` warm-up in venv-nemo (the exact load path of the canary-qwen step; `PREFETCH_SALM=0` skips it), then the LLM list. WhisperX's VAD weights ship inside the package (`whisperx/assets/pytorch_model.bin` on main), nothing to stage.

## If something is off

- `bjobs` shows `PEND` for long: the queue is shared (41 running at probe time); `bjobs -l <ID>` gives the pending reason. Nothing to fix.
- Job ended early: `bhist -l <ID>`. `TERM_RUNLIMIT` = raise `-W` (cap 24:00) or resubmit, finished results are kept and skipped next time; an LLM run that was cut off leaves only `bench/results/llm/<name>.partial.json` (bench.py's checkpoint, `"partial": true`), which `llm_bench.lsf` does not count as done, so the resubmitted job reruns the full bench for that model/variant. `TERM_MEMLIMIT` = raise `rusage[mem]`. A traceback = the step log under `bench/results/logs/`.
- A step says `MISSING <model>` in the cached-weights table: `bash bench/hpc/env.sh prefetch` on the login node, resubmit with `STEPS="..."`.
- `Disk quota exceeded` in a log: something wrote to `$HOME`; every cache variable (`HF_HOME`, `TORCH_HOME`, `NLTK_DATA`, `NEMO_CACHE_DIR`, `TMPDIR`, and `XDG_CACHE_HOME` for vLLM) is set in the scripts, check for a stray `~/.cache`. `run_whisperx.py` exits with `model caches under $HOME: ...` when one of its three is unset.
- vLLM never answers `/health`: `bench/results/llm/vllm.<model>.log`. For Ministral try `EXTRA_MINISTRAL="--tokenizer-mode mistral --config-format mistral --load-format mistral"` (unverified, see below). The Qwen3.5/3.6/3.8 vision-language repos need nothing extra: `serve_vllm.sh` already serves every `Qwen/Qwen3.[5-9]*` with `--language-model-only`, the documented text-only switch. `No available memory for the cache blocks` in the log = the model does not fit at `GPU_UTIL` 0.95 with `MAX_LEN` 8192: use an FP8 repo (as the two 35B-A3B entries do) or `EXTRA_FOR["<id>"]="--quantization fp8"`.
- Compute node reaches the internet after all: harmless, offline mode still resolves from the cache. Set `HF_HUB_OFFLINE=0` at submit time to let a job download.

## Unverified points (marked `UNVERIFIED` in the files)

1. (resolved) every runner's flags were checked against its argparse before the job scripts were finalised; `run_hf_whisper.py --assistant` takes the assistant's HF id, so the step passes `distil-whisper/distil-large-v3` (prefetched).
2. uv `--torch-backend=cu129` by name (docs list cu126/cu128/cu130 as examples); `env.sh` retries with `--extra-index-url` if uv rejects it.
3. `nemo_toolkit` 3.0.0 from PyPI shipping `nemo.collections.speechlm2.SALM` for canary-qwen (the model card installs NeMo from git). The venv-nemo import check prints `WARN speechlm2 ...` instead of failing; if it warns, `run_canary_qwen.py` will fail and the card's git install line must be added to `NEMO_PKGS`.
4. `NEMO_CACHE_DIR` as the env var NeMo reads for its extracted-checkpoint cache (`TORCH_HOME` is also set to blackhole, which covers the default `~/.cache/torch/NeMo`).
5. (resolved) the HF API reports `Qwen/Qwen3.8-27B` and `Qwen3.6-27B` as `Qwen3_5ForConditionalGeneration` (`model_type qwen3_5`) and the 35B-A3B MoEs as `Qwen3_5MoeForConditionalGeneration`; both architectures are in vLLM's supported-models table and `--language-model-only` is the documented text-only switch, which `serve_vllm.sh` passes for every `Qwen/Qwen3.[5-9]*`. The job stays failure-tolerant; a load failure costs a few minutes.
6. Ministral-3 repo format (HF safetensors vs mistral `consolidated.safetensors`); `EXTRA_MINISTRAL` is the hook.
7. Whether whisperx 3.8.6's wheel (not just `main`) bundles the VAD checkpoint; if `run_whisperx.py` fails offline with a download URL in the log, run it once on the login node with `--device cpu --limit 1` (it saves into `TORCH_HOME`). `run_whisperx.py --prefetch` (which `env.sh prefetch` runs) stages the Whisper weights, the aligner and nltk `punkt_tab`, not the pyannote VAD.
8. `run_whisperx.py` using the default English aligner `WAV2VEC2_ASR_BASE_960H` (whisperx's default) rather than a HF wav2vec2 id; the torchaudio checkpoint is prefetched, a HF id would not be.

Deliberately left out: no cuda/python module loads (reasons above), no `--delete` in either sync direction, no `-M` per-process memory cap, no job arrays (one GPU each, sequential steps make the logs readable), no pre-quantised checkpoints except the two 35B-A3B FP8 repos (BF16 fits every other listed model on 80 GB).
