#!/usr/bin/env bash
# bench/llm/serve_vllm.sh: start vLLM for one model on one H100, wait until
# /health answers, print the pid. Companion of bench/llm/bench.py.
#
# Usage
#   bash bench/llm/serve_vllm.sh Qwen/Qwen3.6-27B                            # BF16, model defaults below
#   EXTRA="--quantization fp8" bash bench/llm/serve_vllm.sh Qwen/Qwen3.6-27B  # online FP8 (defaults kept)
#   MODEL=Qwen/Qwen3.5-9B PORT=8001 bash bench/llm/serve_vllm.sh
#   bash bench/llm/serve_vllm.sh stop [Qwen/Qwen3.6-27B]                      # kill the pid file's process
#
# Environment (all optional except the model)
#   MODEL          HF id or local path (or first positional argument)
#   PORT           8000
#   MAX_LEN        8192      --max-model-len (prompt+output; a consultation is ~1.5k tokens)
#   GPU_UTIL       0.92      --gpu-memory-utilization
#   DEFAULT_EXTRA  model-aware flags picked below from $MODEL. For Qwen3.5 / 3.6 / 3.8 / Qwen3-VL
#                  (all native vision-language models that think by default, HF cards 2026-09-17):
#                    --reasoning-parser qwen3 --language-model-only
#                    --default-chat-template-kwargs '{"enable_thinking": false}'
#                  (the "Text-Only" serve line of https://huggingface.co/Qwen/Qwen3.6-27B, plus
#                  thinking off server-side). DEFAULT_EXTRA='' drops them; any other value
#                  replaces them.
#   EXTRA          ''        appended after DEFAULT_EXTRA. Parsed by the shell like a command
#                  line (eval into an array), so quotes inside it work exactly as on a
#                  command line. One form, use this one:
#                    EXTRA="--quantization fp8"
#                    EXTRA="--served-model-name qwen --max-num-seqs 32"
#                    EXTRA="--default-chat-template-kwargs '{\"enable_thinking\": true}'"
#   HF_HOME        /dtu/blackhole/1e/205502/hf     weights   (bench/README.md: nothing under $HOME)
#   XDG_CACHE_HOME /dtu/blackhole/1e/205502/cache  vLLM, torch.compile/inductor, triton, flashinfer
#                  caches (hundreds of MB, rebuilt on the first start of every model+config;
#                  a full $HOME quota kills vLLM during compile with an obscure OSError)
#   WAIT           1800      seconds to wait for /health (first start downloads weights)
#   LOG            bench/results/llm/vllm.<model short>.log
#   VLLM           vllm      the binary (set to an absolute venv path from an LSF job)
#
# Install (Linux venv on blackhole; bench/hpc/env.sh builds it)
#   bash bench/hpc/env.sh venvs         # /dtu/blackhole/1e/205502/venvs/venv-vllm: vllm 0.29.0, requests, pydantic
#   VLLM=/dtu/blackhole/1e/205502/venvs/venv-vllm/bin/vllm bash bench/llm/serve_vllm.sh Qwen/Qwen3.6-27B
#   (bench/hpc/llm_bench.lsf does exactly that.) Qwen3.6 needs vllm>=0.19.0 (HF card,
#   2026-09-17); the script warns below if the binary is older.
#
# Flags verified against https://docs.vllm.ai/en/latest/cli/serve/ (2026-09-17):
#   --port, --max-model-len, --enable-prefix-caching, --gpu-memory-utilization,
#   --quantization (value "fp8" = online dynamic FP8 of a BF16 checkpoint:
#   https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/),
#   --language-model-only ("disables all multimodal inputs by setting all modality
#   limits to 0": Qwen3.5/3.6/3.8 are vision-language models; without it vLLM loads the
#   vision tower and profiles multimodal dummy inputs, shrinking the KV cache),
#   --reasoning-parser and --default-chat-template-kwargs ("a valid JSON string or
#   JSON keys passed individually", i.e. --default-chat-template-kwargs.enable_thinking=false
#   also parses): https://docs.vllm.ai/en/latest/features/reasoning_outputs/
#   Cache roots VLLM_CACHE_ROOT / VLLM_CONFIG_ROOT / XDG_CACHE_HOME:
#   https://docs.vllm.ai/en/latest/configuration/env_vars/
#   GET /health ("Health check") and GET /v1/models:
#   https://docs.vllm.ai/en/latest/serving/online_serving/
#
# Caveats
#   - Prefix caching is on by default in vLLM V1; the flag is passed anyway so the
#     command documents the intent.
#   - Thinking models (Qwen3.x, Gemma 4, Nemotron): Qwen3.5/3.6/3.8 think by default. For
#     them DEFAULT_EXTRA sets enable_thinking=false server-side AND bench.py sends
#     chat_template_kwargs {"enable_thinking": false} per request (--no-think vllm, the
#     auto default), so either side alone is enough. --reasoning-parser qwen3 keeps a
#     stray <think> block out of the JSON content (structured outputs are disabled when
#     reasoning content is not parsed). To bench thinking ON: DEFAULT_EXTRA='--reasoning-parser
#     qwen3 --language-model-only', bench.py --no-think none --max-tokens 4000.
#   - gpt-oss models cannot switch reasoning off (research/04); expect longer outputs.
#   - The script backgrounds the server with nohup; an LSF job must keep running
#     (bench.py) or the job ends and the server with it. Use "stop" at the end.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # bench/llm
CASE="$(cd "$HERE/../.." && pwd)"                            # medical-appointment
RESULTS="$CASE/bench/results/llm"
mkdir -p "$RESULTS"

short() { echo "${1##*/}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9.+-]+/-/g; s/^-+|-+$//g'; }

if [[ "${1:-}" == "stop" ]]; then
    MODEL="${2:-${MODEL:-}}"
    if [[ -n "$MODEL" ]]; then PIDFILE="$RESULTS/vllm.$(short "$MODEL").pid"; else PIDFILE="$(ls -t "$RESULTS"/vllm.*.pid 2>/dev/null | head -1)"; fi
    if [[ -z "${PIDFILE:-}" || ! -f "$PIDFILE" ]]; then echo "no pid file to stop" >&2; exit 1; fi
    PID="$(cat "$PIDFILE")"
    echo "stopping vllm pid $PID ($PIDFILE)"
    kill "$PID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$PID" 2>/dev/null || true
    rm -f "$PIDFILE"
    exit 0
fi

MODEL="${MODEL:-${1:-}}"
if [[ -z "$MODEL" ]]; then echo "usage: [MODEL=...] $0 <model> | stop [model]" >&2; exit 2; fi
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.92}"
EXTRA="${EXTRA:-}"
WAIT="${WAIT:-1800}"
VLLM="${VLLM:-vllm}"

# Everything a first start writes goes under blackhole, never $HOME (bench/README.md).
export HF_HOME="${HF_HOME:-/dtu/blackhole/1e/205502/hf}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/dtu/blackhole/1e/205502/cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$XDG_CACHE_HOME/vllm}"                   # torch.compile artefacts, usage stats
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-$XDG_CACHE_HOME/vllm-config}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/torchinductor}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$XDG_CACHE_HOME}"   # JIT kernels -> $XDG_CACHE_HOME/.cache/flashinfer
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

SHORT="$(short "$MODEL")"
LOG="${LOG:-$RESULTS/vllm.$SHORT.log}"
PIDFILE="$RESULTS/vllm.$SHORT.pid"

if ! command -v "$VLLM" >/dev/null 2>&1; then
    echo "vllm binary not found: $VLLM (set VLLM to the venv's bin/vllm; see bench/hpc/env.sh)" >&2
    exit 1
fi
VLLM_VERSION="$("$VLLM" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+[a-z0-9.+-]*' | head -1 || true)"

# Model-aware defaults. Qwen3.5 / 3.6 / 3.8 and Qwen3-VL are vision-language models
# that think by default (HF cards, 2026-09-17): serve them text-only, with the qwen3
# reasoning parser and thinking off (the "Text-Only" serve line of the Qwen3.6 cards).
case "$MODEL" in
    Qwen/Qwen3.[5-9]*|*/Qwen3.[5-9]*|Qwen/Qwen3-VL*|*/Qwen3-VL*)
        _defaults="--reasoning-parser qwen3 --language-model-only --default-chat-template-kwargs '{\"enable_thinking\": false}'"
        if ! grep -qE '^(0\.(19|[2-9][0-9])|[1-9][0-9]*\.)' <<<"$VLLM_VERSION"; then
            echo "warning: Qwen3.6 needs vllm>=0.19.0 (HF card; assume at least that for $MODEL); $VLLM reports version '${VLLM_VERSION:-unknown}'" >&2
        fi
        ;;
    *)  _defaults='' ;;
esac
DEFAULT_EXTRA="${DEFAULT_EXTRA-$_defaults}"        # '-' not ':-': DEFAULT_EXTRA='' is a valid opt-out

# Advisory only (no flag is changed here): a dense checkpoint in the ~27B BF16 class
# (~56 GB weights, bench/hpc/llm_bench.lsf's own size accounting) has not been run
# under concurrent load on a single H100 before, unlike the two 35B-A3B models that
# already get an FP8 repo for exactly this reason. bench.py defaults to 10 concurrent
# --workers, each wanting up to MAX_LEN tokens of KV cache; that combination can OOM
# mid-run (after /health already answered) rather than at load, where the caller's
# own health-check loop would have caught it. This only warns; pass
# EXTRA="--quantization fp8" (or bench.py --workers N) yourself to mitigate.
case "$MODEL" in
    Qwen/Qwen3.[5-8]-27B|*/Qwen3.[5-8]-27B|*/gemma-4-31B-it|*/GLM-4.7-Flash|*/gpt-oss-120b)
        if [[ "$DEFAULT_EXTRA$EXTRA" != *quantization* ]]; then
            echo "warning: $MODEL looks like a dense ~27B+ BF16 checkpoint with no --quantization" \
                 "set; bench.py's default 10 concurrent workers were never load-tested against" \
                 "this combination (see bench/README.md / bench/hpc/llm_bench.lsf sizing notes)." \
                 "If it OOMs mid-run instead of at startup, consider EXTRA=\"--quantization fp8\"" \
                 "or a lower bench.py --workers." >&2
        fi
        ;;
esac

# DEFAULT_EXTRA and EXTRA are parsed by the shell parser, so quoted arguments with
# spaces (the JSON above) arrive at vllm as one argument. A bare $EXTRA would only
# word-split and pass the quote characters literally.
declare -a ARGS=()
if [[ -n "$DEFAULT_EXTRA" ]]; then eval "ARGS+=($DEFAULT_EXTRA)"; fi
if [[ -n "$EXTRA" ]]; then eval "ARGS+=($EXTRA)"; fi
SHOW=''
if (( ${#ARGS[@]} )); then SHOW="$(printf ' %q' "${ARGS[@]}")"; fi

if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "something already answers /health on port $PORT; stop it first (or set PORT)" >&2
    exit 1
fi

echo "starting: $VLLM serve $MODEL --port $PORT --max-model-len $MAX_LEN --enable-prefix-caching --gpu-memory-utilization $GPU_UTIL$SHOW"
echo "vllm ${VLLM_VERSION:-version unknown}   log: $LOG"
echo "HF_HOME: $HF_HOME   XDG_CACHE_HOME: $XDG_CACHE_HOME   VLLM_CACHE_ROOT: $VLLM_CACHE_ROOT"
echo "TRITON_CACHE_DIR: $TRITON_CACHE_DIR   TORCHINDUCTOR_CACHE_DIR: $TORCHINDUCTOR_CACHE_DIR"
nohup "$VLLM" serve "$MODEL" \
    --port "$PORT" \
    --max-model-len "$MAX_LEN" \
    --enable-prefix-caching --max-num-seqs 64 \
    --gpu-memory-utilization "$GPU_UTIL" \
    ${ARGS[@]+"${ARGS[@]}"} > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"

t0=$(date +%s)
while ! curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "vllm (pid $PID) exited before /health answered; last log lines:" >&2
        tail -n 40 "$LOG" >&2
        rm -f "$PIDFILE"
        exit 1
    fi
    if (( $(date +%s) - t0 > WAIT )); then
        echo "gave up after ${WAIT}s waiting for http://localhost:$PORT/health (pid $PID still running, log $LOG)" >&2
        exit 1
    fi
    sleep 5
done
echo "ready after $(( $(date +%s) - t0 ))s: $(curl -s "http://localhost:$PORT/v1/models" | head -c 300)"
echo "vllm pid $PID (pid file $PIDFILE)"
echo "$PID"
