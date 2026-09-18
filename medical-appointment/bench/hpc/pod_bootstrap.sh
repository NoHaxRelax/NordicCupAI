#!/bin/bash
# bench/hpc/pod_bootstrap.sh: bring a freshly (re)started RunPod pod to a serving vLLM.
# Run ON the pod (copy it over with the code refresh, see RUNPOD.md):
#
#     bash /workspace/medical-appointment/bench/hpc/pod_bootstrap.sh          # install + serve + health
#     bash .../pod_bootstrap.sh serve                                         # skip the install
#     bash .../pod_bootstrap.sh stop
#
# Idempotent: an existing /opt/venv-vllm with vllm importable is kept; a vLLM already
# answering /health is kept. The container disk is wiped on every stop, so the venv
# is rebuilt after a restart (a few minutes); the weights and code live on the volume.
#
# Env (all optional): MODEL (Qwen/Qwen3.8-27B), PORT (8000), MAX_LEN (73728: room for the
# 60k-token many-shot prompt), GPU_UTIL (0.92), MAX_SEQS (64: hybrid Mamba models refuse the
# default 1024), HF_HOME (/workspace/hf), LOG (/workspace/logs/vllm.log), WAIT (900 s).
set -uo pipefail
MODEL=${MODEL:-Qwen/Qwen3.8-27B}
PORT=${PORT:-8000}
MAX_LEN=${MAX_LEN:-73728}
GPU_UTIL=${GPU_UTIL:-0.92}
MAX_SEQS=${MAX_SEQS:-64}
export HF_HOME=${HF_HOME:-/workspace/hf}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
LOG=${LOG:-/workspace/logs/vllm.log}
WAIT=${WAIT:-900}
VENV=/opt/venv-vllm
mkdir -p "$(dirname "$LOG")"

healthy() { curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; }

install() {
  if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import vllm" 2>/dev/null; then
    echo "venv ok: $("$VENV/bin/python" -c 'import vllm; print(vllm.__version__)')"; return 0
  fi
  echo "installing vLLM into $VENV (local disk) ..."
  python -m venv "$VENV" && "$VENV/bin/pip" install -q -U pip && "$VENV/bin/pip" install -q vllm requests pydantic ninja \
    || { echo "install failed"; return 1; }
  echo "installed $("$VENV/bin/python" -c 'import vllm; print(vllm.__version__)')"
}

serve() {
  if healthy; then echo "vLLM already healthy on :$PORT"; return 0; fi
  [ -d "$HF_HOME/hub/models--${MODEL//\//--}" ] || { echo "weights for $MODEL not under $HF_HOME/hub"; return 1; }
  export PATH="$VENV/bin:$PATH"          # the kernel compiler needs ninja on PATH
  export VLLM_LOGGING_LEVEL=INFO
  echo "starting vLLM $MODEL on :$PORT (max_len $MAX_LEN, seqs $MAX_SEQS, util $GPU_UTIL) -> $LOG"
  nohup "$VENV/bin/vllm" serve "$MODEL" --port "$PORT" --host 0.0.0.0 \
      --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" --gpu-memory-utilization "$GPU_UTIL" \
      --enable-prefix-caching --reasoning-parser qwen3 --language-model-only \
      --default-chat-template-kwargs '{"enable_thinking": false}' \
      >> "$LOG" 2>&1 &
  echo $! > /workspace/logs/vllm.pid
  local t0=$SECONDS
  while [ $((SECONDS - t0)) -lt "$WAIT" ]; do
    if healthy; then
      echo "ready after $((SECONDS - t0)) s: $(curl -s http://localhost:$PORT/v1/models | head -c 200)"; return 0
    fi
    if ! kill -0 "$(cat /workspace/logs/vllm.pid)" 2>/dev/null; then echo "vLLM died; tail of $LOG:"; tail -n 30 "$LOG"; return 1; fi
    sleep 5
  done
  echo "not healthy after $WAIT s; tail of $LOG:"; tail -n 30 "$LOG"; return 1
}

stop() {
  [ -f /workspace/logs/vllm.pid ] && kill "$(cat /workspace/logs/vllm.pid)" 2>/dev/null
  pkill -f "vllm serve" 2>/dev/null; sleep 3; pkill -f "VLLM" 2>/dev/null
  echo "stopped (gpu mem: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null))"
}

case "${1:-all}" in
  all) install && serve ;;
  install) install ;;
  serve) serve ;;
  stop) stop ;;
  *) echo "usage: $0 [all|install|serve|stop]"; exit 2 ;;
esac
