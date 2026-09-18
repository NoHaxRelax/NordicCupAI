#!/bin/bash
# bench/hpc/pod_endpoint.sh: run the WHOLE endpoint on the pod (ASR + api.py + vLLM + Ollama
# fallback), so the submitted URL is the pod's stable proxy hostname
# (https://<POD_ID>-9054.proxy.runpod.net/predict) and nothing depends on the laptop or a tunnel
# (committee report 02, finding 10). Companion of pod_bootstrap.sh (which runs vLLM).
#
#     bash /workspace/medical-appointment/bench/hpc/pod_endpoint.sh install   # venv-api, ollama, qwen3:4b, turbo weights
#     bash .../pod_endpoint.sh serve                                          # supervisor loop: api.py on :9054, restart on exit
#     bash .../pod_endpoint.sh status | stop
#
# Env for api.py (edit /workspace/serve.env; the supervisor re-reads it on every restart).
set -uo pipefail
CASE=/workspace/medical-appointment
VENV=/opt/venv-api
LOGS=/workspace/logs
export HF_HOME=${HF_HOME:-/workspace/hf}
export OLLAMA_MODELS=${OLLAMA_MODELS:-/workspace/ollama}
mkdir -p "$LOGS" "$OLLAMA_MODELS"

install() {
  if ! [ -x "$VENV/bin/python" ]; then
    python -m venv "$VENV" && "$VENV/bin/pip" install -q -U pip || return 1
  fi
  # faster-whisper's ctranslate2 wheel links against CUDA 12 libraries: take them from pip, not from
  # the CUDA 13 torch in venv-vllm. fastapi/uvicorn/pydantic/requests are api.py's own needs.
  "$VENV/bin/pip" install -q faster-whisper fastapi "uvicorn[standard]" requests pydantic numpy \
      nvidia-cublas-cu12 nvidia-cudnn-cu12 || return 1
  echo "venv-api: $("$VENV/bin/python" -c 'import faster_whisper, ctranslate2; print("faster-whisper", faster_whisper.__version__, "ctranslate2", ctranslate2.__version__)')"
  if ! command -v ollama >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh >> "$LOGS/ollama_install.log" 2>&1 || { echo "ollama install failed"; tail -n 5 "$LOGS/ollama_install.log"; return 1; }
  fi
  ollama_up
  ollama pull qwen3:4b >> "$LOGS/ollama.log" 2>&1 && echo "qwen3:4b pulled"
  # turbo weights into HF_HOME (faster-whisper downloads on first construction)
  "$VENV/bin/python" - <<'PY'
import os, time
from faster_whisper import WhisperModel
t0 = time.time()
m = WhisperModel('large-v3-turbo', device='cpu', compute_type='int8')   # weights only; CUDA is exercised by api.py's warm-up
print(f'turbo weights present ({time.time() - t0:.0f} s)')
PY
}

libpath() {   # CUDA 12 runtime libs for ctranslate2, from the pip packages in venv-api
  local sp; sp=$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')
  echo "$sp/nvidia/cublas/lib:$sp/nvidia/cudnn/lib:$sp/nvidia/cuda_runtime/lib:$sp/nvidia/cuda_nvrtc/lib"
}

ollama_up() {
  curl -sf localhost:11434/api/tags >/dev/null 2>&1 && return 0
  OLLAMA_NUM_PARALLEL=4 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_CONTEXT_LENGTH=6144 OLLAMA_FLASH_ATTENTION=1 \
    nohup ollama serve >> "$LOGS/ollama.log" 2>&1 &
  for _ in $(seq 1 30); do curl -sf localhost:11434/api/tags >/dev/null 2>&1 && return 0; sleep 1; done
  echo "ollama did not come up"; return 1
}

serve() {
  ollama_up
  cd "$CASE" || return 1
  export LD_LIBRARY_PATH="$(libpath)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  while true; do
    export ASR_MODEL=large-v3-turbo TRANSCRIPT_CACHE=0 LLM_BACKEND=vllm LLM_URL=http://localhost:8000/v1 \
           LLM_MODEL=Qwen/Qwen3.8-27B LLM_VARIANT=units-fewshot LLM_NO_THINK=vllm UNIT_SPLIT=clause-and \
           LLM_FALLBACK_URL=http://localhost:11434 LLM_FALLBACK_MODEL=qwen3:4b LLM_DEADLINE=40 PREDICT_DEADLINE=55 \
           REQUEST_DUMP_DIR=/workspace/request_dump
    [ -f /workspace/serve.env ] && set -a && . /workspace/serve.env && set +a
    echo "$(date -Is) supervisor: starting api.py (LLM_URL=$LLM_URL UNIT_SPLIT=$UNIT_SPLIT variant=$LLM_VARIANT fallback=$LLM_FALLBACK_MODEL)" >> "$LOGS/api.supervisor.log"
    "$VENV/bin/python" api.py >> "$LOGS/api.log" 2>&1
    echo "$(date -Is) supervisor: api.py exited with $?; restarting in 3 s" >> "$LOGS/api.supervisor.log"
    sleep 3
  done
}

status() {
  echo "vllm:   $(curl -sf localhost:8000/health >/dev/null && echo healthy || echo DOWN)"
  echo "ollama: $(curl -sf localhost:11434/api/tags >/dev/null && echo up || echo DOWN)"
  echo "api:    $(curl -sf localhost:9054/ >/dev/null && echo up || echo DOWN)"
  curl -sf localhost:9054/api 2>/dev/null | head -c 1500; echo
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
}

stop() { pkill -f "pod_endpoint.sh serve" 2>/dev/null; pkill -f "python api.py" 2>/dev/null; echo stopped; }

case "${1:-status}" in
  install) install ;;
  serve) serve ;;
  status) status ;;
  stop) stop ;;
  *) echo "usage: $0 install|serve|status|stop"; exit 2 ;;
esac
