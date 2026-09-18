#!/bin/bash
# bench/hpc/pod_bringup.sh: bare RunPod pod -> warm, serving endpoint, in one command.
#
# This is the script that built pod 9rf8oeyh70minl on 2026-09-18: bring-up start 14:26:34,
# weights done 14:29:58, both venvs 14:31:59, vLLM ready +275 s, endpoint warm, BRINGUP_DONE
# 14:37:15. Ten minutes forty seconds from an empty /workspace/hf to a scored /predict.
#
# Run ON the pod after bench/hpc/pod_upload.sh has put the code in /workspace/medical-appointment:
#
#     nohup bash /workspace/medical-appointment/bench/hpc/pod_bringup.sh > /workspace/logs/bringup_outer.log 2>&1 &
#     tail -f /workspace/logs/bringup_outer.log        # wait for BRINGUP_DONE and the submit URL
#
# Pod requirements (RunPod MCP create-pod): A100 SXM 80 GB, minCudaVersion 13.0 (the cu130 vLLM
# wheel needs driver >= 580), ports 8000/http 9054/http 22/tcp, startSsh, /workspace persistent.
set -uo pipefail
cd /workspace/medical-appointment || { echo "code not uploaded: run bench/hpc/pod_upload.sh first"; exit 1; }
mkdir -p /workspace/logs /workspace/hf
echo "=== bring-up start $(date -Is)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# Three installs in parallel: the 27B weights (54 GB, ~3.5 min with hf_transfer), the vLLM venv, the api venv.
( python -m venv /opt/venv-hf && /opt/venv-hf/bin/pip install -q -U pip huggingface_hub hf_transfer \
  && HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_OFFLINE=0 \
     /opt/venv-hf/bin/hf download Qwen/Qwen3.8-27B --max-workers 16 >/dev/null \
  && echo "WEIGHTS_DONE $(date -Is)" || echo "WEIGHTS_FAILED $(date -Is)" ) & W=$!
( bash bench/hpc/pod_bootstrap.sh install && echo "VLLM_VENV_DONE $(date -Is)" || echo "VLLM_VENV_FAILED" ) & V=$!
( bash bench/hpc/pod_endpoint.sh install && echo "API_VENV_DONE $(date -Is)" || echo "API_VENV_FAILED" ) & A=$!
wait $W $V $A
echo "=== installs finished $(date -Is); hf: $(du -sh /workspace/hf | cut -f1)"
[ -d /workspace/hf/hub/models--Qwen--Qwen3.8-27B ] || { echo "BRINGUP_FAILED: no 27B weights"; exit 1; }

MAX_LEN=16384 GPU_UTIL=0.80 bash bench/hpc/pod_bootstrap.sh serve || { echo "BRINGUP_FAILED: vLLM"; exit 1; }
setsid nohup bash bench/hpc/pod_endpoint.sh serve > /workspace/logs/endpoint_serve.log 2>&1 < /dev/null &
for _ in $(seq 1 90); do curl -sf localhost:9054/ >/dev/null && break; sleep 5; done
grep -E "ASR warm|few-shot pool|LLM backend|LLM warm|fallback .* warm" /workspace/logs/api.log | tail -n 5
bash bench/hpc/pod_endpoint.sh status

# The config that measured 0.8261 training / 0.8084 validation. A green health check with the wrong
# variant would cost 0.015 silently, so this is checked, not assumed.
api=$(curl -sf localhost:9054/api)
echo "$api" | grep -q '"llm_variant": *"units-fewshot-both"' && echo "$api" | grep -q '"unit_split": *"clause-and"' \
  && echo "CONFIG_OK units-fewshot-both / clause-and" \
  || { echo "BRINGUP_FAILED: wrong config in GET /api:"; echo "$api" | head -c 600; exit 1; }
echo "=== BRINGUP_DONE $(date -Is)"
[ -n "${RUNPOD_POD_ID:-}" ] && echo "submit URL: https://${RUNPOD_POD_ID}-9054.proxy.runpod.net/predict"
