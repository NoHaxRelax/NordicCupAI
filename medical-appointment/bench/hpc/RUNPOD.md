# RunPod runbook (Oscar's account)

State on 2026-09-17 23:35. Pod `wasteful_silver_kingfisher` (1x H100 SXM 80 GB, $3.49/h, datacenter
EU-NL-1) is **stopped**; the 120 GB network volume `nordic-weights` (about $8/month) keeps:

- `/workspace/hf` — Qwen/Qwen3.8-27B weights (52 GB), `HF_HOME` for vLLM
- `/workspace/medical-appointment` — serving code, bench harness, training questions, transcripts
- `/workspace/venv-vllm` — a vLLM venv that was still installing when the pod stopped; treat as broken
- `/workspace/serve27b.sh`, `/workspace/bench27b.sh`, `/workspace/run_benches.sh`, `/workspace/logs/`

The container disk is erased on every stop, so anything under `/opt` or `/root` is gone.

## Restart procedure (about 8 minutes to a serving vLLM)

1. RunPod UI: Pods → start the pod. Read the new SSH line from its Connect panel (host and port
   change). Add ports **8000** and **9054** as HTTP ports under Edit Pod before the endpoint is
   needed (only 8888 and TCP 22 are exposed now).
2. Install vLLM on the local disk (fast; the volume takes an hour for the same install):
   ```
   python -m venv /opt/venv-vllm && /opt/venv-vllm/bin/pip install -q -U pip && /opt/venv-vllm/bin/pip install -q vllm requests ninja
   ```
3. `bash /workspace/serve27b.sh` starts vLLM 0.29 on port 8000 with the 27B, thinking off, text
   only, `--max-num-seqs 64` (hybrid Mamba models refuse the default 1024), context 98k tokens for
   the many-shot prompt, and the venv's `bin` on PATH (its kernel compiler needs `ninja`).
4. `bash /workspace/run_benches.sh units-fewshot units-joint-demo-fewshot units-joint-demo-all`
   runs the three pending prompt designs against it; results land in
   `/workspace/medical-appointment/bench/results/llm/`, summary lines in `/workspace/logs/benches.log`.
   Copy the JSONs back to the laptop's `bench/results/llm/` and rank them with the snippet in
   findings log entry 34.
5. Stop the pod in the UI when idle. Without an API key nothing can stop it from the inside.

## What we know about pods here

- No API key: Elias's login has no API-key page; only the account owner can create one
  (Settings → API Keys). With a key, `pip install runpod` plus `runpod.stop_pod(id)` works, and
  the Claude Code RunPod plugin (installed) can be signed in from the terminal CLI's `/mcp`.
- The pod template (Runpod PyTorch 2.8) ships Python 3.12, torch 2.8 cu128, no ninja, and a
  system Python that refuses `pip install` without `--break-system-packages`.
- Network volume writes of many small files are slow; big sequential files are fine.
- Oscar's own pods (`nordic-a100-*-overnight`) bill the same $150 balance. Never touch them.
- For a second, bigger pod: Qwen3.6-235B needs 2x H200 (FP8) and the same volume; the
  template `vllm/vllm-openai` would skip the install step entirely.
