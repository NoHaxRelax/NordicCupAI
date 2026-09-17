#!/bin/bash
# bench/hpc/env.sh: build the bench venvs on DTU blackhole and stage every model
# weight. Run ON THE LOGIN NODE (gbarlogin has internet; the compute nodes do
# not, see bench/hpc/README.md). Nothing here needs a GPU.
#
#   ssh dtu
#   cd /dtu/blackhole/1e/205502/nordic/medical-appointment
#   bash bench/hpc/env.sh venvs       # 5 venvs under /dtu/blackhole/1e/205502/venvs  (20-60 min, idempotent)
#   bash bench/hpc/env.sh prefetch    # HF weights into HF_HOME: ASR ~50 GB, LLMs ~550 GB (hours: use nohup),
#                                     # plus torchaudio aligners, nltk punkt_tab (whisperx) and a SALM warm-up (canary-qwen)
#   bash bench/hpc/env.sh check       # what is built and what is cached
#   bash bench/hpc/env.sh all         # venvs, then prefetch
#
#   nohup bash bench/hpc/env.sh all > /dtu/blackhole/1e/205502/nordic/logs/env-all.log 2>&1 &
#   PREFETCH_LLM=0 bash bench/hpc/env.sh prefetch      # ASR weights only
#   PREFETCH_SALM=0 bash bench/hpc/env.sh prefetch     # skip the CPU SALM.from_pretrained warm-up (~5-10 GB RAM on the login node)
#   LLM_MODELS="Qwen/Qwen3.8-27B openai/gpt-oss-20b" bash bench/hpc/env.sh prefetch
#
# Venvs (one per dependency family). The only pins are the libraries' own, read
# from their PyPI metadata on 2026-09-17; everything else resolves to latest:
#   venv-asr      py3.11  torch (latest, cu129)     faster-whisper 1.2.x + nvidia cuDNN 9/cuBLAS wheels, transformers>=5 (5.17),
#                                                  accelerate, peft, torchaudio (MMS_FA), torchcodec, num2words, jiwer 4,
#                                                  librosa, soundfile, requests, pydantic, huggingface_hub (the `hf` CLI)
#   venv-whisperx py3.11  torch~=2.8.0 (cu128)      whisperx 3.8.6: pins torch~=2.8.0, torchaudio~=2.8.0, huggingface-hub<1.0,
#                                                  which transformers 5.x (huggingface-hub>=1.5) cannot share, hence its own venv;
#                                                  plus num2words (run_whisperx.py's normaliser, required), soundfile, librosa
#   venv-nemo     py3.12  torch==2.12.0+cu126       nemo_toolkit[asr,speechlm2,cu12] 3.0.0: its cu12 extra pins that torch
#   venv-qwen     py3.11  torch (latest, cu128)     qwen-asr 0.0.6: pins transformers==4.57.6, accelerate==1.12.0 (transformers
#                                                  backend; the [vllm] extra pins vllm==0.14.0 and is not installed)
#   venv-vllm     py3.12  torch==2.13.0 (cu129)     vllm 0.29.0 (wheels built against CUDA 12.9), requests, pydantic (bench.py)
# H100 nodes run driver 610.57.04 (lshosts -gpu, 2026-09-17): any cu12x/cu13x wheel loads.
#
# Verified against (2026-09-17):
#   uv venv / uv pip / --python:   https://docs.astral.sh/uv/pip/environments/
#   --python <abs path>:            https://docs.astral.sh/uv/concepts/python-versions/
#   --torch-backend (uv pip only; `auto` reads the local driver, which the login node
#                    lacks, so the backend is spelled out):  https://docs.astral.sh/uv/guides/integration/pytorch/
#   UV_CACHE_DIR / UV_PYTHON_INSTALL_DIR: https://docs.astral.sh/uv/reference/environment/
#   torch wheels present: https://download.pytorch.org/whl/cu126/torch/ (2.12.0+cu126 cp312),
#                         https://download.pytorch.org/whl/cu128/torch/ (2.8.0+cu128 cp311),
#                         https://download.pytorch.org/whl/cu129/torch/ (2.13.0+cu129 cp311)
#   faster-whisper 1.2.1 GPU deps (cuBLAS cu12, cuDNN 9): https://pypi.org/project/faster-whisper/
#                    and the LD_LIBRARY_PATH one-liner:   https://github.com/SYSTRAN/faster-whisper#gpu
#   whisperx 3.8.6 requires_dist:  https://pypi.org/pypi/whisperx/json
#   nemo-toolkit 3.0.0 extras/pins: https://pypi.org/pypi/nemo-toolkit/json ; install line on
#                    https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2 ; SALM on https://huggingface.co/nvidia/canary-qwen-2.5b
#   qwen-asr 0.0.6:  https://pypi.org/pypi/qwen-asr/json and https://github.com/QwenLM/Qwen3-ASR (pip install -U qwen-asr)
#   vllm 0.29.0:     https://pypi.org/pypi/vllm/json ; https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html
#   transformers 5.17.0: https://pypi.org/pypi/transformers/json
#   granite-speech-4.1-2b-plus deps: https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus
#   torchcodec 0.13-0.16 need torch>=2.11 and FFmpeg 4-9 at runtime: https://github.com/pytorch/torchcodec
#   hf CLI, token at $HF_HOME/token: https://huggingface.co/docs/huggingface_hub/guides/cli
#   snapshot_download(repo, ignore_patterns=[...]): the user's own working /dtu/blackhole/1e/205502/pull7b.py
#   MMS_FA pipeline: https://docs.pytorch.org/audio/main/generated/torchaudio.pipelines.MMS_FA.html
#   canary-qwen loads two more repos: https://huggingface.co/nvidia/canary-qwen-2.5b/raw/main/config.json
#                    (pretrained_llm "Qwen/Qwen3-1.7B", pretrained_asr "nvidia/canary-1b-flash") resolved by
#                    nemo/collections/speechlm2/parts/pretrained.py (AutoConfig.from_pretrained, then
#                    ASRModel.from_pretrained(..., return_config=True) = hf_hub_download of the .nemo)
#   whisperx.align: nltk.download('punkt_tab') inside align(), RuntimeError when it returns False (offline):
#                    https://raw.githubusercontent.com/m-bain/whisperX/main/whisperx/alignment.py
#   FP8 twins Qwen/Qwen3.6-35B-A3B-FP8 and Qwen/Qwen3.5-35B-A3B-FP8: gated=false, 36 GB (HF API, 2026-09-17);
#                    the BF16 repos are 72 GB (35.95 B params) and leave no KV cache on 80 GB
#
# Caveats
#   - Idempotent: a venv with a .built-<STAMP> marker is skipped; FORCE=1 rebuilds. A venv
#     directory without the marker is wiped and rebuilt (half-built venvs are worse than none).
#   - Every cache is on blackhole (UV_CACHE_DIR, UV_PYTHON_INSTALL_DIR, PIP_CACHE_DIR, TMPDIR,
#     HF_HOME, TORCH_HOME, NLTK_DATA, NEMO_CACHE_DIR): $HOME is quota-limited and ~/.cache/uv was already 19 GB.
#     asr_bench.lsf exports the same paths, so what prefetch stages here is what the job finds.
#   - No `module load python3`: python3/3.11.9 exports PYTHONHOME and PYTHONPATH (module show),
#     which leak into any venv. The /appl9 interpreters are used by absolute path instead (they
#     run under `env -i`, checked; the prep venv-nordic was built the same way and ran in jobs).
#   - No cuda module: the wheels ship the CUDA runtime; a system CUDA on LD_LIBRARY_PATH can
#     shadow the pip cuDNN 9 that CTranslate2 needs.
#   - Prefetch runs online; the jobs run HF_HUB_OFFLINE=1 and fail fast on anything not staged.
#   - None of the model ids is gated (HF API, 2026-09-17). If one becomes gated:
#     HF_HOME=/dtu/blackhole/1e/205502/hf /dtu/blackhole/1e/205502/venvs/venv-asr/bin/hf auth login --token <token>
#   - uv's --torch-backend values are auto/cpu/cu118/cu126/cu128/cu130/rocm7.2/xpu (docs.astral.sh/uv/guides/integration/pytorch,
#     2026-09-17); cu129 is not among them, so pyinstall() sends cu129 (and any other unlisted value)
#     straight to --extra-index-url https://download.pytorch.org/whl/cu129 instead of trying --torch-backend=cu129 first.
#   - UNVERIFIED: that nemo_toolkit 3.0.0's PyPI wheel ships nemo.collections.speechlm2.SALM for
#     canary-qwen (the model card installs NeMo from git). The import check prints a WARN, not a
#     failure; if it warns, add the card's line to NEMO_PKGS.
#   - UNVERIFIED: NEMO_CACHE_DIR is the env var NeMo reads for its extracted .nemo cache.
set -uo pipefail      # no -e at top level: each build reports and the script goes on

BH=/dtu/blackhole/1e/205502
ROOT=$BH/nordic
VENVS=$BH/venvs
LOGS=$ROOT/logs
STAMP=${STAMP:-2026-09-17}          # bump to rebuild every venv
FORCE=${FORCE:-0}

export HF_HOME=$BH/hf TMPDIR=$BH/tmp TORCH_HOME=$BH/torch-home NLTK_DATA=$BH/nltk_data NEMO_CACHE_DIR=$BH/nemo-cache
export UV_CACHE_DIR=$BH/uv-cache UV_PYTHON_INSTALL_DIR=$BH/uv-python PIP_CACHE_DIR=$BH/pip-cache
export HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
unset PYTHONHOME PYTHONPATH HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
mkdir -p "$VENVS" "$LOGS" "$HF_HOME" "$TMPDIR" "$TORCH_HOME" "$NLTK_DATA" "$NEMO_CACHE_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PIP_CACHE_DIR"
module load ffmpeg/7.1 2>/dev/null || true
for d in /appl9/ffmpeg/7.1.0/lib64 /appl9/ffmpeg/7.1.0/lib; do   # same as asr_bench.lsf: the module sets no LD_LIBRARY_PATH;
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"  # torchcodec (pyannote-audio 4) wants FFmpeg shared libs at import
done

# Base interpreters: the module trees by path (see caveats). Fallback: a uv-managed
# CPython downloaded into UV_PYTHON_INSTALL_DIR (uv downloads when a version is missing).
PY311=/appl9/python/3.11.13/bin/python3.11; [ -x "$PY311" ] || PY311=3.11
PY312=/appl9/python/3.12.11/bin/python3.12; [ -x "$PY312" ] || PY312=3.12
UV=$(command -v uv || true)          # /zhome/38/c/205502/.local/bin/uv 0.10.12 on 2026-09-17

# --- package sets ---------------------------------------------------------------
ASR_PKGS=(torch torchaudio torchcodec "transformers>=5" accelerate peft faster-whisper
          "nvidia-cudnn-cu12==9.*" nvidia-cublas-cu12 num2words "jiwer>=4,<5" librosa soundfile numpy
          requests pydantic huggingface_hub)
WHISPERX_PKGS=(whisperx num2words soundfile librosa)   # num2words: run_whisperx.py exits without it (its normaliser), also for --prefetch
NEMO_PKGS=("nemo_toolkit[asr,speechlm2,cu12]" soundfile librosa)
QWEN_PKGS=(qwen-asr soundfile librosa)
VLLM_PKGS=(vllm requests pydantic huggingface_hub)

# --- helpers ---------------------------------------------------------------------
mkvenv() {  # mkvenv <dir> <python>
  if [ -n "$UV" ]; then
    "$UV" venv --python "$2" "$1"                              # docs.astral.sh/uv/pip/environments
  else
    case "$2" in /*) "$2" -m venv "$1" && "$1/bin/python" -m pip install -U pip ;;
      *) echo "no uv and no /appl9 interpreter for python $2"; return 1 ;; esac
  fi
}

pyinstall() {  # pyinstall <dir> <torch-backend e.g. cu128> <pkgs...>
  local venv=$1 cu=$2; shift 2
  if [ -n "$UV" ]; then
    # --torch-backend routes torch/torchaudio/torchvision/nvidia-* to the matching
    # PyTorch index (docs.astral.sh/uv/guides/integration/pytorch); fallback: the plain index URL.
    # uv's documented --torch-backend values are auto/cpu/cu118/cu126/cu128/cu130/rocm7.2/xpu — cu129
    # is not one of them, so go straight to --extra-index-url for it (and any other unlisted value)
    # instead of burning a doomed attempt every build.
    case "$cu" in
      cu118|cu126|cu128|cu130|cpu|auto|rocm*|xpu)
        "$UV" pip install --python "$venv/bin/python" --torch-backend="$cu" "$@" \
          || "$UV" pip install --python "$venv/bin/python" --extra-index-url "https://download.pytorch.org/whl/$cu" "$@"
        ;;
      *)
        "$UV" pip install --python "$venv/bin/python" --extra-index-url "https://download.pytorch.org/whl/$cu" "$@"
        ;;
    esac
  else
    "$venv/bin/python" -m pip install --extra-index-url "https://download.pytorch.org/whl/$cu" "$@"
  fi
}

pylist() { if [ -n "$UV" ]; then "$UV" pip list --python "$1/bin/python"; else "$1/bin/python" -m pip list; fi; }

FAILED=()
build() {  # build <name> <python> <torch-backend> <check-snippet> <pkgs...>
  local name=$1 py=$2 cu=$3 check=$4; shift 4
  local venv="$VENVS/$name" log="$LOGS/env-$name.log" marker="$VENVS/$name/.built-$STAMP"
  if [ -f "$marker" ] && [ "$FORCE" != 1 ]; then echo "[$name] built already ($marker); FORCE=1 to rebuild"; return 0; fi
  echo "[$name] building $venv   python=$py backend=$cu   log: $log"
  local t0=$SECONDS
  (
    set -e
    echo "=== $(date -Is) build $name  python=$py  backend=$cu"
    echo "packages: $*"
    rm -rf "$venv"
    mkvenv "$venv" "$py"
    "$venv/bin/python" --version
    pyinstall "$venv" "$cu" "$@"
    echo "--- import check"
    "$venv/bin/python" -c "$check"
    echo "--- installed"
    pylist "$venv"
  ) >> "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    touch "$marker"; echo "[$name] OK in $((SECONDS - t0))s"
    grep -A1 -- '--- import check' "$log" | tail -n 1 | sed 's/^/    /'
  else
    echo "[$name] FAILED (rc=$rc) after $((SECONDS - t0))s; tail of $log:"; tail -n 12 "$log" | sed 's/^/    | /'
    FAILED+=("$name")
  fi
}

# Import checks: prove the venv works on the login node (CPU) without touching a GPU.
CHECK_ASR='import torch, torchaudio, transformers, faster_whisper, ctranslate2, jiwer, num2words, librosa, soundfile, peft, accelerate, huggingface_hub
print("torch", torch.__version__, "cuda", torch.version.cuda, "torchaudio", torchaudio.__version__, "transformers", transformers.__version__, "faster_whisper", faster_whisper.__version__, "ctranslate2", ctranslate2.__version__, "hf_hub", huggingface_hub.__version__)
import nvidia.cublas.lib, nvidia.cudnn.lib'
CHECK_WHISPERX='import torch, whisperx, faster_whisper, nltk, num2words, soundfile, librosa
print("torch", torch.__version__, "cuda", torch.version.cuda, "whisperx", getattr(whisperx, "__version__", "?"), "faster_whisper", faster_whisper.__version__, "nltk", nltk.__version__)'
CHECK_NEMO='import torch, nemo, nemo.collections.asr as asr
print("torch", torch.__version__, "cuda", torch.version.cuda, "nemo", nemo.__version__)
try:
    import nemo.collections.speechlm2 as slm; print("speechlm2 SALM", hasattr(slm.models, "SALM"))
except Exception as e:
    print("WARN speechlm2 import failed (canary-qwen will not run):", repr(e)[:300])'
CHECK_QWEN='import torch, transformers, qwen_asr
print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__, "qwen_asr", getattr(qwen_asr, "__version__", "?"))'
CHECK_VLLM='import importlib.metadata as m
print("vllm", m.version("vllm"), "torch", m.version("torch"), "transformers", m.version("transformers"), "pydantic", m.version("pydantic"))'
# (vllm itself is not imported: on a GPU-less login node its import probes for a device.)

venvs() {
  echo "=== venvs $(date -Is)  uv=${UV:-none}  py311=$PY311  py312=$PY312  venvs=$VENVS"
  build venv-asr      "$PY311" cu129 "$CHECK_ASR"      "${ASR_PKGS[@]}"
  build venv-whisperx "$PY311" cu128 "$CHECK_WHISPERX" "${WHISPERX_PKGS[@]}"
  build venv-nemo     "$PY312" cu126 "$CHECK_NEMO"     "${NEMO_PKGS[@]}"
  build venv-qwen     "$PY311" cu128 "$CHECK_QWEN"     "${QWEN_PKGS[@]}"
  build venv-vllm     "$PY312" cu129 "$CHECK_VLLM"     "${VLLM_PKGS[@]}"
  echo "=== venvs done $(date -Is): $((SECONDS / 60)) min; failed: ${FAILED[*]:-none}"
  du -sh "$VENVS"/venv-* 2>/dev/null
}

# --- model weights ---------------------------------------------------------------
# HF repo ids used by the runners (their docstrings / faster_whisper.utils._MODELS).
# Qwen/Qwen3-1.7B and nvidia/canary-1b-flash are not runners of their own: nvidia/canary-qwen-2.5b's
# config.json names them (pretrained_llm, pretrained_asr, pretrained_weights=false) and
# SALM.from_pretrained resolves both at load (AutoConfig for the Qwen3 config, hf_hub_download of the
# whole canary-1b-flash .nemo, kept: *.nemo is not in the ignore list), so offline they must be cached.
ASR_MODELS=(
  Systran/faster-whisper-large-v3 mobiuslabsgmbh/faster-whisper-large-v3-turbo Systran/faster-distil-whisper-large-v3
  openai/whisper-large-v3 distil-whisper/distil-large-v3
  nvidia/parakeet-tdt-0.6b-v2 nvidia/parakeet-tdt-0.6b-v3 nvidia/canary-qwen-2.5b
  Qwen/Qwen3-1.7B nvidia/canary-1b-flash
  Qwen/Qwen3-ASR-1.7B Qwen/Qwen3-ForcedAligner-0.6B
  ibm-granite/granite-speech-4.1-2b-plus ibm-granite/granite-speech-4.1-2b
)
# Answering models: each id answered https://huggingface.co/api/models/<id> with gated=false (2026-09-17).
# The two 35B-A3B MoEs are the FP8 repos (36 GB): their BF16 twins are 72 GB and leave no KV cache on 80 GB.
LLM_MODELS_DEFAULT="Qwen/Qwen3.8-27B Qwen/Qwen3.6-27B openai/gpt-oss-20b Qwen/Qwen3.6-35B-A3B-FP8 google/gemma-4-31B-it zai-org/GLM-4.7-Flash openai/gpt-oss-120b mistralai/Ministral-3-14B-Instruct-2512 Qwen/Qwen3.5-27B Qwen/Qwen3.5-35B-A3B-FP8 Qwen/Qwen3-30B-A3B-Instruct-2507"
LLM_MODELS=${LLM_MODELS:-$LLM_MODELS_DEFAULT}
PREFETCH_LLM=${PREFETCH_LLM:-1}
PREFETCH_SALM=${PREFETCH_SALM:-1}

cached() { [ -d "$HF_HOME/hub/models--${1//\//--}" ]; }

hfpy() {  # a python with huggingface_hub, for downloads
  for v in venv-asr venv-vllm; do [ -x "$VENVS/$v/bin/python" ] && { echo "$VENVS/$v/bin/python"; return 0; }; done
  return 1
}

fetch_repo() {  # fetch_repo <repo id>: snapshot_download minus formats we never load
  local id=$1 py; py=$(hfpy) || { echo "no venv with huggingface_hub yet (run: env.sh venvs)"; return 1; }
  "$py" - "$id" <<'PY'
import sys, time
from huggingface_hub import snapshot_download
repo = sys.argv[1]
t0 = time.time()
# skip: original/ and metal/ (gpt-oss reference weights), gguf/pth/h5/msgpack/ot/onnx duplicates.
# Keep *.bin: Systran faster-whisper repos ship CTranslate2 weights as model.bin.
p = snapshot_download(repo, ignore_patterns=["original/*", "metal/*", "*.gguf", "*.pth", "*.h5", "*.msgpack", "*.ot", "onnx/*", "*.onnx"])
print(f"DONE {repo} -> {p}  ({time.time() - t0:.0f}s)")
PY
}

fetch_torchaudio() {  # MMS_FA (align_mms.py) and the wav2vec2 English aligner whisperx uses, into TORCH_HOME
  local py="$VENVS/venv-asr/bin/python"
  [ -x "$py" ] || { echo "venv-asr missing: skip torchaudio pipelines"; return 1; }
  "$py" - <<'PY'
import torchaudio
for name in ("MMS_FA", "WAV2VEC2_ASR_BASE_960H"):
    bundle = getattr(torchaudio.pipelines, name)     # docs.pytorch.org/audio/main/generated/torchaudio.pipelines.MMS_FA.html
    bundle.get_model()                               # downloads to $TORCH_HOME/hub/checkpoints
    print("DONE torchaudio.pipelines." + name)
PY
}

fetch_whisperx() {  # run_whisperx.py --prefetch: nltk punkt_tab into NLTK_DATA (whisperx.align calls
  # nltk.download('punkt_tab') and raises RuntimeError when that returns False offline), the wav2vec2
  # aligner through whisperx's own loader, and the CT2 weights (already cached above, a no-op).
  # Its check_caches() refuses HF_HOME/TORCH_HOME/NLTK_DATA under $HOME; all three are exported above.
  local py="$VENVS/venv-whisperx/bin/python"
  [ -x "$py" ] || { echo "venv-whisperx missing: skip whisperx prefetch (nltk punkt_tab)"; return 1; }
  (cd "$ROOT/medical-appointment" && "$py" bench/asr/run_whisperx.py --prefetch)
}

warm_salm() {  # canary-qwen: run SALM.from_pretrained once on the login node (CPU, no GPU needed) so the
  # exact load path is exercised online: nvidia/canary-qwen-2.5b, then Qwen/Qwen3-1.7B (AutoConfig) and
  # nvidia/canary-1b-flash (ASRModel.from_pretrained(..., return_config=True), the whole .nemo), through
  # nemo/collections/speechlm2/parts/pretrained.py. Whatever NeMo caches lands in HF_HOME and
  # NEMO_CACHE_DIR, the paths asr_bench.lsf exports. This is run_canary_qwen.py's documented warm-up.
  # Loads ~5 GB of weights into RAM and unpacks the .nemo under TMPDIR; PREFETCH_SALM=0 skips it.
  local py="$VENVS/venv-nemo/bin/python"
  [ -x "$py" ] || { echo "venv-nemo missing: skip SALM warm-up"; return 1; }
  "$py" -c "from nemo.collections.speechlm2.models import SALM; SALM.from_pretrained('nvidia/canary-qwen-2.5b'); print('DONE SALM warm-up: nvidia/canary-qwen-2.5b + Qwen/Qwen3-1.7B + nvidia/canary-1b-flash')"
}

prefetch() {
  echo "=== prefetch $(date -Is)  HF_HOME=$HF_HOME  TORCH_HOME=$TORCH_HOME  (log: $LOGS/env-prefetch.log)"
  local log="$LOGS/env-prefetch.log" ok=0 bad=()
  {
    echo "=== $(date -Is) prefetch start"
    for id in "${ASR_MODELS[@]}"; do
      if cached "$id" && [ "$FORCE" != 1 ]; then echo "have $id"; ok=$((ok+1)); continue; fi
      echo "--- $(date -Is) $id"; fetch_repo "$id" && ok=$((ok+1)) || bad+=("$id")
    done
    fetch_torchaudio || bad+=("torchaudio-pipelines")
    echo "--- $(date -Is) whisperx prefetch (nltk punkt_tab -> $NLTK_DATA)"
    fetch_whisperx || bad+=("whisperx-prefetch")
    if [ "$PREFETCH_SALM" = 1 ]; then
      echo "--- $(date -Is) SALM warm-up (canary-qwen-2.5b + Qwen3-1.7B + canary-1b-flash)"
      warm_salm || bad+=("salm-warmup")
    else
      echo "PREFETCH_SALM=0: SALM warm-up skipped (the two extra repos are still fetched above)"
    fi
    if [ "$PREFETCH_LLM" = 1 ]; then
      for id in $LLM_MODELS; do
        if cached "$id" && [ "$FORCE" != 1 ]; then echo "have $id"; ok=$((ok+1)); continue; fi
        echo "--- $(date -Is) $id"; fetch_repo "$id" && ok=$((ok+1)) || bad+=("$id")
      done
    else
      echo "PREFETCH_LLM=0: LLM weights skipped"
    fi
    echo "=== $(date -Is) prefetch done: $ok ok, failed: ${bad[*]:-none}"
    du -sh "$HF_HOME/hub" "$TORCH_HOME" 2>/dev/null
  } 2>&1 | tee -a "$log"
}

check() {
  echo "=== venvs ($VENVS):"
  for v in venv-asr venv-whisperx venv-nemo venv-qwen venv-vllm; do
    if [ -f "$VENVS/$v/.built-$STAMP" ]; then printf '  %-14s built  %s\n' "$v" "$("$VENVS/$v/bin/python" --version 2>&1)"
    elif [ -d "$VENVS/$v" ]; then printf '  %-14s INCOMPLETE (no marker; env.sh venvs rebuilds it)\n' "$v"
    else printf '  %-14s missing\n' "$v"; fi
  done
  echo "=== HF cache ($HF_HOME/hub):"
  for id in "${ASR_MODELS[@]}" $LLM_MODELS; do
    if cached "$id"; then printf '  ok       %-45s %s\n' "$id" "$(du -sh "$HF_HOME/hub/models--${id//\//--}" 2>/dev/null | cut -f1)"
    else printf '  MISSING  %s\n' "$id"; fi
  done
  echo "=== torch hub checkpoints ($TORCH_HOME/hub/checkpoints): $(ls "$TORCH_HOME/hub/checkpoints" 2>/dev/null | tr '\n' ' ')"
  if [ -d "$NLTK_DATA/tokenizers/punkt_tab" ]; then echo "=== nltk punkt_tab ($NLTK_DATA): ok"; else echo "=== nltk punkt_tab ($NLTK_DATA): MISSING (env.sh prefetch)"; fi
  echo "=== NeMo cache ($NEMO_CACHE_DIR): $(du -sh "$NEMO_CACHE_DIR" 2>/dev/null | cut -f1)"
  echo "=== disk:"; df -h "$BH" | tail -n 1; du -sh "$HF_HOME" "$VENVS" "$UV_CACHE_DIR" 2>/dev/null
}

case "${1:-venvs}" in
  venvs)    venvs ;;
  prefetch) prefetch ;;
  check)    check ;;
  all)      venvs; prefetch ;;
  *) echo "usage: $0 venvs|prefetch|check|all   (FORCE=1, STAMP=..., PREFETCH_LLM=0, PREFETCH_SALM=0, LLM_MODELS=\"...\")" >&2; exit 2 ;;
esac
