#!/bin/bash
# Supervisor: keep api.py alive; restart on any exit, with a timestamped note.
# Per-restart overrides are read from serve.env next to this script (e.g. SPAN_ON_NO=0),
# so an A/B flip only needs: edit serve.env, kill the python api.py process.
cd "C:/Users/edlun/Desktop/lucky shots/NordicCupAI/medical-appointment"
SCR="C:/Users/edlun/AppData/Local/Temp/claude/c--Users-edlun-Desktop-lucky-shots-NordicCupAI/bf48d8ae-2b16-4039-be3b-c570527b37ee/scratchpad"
LOG="$SCR/api.log"
while true; do
  export TRANSCRIPT_CACHE=0 ASR_MODEL=large-v3-turbo LLM_MODEL=qwen3:4b KMP_DUPLICATE_LIB_OK=TRUE
  unset SPAN_ON_NO START_RULE START_OFFSET END_OFFSET ASR_CLEAN REQUEST_DUMP_DIR
  unset LLM_BACKEND LLM_URL LLM_VARIANT LLM_NO_THINK LLM_TIMEOUT LLM_DEADLINE LLM_MAX_TOKENS
  unset LLM_FALLBACK_URL LLM_FALLBACK_MODEL UNIT_SPLIT PREDICT_DEADLINE
  [ -f "$SCR/serve.env" ] && set -a && . "$SCR/serve.env" && set +a
  echo "$(date -Is) supervisor: starting api.py with LLM_BACKEND=${LLM_BACKEND:-ollama} LLM_MODEL=${LLM_MODEL} LLM_VARIANT=${LLM_VARIANT:-default} UNIT_SPLIT=${UNIT_SPLIT:-sentence} SPAN_ON_NO=${SPAN_ON_NO:-default}" >> "$LOG.supervisor"
  python api.py >> "$LOG" 2>> "$LOG.err"
  echo "$(date -Is) supervisor: api.py exited with code $? ; restarting in 3s" >> "$LOG.supervisor"
  sleep 3
done
