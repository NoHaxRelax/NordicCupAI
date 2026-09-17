#!/bin/bash
# bench/hpc/sync.sh: move the medical-appointment case between the laptop and
# DTU blackhole. Runs on the laptop (Git Bash on Windows, or any Linux/macOS
# shell); never on the cluster.
#
#   bash bench/hpc/sync.sh up      code + data/audio + data/question_train.csv + transcripts/ + bench/
#                                  -> dtu:/dtu/blackhole/1e/205502/nordic/medical-appointment
#                                  excluding .git, bench/results, __pycache__, *.pyc, models/, .venv
#   bash bench/hpc/sync.sh down    bench/results/ and transcripts/ <- cluster (adds/overwrites, never deletes local files)
#   bash bench/hpc/sync.sh ls      what is on the cluster right now (sizes of transcripts/ and bench/results/)
#
# Requirements: an ssh alias `dtu` in ~/.ssh/config (dtu-hpc-shared skill), the
# DTU VPN connected, and either rsync or (fallback) tar + ssh. Git Bash on the
# laptop has no rsync (checked 2026-09-17), so there the script streams a tar
# over ssh: it re-sends everything (~100 MB: 73 MB of MP3 plus code and
# transcripts) instead of only the deltas, but the result is the same and
# nothing is ever deleted on either side. WSL's Ubuntu has rsync, but its ssh
# does not see the Windows key/alias, so it is not used.
#
# Environment:
#   SYNC_HOST   ssh alias (default dtu; `dtu-transfer` = transfer.gbar.dtu.dk for big pulls)
#   REMOTE      remote case folder (default /dtu/blackhole/1e/205502/nordic/medical-appointment)
#
# Caveats:
#   - `up` never deletes on the cluster (a stale transcript there is harmless:
#     every runner skips outputs that already exist unless --force).
#   - `down` overwrites a local file that has the same path (cluster results are
#     the newer ones); it never removes anything local.
#   - The remote path is literal: $BLACKHOLE is only set in a login shell on the
#     cluster, and rsync/tar over ssh do not get one.
set -euo pipefail

CASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"     # medical-appointment
HOST="${SYNC_HOST:-dtu}"
REMOTE="${REMOTE:-/dtu/blackhole/1e/205502/nordic/medical-appointment}"

# Same list for rsync and tar: patterns match a path component at any depth.
EXCLUDES=(.git bench/results __pycache__ '*.pyc' models .venv .impeccable .pytest_cache .mypy_cache)
RS_EXCL=(); TAR_EXCL=()
for e in "${EXCLUDES[@]}"; do RS_EXCL+=(--exclude "$e"); TAR_EXCL+=("--exclude=$e"); done

have_rsync() { command -v rsync >/dev/null 2>&1; }

case "${1:-}" in
  up)
    echo "up: $CASE -> $HOST:$REMOTE"
    ssh -q "$HOST" "mkdir -p '$REMOTE' '$REMOTE/bench/results' '$REMOTE/transcripts'"
    if have_rsync; then
      rsync -az --info=progress2 "${RS_EXCL[@]}" "$CASE/" "$HOST:$REMOTE/"
    else
      echo "(no rsync here: streaming a tar over ssh, full copy, no deletes)"
      # GNU tar: --exclude patterns are unanchored, so `bench/results` matches
      # ./bench/results and `__pycache__` matches at any depth.
      tar -C "$CASE" "${TAR_EXCL[@]}" -czf - . | ssh -q "$HOST" "tar -C '$REMOTE' -xzf -"
    fi
    # The repo has core.autocrlf=true, so a Windows checkout can carry CRLF shell
    # scripts; a '\r' after '#!/bin/bash' or on a '#BSUB' line breaks them on
    # Linux. Normalise every shell/LSF script under bench/ on the remote side.
    ssh -q "$HOST" "cd '$REMOTE' && find bench -type f \( -name '*.sh' -o -name '*.lsf' \) -exec sed -i 's/\r\$//' {} + && chmod +x bench/hpc/*.sh bench/llm/*.sh 2>/dev/null; true"
    echo "done. remote listing:"
    ssh -q "$HOST" "cd '$REMOTE' && ls && echo && du -sh data transcripts bench 2>/dev/null"
    ;;
  down)
    echo "down: $HOST:$REMOTE/{bench/results,transcripts} -> $CASE"
    mkdir -p "$CASE/bench/results" "$CASE/transcripts"
    if have_rsync; then
      rsync -az --info=progress2 "$HOST:$REMOTE/bench/results/" "$CASE/bench/results/"
      rsync -az --info=progress2 "$HOST:$REMOTE/transcripts/" "$CASE/transcripts/"
    else
      echo "(no rsync here: streaming a tar over ssh, adds/overwrites, no deletes)"
      ssh -q "$HOST" "cd '$REMOTE' && tar -czf - \$(ls -d bench/results transcripts 2>/dev/null)" \
        | tar -C "$CASE" -xzf -
    fi
    echo "done:"
    ls "$CASE/bench/results" 2>/dev/null | sed 's/^/  bench\/results\//'
    echo "  transcripts: $(ls "$CASE/transcripts" 2>/dev/null | wc -l) files"
    ;;
  ls)
    ssh -q "$HOST" "cd '$REMOTE' 2>/dev/null || { echo 'remote case folder missing (run: sync.sh up)'; exit 1; };
      echo '== transcripts by tag =='; ls transcripts 2>/dev/null | sed -E 's/^conversation_sample_[0-9]+\.//' | sort | uniq -c;
      echo '== bench/results =='; du -sh bench/results/* 2>/dev/null; ls bench/results/asr bench/results/llm 2>/dev/null"
    ;;
  *)
    echo "usage: $0 up|down|ls" >&2; exit 2 ;;
esac
