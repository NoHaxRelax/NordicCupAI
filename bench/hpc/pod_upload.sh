#!/bin/bash
# bench/hpc/pod_upload.sh HOST PORT [SSH_KEY]: ship the case folder to a pod and prove it arrived.
#
# Run from the laptop (Git Bash). Tars the working tree minus the things that must never go
# (.git, venvs, results, transcripts, request dumps) into /workspace/medical-appointment, strips
# CR from scripts (a Windows checkout may carry CRLF), then compares md5 of the files that matter
# against the committed blobs in HEAD. The repo-root .claude/ (API key, control token, guardrail)
# is outside the case folder and can never be included by construction.
#
#     bash bench/hpc/pod_upload.sh 216.81.248.126 11358      # host/port from the RunPod MCP get-pod
set -euo pipefail
HOST=${1:?usage: pod_upload.sh HOST PORT [KEY]}; PORT=${2:?usage: pod_upload.sh HOST PORT [KEY]}
KEY=${3:-$HOME/.ssh/id_ed25519}
CASE="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE=/workspace/medical-appointment
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -i "$KEY" -p "$PORT" "root@$HOST")
EXCL=(--exclude=.git --exclude='bench/.venv*' --exclude=bench/results --exclude=transcripts
      --exclude=request_dump --exclude=__pycache__ --exclude='*.pyc' --exclude='*.tgz' --exclude=medvoice
      --exclude=data/audio --exclude=bench/ref --exclude=bench/mine/requests.jsonl)
# data/audio and bench/ref are never read when serving (84 MB -> 0.5 MB). Without the warm-up clip model.warm_up()
# uses 20 s of zeros, which is how the pod that scored ran (api.log: 'Processing audio with duration 00:20.000').

ssh-keygen -R "[$HOST]:$PORT" >/dev/null 2>&1 || true   # disposable pod: a reused IP:port carries a new host key
echo "uploading $CASE -> root@$HOST:$PORT:$REMOTE"
"${SSH[@]}" "mkdir -p $REMOTE /workspace/logs"
tar -C "$CASE" "${EXCL[@]}" -czf - . | "${SSH[@]}" "tar -C $REMOTE --no-same-owner -xzf -"
"${SSH[@]}" "cd $REMOTE && find . \( -name '*.sh' -o -name '*.py' -o -name '*.json' \) -print0 | xargs -0 sed -i 's/\r\$//' && chmod +x bench/hpc/*.sh"

# Verify: md5 on the pod must equal md5 of the committed blob (LF-normalised) for every serving file.
FILES=(model.py example.py api.py dtos.py bench/llm/prompts.py bench/llm/pool/large-v3-turbo.json
       bench/hpc/pod_bringup.sh bench/hpc/pod_bootstrap.sh bench/hpc/pod_endpoint.sh)
fail=0
remote=$("${SSH[@]}" "cd $REMOTE && md5sum ${FILES[*]}")
for f in "${FILES[@]}"; do
  want=$(cd "$CASE" && git show "HEAD:$(git rev-parse --show-prefix)$f" | md5sum | cut -d' ' -f1)   # works whether the case is the repo root or a subfolder
  got=$(echo "$remote" | awk -v f="$f" '$2==f {print $1}')
  if [ "$want" = "$got" ]; then echo "  ok   $f"; else echo "  DIFF $f  (HEAD $want, pod $got)"; fail=1; fi
done
[ $fail -eq 0 ] && echo "UPLOAD_OK: pod matches HEAD $(cd "$CASE" && git rev-parse --short HEAD)" \
                || { echo "UPLOAD_MISMATCH: commit your working tree or check the file above"; exit 1; }
echo "next, on the pod:  nohup bash $REMOTE/bench/hpc/pod_bringup.sh > /workspace/logs/bringup_outer.log 2>&1 &"
