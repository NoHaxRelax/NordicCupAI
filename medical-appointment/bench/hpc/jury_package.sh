#!/bin/bash
# bench/hpc/jury_package.sh [OUT.zip]: the archive the Scientific Jury asks the top five for.
#
# Tracked files of the case folder at HEAD: code, research log, training data, the tracked probe
# results and large-v3 transcripts, and bench/mine with the recovered validation labels (disclosed).
# Never the venvs, request dumps or anything under the repo-root .claude/. research/12-jury-submission.md
# goes to the archive root as README_JURY.md, with the served URL and final score appended from the
# environment so the tree stays clean and HEAD stays on the served tag. Refuses to run on a dirty tree.
#
#     bash bench/hpc/jury_package.sh                                   # before the attempt
#     SERVED_URL=https://<POD_ID>-9054.proxy.runpod.net/predict FINAL_SCORE=0.8xxx bash bench/hpc/jury_package.sh
set -euo pipefail
CASE="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$CASE"
if [ -n "$(git status --porcelain -- .)" ]; then echo "working tree not clean: commit first"; git status --short -- . | head; exit 1; fi
SHA=$(git rev-parse --short HEAD)
OUT=${1:-$CASE/jury_submission_$SHA.zip}
# From the repo root with the folder as pathspec: `git archive HEAD:medical-appointment` run inside the
# folder applies the folder as an implicit pathspec on top of the subtree and yields an empty archive.
ROOT=$(git rev-parse --show-toplevel)
git -C "$ROOT" archive --format=zip -o "$OUT" HEAD medical-appointment
SERVED_URL="${SERVED_URL:-}" FINAL_SCORE="${FINAL_SCORE:-}" python - "$OUT" <<'PY'
import os, sys, zipfile, pathlib
out = sys.argv[1]
readme = pathlib.Path('research/12-jury-submission.md').read_text(encoding='utf-8')
url, score = os.environ.get('SERVED_URL', ''), os.environ.get('FINAL_SCORE', '')
readme += '\n## Evaluation attempt\n\n- Served URL: ' + (url or 'TO BE FILLED') + '\n- Final score: ' + (score or 'TO BE FILLED') + '\n'
with zipfile.ZipFile(out, 'a', zipfile.ZIP_DEFLATED) as z:
    z.writestr('medical-appointment/README_JURY.md', readme)
    names = z.namelist()
# bench/results/probe (hosted-model probe answers on training files) is tracked on purpose; secrets and dumps are not.
bad = [n for n in names if '.claude' in n or 'nordic-api-key' in n or 'nordic-control-token' in n or '.venv' in n or 'request_dump' in n]
print(f'{out}: {len(names)} entries, {pathlib.Path(out).stat().st_size/1e6:.1f} MB', 'CLEAN' if not bad else f'FORBIDDEN CONTENT: {bad[:5]}')
sys.exit(1 if bad else 0)
PY
echo "built from HEAD $SHA; rerun with SERVED_URL=... FINAL_SCORE=... after the attempt to stamp them into README_JURY.md"
