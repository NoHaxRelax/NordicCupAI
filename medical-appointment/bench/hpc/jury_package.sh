#!/bin/bash
# bench/hpc/jury_package.sh [OUT.zip]: the archive the Scientific Jury asks the top five for.
#
# Tracked files of the case folder at HEAD (so no venvs, results, request dumps, transcripts or
# anything under the repo-root .claude/), with research/12-jury-submission.md copied to the archive
# root as README_JURY.md. Refuses to run on a dirty tree: the archive must equal what was served.
#
#     bash bench/hpc/jury_package.sh                       # -> jury_submission_<HEAD>.zip in the case folder
set -euo pipefail
CASE="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$CASE"
if [ -n "$(git status --porcelain -- .)" ]; then echo "working tree not clean: commit first"; git status --short -- . | head; exit 1; fi
SHA=$(git rev-parse --short HEAD)
OUT=${1:-$CASE/jury_submission_$SHA.zip}
# From the repo root with the folder as pathspec: `git archive HEAD:medical-appointment` run inside the
# folder applies the folder as an implicit pathspec on top of the subtree and yields an empty archive.
ROOT=$(git rev-parse --show-toplevel)
git -C "$ROOT" archive --format=zip -o "$OUT" HEAD medical-appointment
python - "$OUT" <<'PY'
import sys, zipfile, pathlib
out = sys.argv[1]
readme = pathlib.Path('research/12-jury-submission.md').read_text(encoding='utf-8')
with zipfile.ZipFile(out, 'a', zipfile.ZIP_DEFLATED) as z:
    z.writestr('medical-appointment/README_JURY.md', readme)
    names = z.namelist()
# bench/results/probe (hosted-model probe answers on training files) is tracked on purpose; secrets and dumps are not.
bad = [n for n in names if '.claude' in n or 'nordic-api-key' in n or 'nordic-control-token' in n or '.venv' in n or 'request_dump' in n]
print(f'{out}: {len(names)} entries, {pathlib.Path(out).stat().st_size/1e6:.1f} MB', 'CLEAN' if not bad else f'FORBIDDEN CONTENT: {bad[:5]}')
sys.exit(1 if bad else 0)
PY
echo "built from HEAD $SHA; add the served pod URL and the final score to README_JURY.md before sending"
