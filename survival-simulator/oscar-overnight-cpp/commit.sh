#!/bin/bash
# snapshot nightsim sources + overnight notes/configs into the local repo and commit (no push). usage: commit.sh "message"
export DEVELOPER_DIR=/Library/Developer/CommandLineTools
O=/Users/bumblebee/Github_repos/Projects/nordic-ai-cup-2026/artifacts/overnight
T=$O/repo/survival-simulator/oscar-overnight-cpp
mkdir -p $T/nightsim $T/notes $T/configs $T/results
N=/Users/bumblebee/Github_repos/Projects/nordic-ai-cup-2026/survival/nightsim
cp $N/*.py $N/_nengine.cpp $N/_npolicy.hpp $T/nightsim/
cp $O/{STATUS.md,PLAN.md,BEST.md,lucas-trap-report.md} $T/notes/ 2>/dev/null
cp $O/{ana.py,gen.py,g_ana.py,esc_ana.py,edge_ana.py,diag_ana.py,ts_ana.py,commit.sh} $T/ 2>/dev/null
cp $O/README-team.md $T/README.md
grep -v '^#' $O/pods.txt | head -0 > /dev/null; { echo '# name host port pod_id venv   (one line per CPU pod; lines starting with # are ignored)'; echo 'p1 213.173.105.80 12345 yourpodid /workspace/night/.venv'; } > $T/pods.example.txt
rm -f $T/pods.txt
# portable driver: paths relative to the script, sources from ./nightsim, key from NIGHT_SSH_KEY
sed -e 's#^D=.*#D=${NIGHT_HOME:-$(cd "$(dirname "$0")" \&\& pwd)}#' \
    -e 's#^SRC=.*#SRC=${NIGHT_SRC:-$D/nightsim}#' \
    -e 's#-i $HOME/.ssh/id_ed25519#-i ${NIGHT_SSH_KEY:-$HOME/.ssh/id_ed25519}#' \
    -e 's#tar -C $SRC/.. -czf - nightsim/.*DTOs.py#tar -C $SRC/.. -czf - $(cd $SRC/.. \&\& ls nightsim/*.py nightsim/*.cpp nightsim/*.hpp)#' \
    $O/night.sh > $T/night.sh; chmod +x $T/night.sh
cp $O/configs-all.json $O/best-configs.json $O/cfg-*.json $T/configs/ 2>/dev/null
cp $O/results-summary.md $T/results/ 2>/dev/null
cd $O/repo && git add -A survival-simulator/oscar-overnight-cpp && git commit -q -m "$1

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" && git log --oneline -1
