#!/bin/bash
# snapshot nightsim sources + overnight notes/configs into the local repo and commit (no push). usage: commit.sh "message"
export DEVELOPER_DIR=/Library/Developer/CommandLineTools
O=/Users/bumblebee/Github_repos/Projects/nordic-ai-cup-2026/artifacts/overnight
T=$O/repo/survival-simulator/oscar-overnight-cpp
mkdir -p $T/nightsim $T/notes $T/configs $T/results
cp /Users/bumblebee/Github_repos/Projects/nordic-ai-cup-2026/survival/nightsim/{_nengine.cpp,_npolicy.hpp,__init__.py,build.py,run.py} $T/nightsim/
cp $O/{STATUS.md,PLAN.md,BEST.md,lucas-trap-report.md} $T/notes/ 2>/dev/null
cp $O/{night.sh,ana.py,gen.py,pods.txt,commit.sh} $T/ 2>/dev/null
cp $O/configs-all.json $O/cfg-*.json $T/configs/ 2>/dev/null
cp $O/results-summary.md $T/results/ 2>/dev/null
cd $O/repo && git add -A survival-simulator/oscar-overnight-cpp && git commit -q -m "$1

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" && git log --oneline -1
