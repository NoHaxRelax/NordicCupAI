#!/usr/bin/env bash
# Read-only status for this fork's burst experiment. No scheduler or thread routing.
set -u
key=/home/Ucals/.ssh/runpod_codex_team
hosts=(213.173.111.99 157.157.221.30 157.157.221.177 157.157.221.29)
ports=(27353 37210 29649 44596)
for i in 0 1 2 3; do
 printf 'Runpod %s: ' "$((i+1))"
 ssh -o BatchMode=yes -o ConnectTimeout=5 -i "$key" -p "${ports[$i]}" "root@${hosts[$i]}" 'cd /workspace/burst-score && if test -f pilot/games.jsonl; then printf "%s/128 games | " "$(wc -l < pilot/games.jsonl)"; fi; tail -n 1 pilot.log' 2>&1
done
printf 'SSH PC: '
ssh -o BatchMode=yes -o ConnectTimeout=5 pc 'cd ~/lucas-burst-score && printf "%s/32 games | " "$(wc -l < pilot-pc/games.jsonl)"; tail -n 1 pilot-pc.log' 2>&1
