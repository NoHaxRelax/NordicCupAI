#!/bin/bash
# Overnight pod driver (no competition API anywhere). usage:
#   night.sh ssh P 'cmd' | bootstrap P | deploy P.. | launch P JOB 'run.py args' | pull P JOB | status | kill P JOB | pods
D=${NIGHT_HOME:-$(cd "$(dirname "$0")" && pwd)}
SRC=${NIGHT_SRC:-$D/nightsim}
KH=$D/known_hosts
row() { grep "^$1 " $D/pods.txt; }
host() { row $1 | awk '{print $2}'; }; port() { row $1 | awk '{print $3}'; }; venv() { row $1 | awk '{print $5}'; }
SSHO="-o IdentitiesOnly=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KH -i ${NIGHT_SSH_KEY:-$HOME/.ssh/id_ed25519}"
sshp() { local p=$1; shift; ssh $SSHO -p $(port $p) root@$(host $p) "$@"; }
allpods() { grep -v '^#' $D/pods.txt | awk '{print $1}'; }
cmd=$1; shift
case $cmd in
  pods) allpods ;;
  ssh) sshp "$@" ;;
  bootstrap) p=$1
    sshp $p "mkdir -p /workspace/night/code/survival /workspace/night/runs && (test -x /workspace/night/.venv/bin/python || python3 -m venv /workspace/night/.venv) && /workspace/night/.venv/bin/pip -q install numpy==2.3.5 pydantic 2>&1 | tail -1; /workspace/night/.venv/bin/python -c 'import numpy; print(\"numpy\", numpy.__version__)'; which c++; nproc" ;;
  deploy) for p in "$@"; do
      tar -C $SRC/.. -czf - $(cd $SRC/.. && ls nightsim/*.py nightsim/*.cpp nightsim/*.hpp) \
        | sshp $p "mkdir -p /workspace/night/code/survival /workspace/night/runs && cd /workspace/night/code/survival && tar -xzf - && $(venv $p)/bin/python nightsim/build.py >/workspace/night/build.log 2>&1 && echo \"$p built \$(md5sum nightsim/_npolicy.hpp | cut -c1-8)\" || { echo \"$p BUILD FAILED\"; tail -20 /workspace/night/build.log; }" &
    done; wait ;;
  launch) p=$1; job=$2; args=$3; envs=${4:-}; script=${5:-run.py}
    sshp $p "cd /workspace/night/code/survival || exit 1; setsid nohup env $envs $(venv $p)/bin/python nightsim/$script $args --out /workspace/night/runs/$job.jsonl > /workspace/night/runs/$job.log 2>&1 < /dev/null & echo launched $job on $p" ;;
  pull) p=$1; job=$2
    scp -q $SSHO -P $(port $p) root@$(host $p):/workspace/night/runs/$job.jsonl $D/runs/$job-$p.jsonl 2>/dev/null && echo "$(wc -l < $D/runs/$job-$p.jsonl) rows $job-$p" ;;
  kill) p=$1; job=$2; sshp $p "pkill -f 'runs/$job[.]jsonl'; echo killed $job on $p" ;;
  qstart) for p in "$@"; do cat $D/qd.sh | sshp $p "cat > /workspace/night/qd.sh; pgrep -f "bash /workspace/night/[q]d.sh" > /dev/null || (setsid nohup bash /workspace/night/qd.sh > /dev/null 2>&1 < /dev/null &); echo qd up on $p" & done; wait ;;
  q) p=$1; name=$2; shift 2; printf '%s\n' "cd /workspace/night/code/survival" "$*" | sshp $p "mkdir -p /workspace/night/queue; cat > /workspace/night/queue/$name.sh; echo queued $name on $p" ;;
  qstat) for p in $(allpods); do echo "$p: $(sshp $p "ls /workspace/night/queue/ 2>/dev/null | grep -E '\.(sh|running)$' | tr '\n' ' '; tail -1 /workspace/night/queue/qd.log 2>/dev/null; cut -d' ' -f1 /proc/loadavg" 2>/dev/null | tr '\n' ' ')" & done; wait ;;
  status) for p in $(allpods); do echo "$p: $(sshp $p "cat /proc/loadavg | cut -d' ' -f1; ps aux | grep -c '[n]ightsim/run.py'; for f in /workspace/night/runs/*.log; do test -f \$f && echo \$(basename \$f .log):\$(wc -l < \${f%.log}.jsonl 2>/dev/null):\$(tail -c 60 \$f | tr '\n' ' '); done" 2>&1 | tr '\n' ' ')"; done ;;
esac
