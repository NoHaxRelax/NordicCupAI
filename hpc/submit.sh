#!/bin/bash
# Submit one LSF job per seed with the seed written LITERALLY into the script.
# The first cohort used an array-style ${LSB_JOBINDEX} default that did not
# survive shell escaping, so five jobs all ran seed 0 into one directory.
#
#   hpc/submit.sh <tag> <seed> [seed ...]        e.g.  hpc/submit.sh v3 0 1 2 3 4
#
# Run ON the cluster from $BLACKHOLE/nordic (or via: ssh dtu 'bash -lc "cd \$BLACKHOLE/nordic && hpc/submit.sh v3 0 1 2"').
set -euo pipefail
TAG=${1:?tag}; shift
ROOT=/dtu/blackhole/1e/205502/nordic
STEPS=${STEPS:-8000000}
ENVS=${ENVS:-64}
WALL=${WALL:-10:00}
mkdir -p "$ROOT/logs" "$ROOT/runs" "$ROOT/hpc/generated"

for SEED in "$@"; do
  RUN="hpc_${TAG}_s${SEED}"
  F="$ROOT/hpc/generated/${RUN}.lsf"
  cat > "$F" <<EOF
#!/bin/bash
#BSUB -J ${RUN}
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W ${WALL}
#BSUB -o ${ROOT}/logs/${RUN}.%J.out
#BSUB -e ${ROOT}/logs/${RUN}.%J.err
set -euo pipefail
cd ${ROOT}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
RESUME=""
[ -f runs/${RUN}/last.pt ] && RESUME="--resume runs/${RUN}/last.pt"
echo "run=${RUN} seed=${SEED} host=\$(hostname) resume=\${RESUME:-none} start=\$(date -Is)"
exec /dtu/blackhole/1e/205502/venv-nordic/bin/python -m survivalsim.ppo \\
  --run ${RUN} --seed ${SEED} --threads 4 \\
  --total-steps ${STEPS} --envs ${ENVS} --horizon 128 \\
  --eval-every 25 --eval-episodes 12 --out runs \$RESUME
EOF
  bsub < "$F" 2>/dev/null | tail -1 || echo "submit failed for $RUN"
done
sleep 2
bjobs -w 2>/dev/null | grep "hpc_${TAG}_" || true
