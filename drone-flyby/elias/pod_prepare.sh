#!/bin/bash
# elias/pod_prepare.sh HOST SSH_PORT PUBLIC_PORT: code, dependencies and the three released checkpoints onto a fresh
# serving pod, checksums verified, address written to elias/out/logs/pod3.env. About two minutes.
set -uo pipefail
H=$1; P=$2; PUB=$3
cd "$(dirname "$0")/.." || exit 1
L=elias/out/logs/pod_prepare.log; mkdir -p elias/out/logs
OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=25 -o LogLevel=ERROR -i $HOME/.ssh/id_ed25519"
echo "=== $(date -Is) prepare $H:$P (public $PUB)" >> "$L"
bash elias/pod_serve_setup.sh $H $P >> "$L" 2>&1
scp -q $OPTS -P $P elias/release/both_m1280.pt root@$H:/root/out/F3_both_m1280.last.pt >> "$L" 2>&1
scp -q $OPTS -P $P elias/release/F5_fixed_m1280.pt root@$H:/root/out/F5_fixed_m1280.last.pt >> "$L" 2>&1
scp -q $OPTS -P $P elias/release/F3HN_m1280.pt root@$H:/root/out/F3HN_m1280.pt >> "$L" 2>&1
ssh $OPTS -p $P root@$H 'md5sum /root/out/*.pt; nvidia-smi --query-gpu=name --format=csv,noheader' >> "$L" 2>&1
if grep -q 8a7c74ade8d41d277fd1e1064ca78d32 "$L" && grep -q 34deb003fd650a290a7c8cd572310c94 "$L" && grep -q 62bb19c09ba41c68d9b98185eeb5a516 "$L"; then
  printf 'POD_HOST=%s\nPOD_PORT=%s\nPUBLIC_PORT=%s\nDIRECT_URL=http://%s:%s\n' $H $P $PUB $H $PUB > elias/out/logs/pod3.env
  echo "=== $(date -Is) POD_READY" >> "$L"; echo POD_READY
else
  echo "=== $(date -Is) POD_PREPARE_FAILED (checksums missing)" >> "$L"; echo POD_PREPARE_FAILED; tail -n 8 "$L"
fi
