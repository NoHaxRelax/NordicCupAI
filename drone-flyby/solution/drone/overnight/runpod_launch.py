"""Run inside compute/gpu.py start; paths derive from its persistent workspace."""
import os
from pathlib import Path
import subprocess
import sys


work = Path(os.environ['NORDIC_WORK'])
run = Path(os.environ['NORDIC_RUN_DIR'])
code = Path(__file__).resolve().parent
data = work / 'data/overnight-runpod-20260917-v1'
shared = ['--data', str(data), '--weights', str(work/'weights'), '--release', str(code/'release.json')]
subprocess.run([sys.executable, str(code/'manage.py'), 'preflight', *shared,
                '--output', str(run/'preflight.json')], check=True)
experiments = {
    'a100-a': 'det-full,det-L0,det-L1,det-L2',
    'a100-b': 'det-partial,det-head,det-scratch',
    'a100-c': 'cls-frozen,cls-partial,cls-full,cls-scratch',
}[work.name]
if len(sys.argv) > 1:
    experiments = sys.argv[1]
subprocess.run([sys.executable, str(code/'manage.py'), 'queue', *shared,
                '--run', str(run/'experiments'), '--host', 'runpod',
                '--experiments', experiments], check=True)
