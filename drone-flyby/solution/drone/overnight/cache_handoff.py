"""Wait for the old classifier queue to finish its current run, then continue."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

work=Path(os.environ['NORDIC_WORK']);run=Path(os.environ['NORDIC_RUN_DIR'])
old=work/'runs/drone-night-v1';code=work/'code/overnight-v2'
while not (old/'result.json').exists():
    time.sleep(10)
receipt=json.loads((old/'result.json').read_text())
if receipt['exit_code']!=0:raise SystemExit('Old queue failed; inspect before continuation')
state=json.loads((old/'experiments/status.json').read_text())
if state.get('failed'):raise SystemExit('Old queue recorded failures; inspect before continuation')
remaining=[e for e in ['cls-partial','cls-full','cls-scratch'] if e not in state['completed']]
if not remaining:raise SystemExit('All classifier runs already completed')
print('Continuing with cached images:',remaining,flush=True)
subprocess.run([sys.executable,str(code/'train.py'),'--data',str(work/'data/smoke-v1'),
    '--weights',str(work/'weights'),'--output',str(run/'cache-smoke'),
    '--release',str(work/'data/smoke-v1/release.json'),'--experiment','cls-full','--host','runpod','--smoke'],check=True)
subprocess.run([sys.executable,str(code/'runpod_launch.py'),','.join(remaining)],check=True)
