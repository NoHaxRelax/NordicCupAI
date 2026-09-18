"""Frozen v13 development regressions; these are not held-out reliability cases."""
from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).resolve().parents[2]
CASES=[(10012,9412),(10014,9414),(10015,9415),(10016,9416)]
rows=[]
for map_seed,fixture_seed in CASES:
 if Path('/tmp/predator-intake-stop').exists():break
 command=['systemd-run','--user','--scope','-p','MemoryMax=3G','-p','MemoryHigh=2G','-p','MemorySwapMax=0','--',sys.executable,
          str(ROOT/'research/simple_chase/run_streaming_v2.py'),'--policy','simple_chase.policy_v13_close_reacquire:SimpleChase',
          '--seconds','180','--map-seed',str(map_seed),'--fixture-seed',str(fixture_seed),'--station-bait','--native-width','320']
 completed=subprocess.run(command,cwd=ROOT)
 rows.append({'map_seed':map_seed,'fixture_seed':fixture_seed,'exit_code':completed.returncode})
 (ROOT/'results/simple_chase/probe-v13-execution.json').write_text(json.dumps({'cases':CASES,'executed':rows},indent=2)+'\n')
 if completed.returncode:break
