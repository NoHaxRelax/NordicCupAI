"""Frozen v10 development regressions; these are not held-out reliability cases."""
from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).resolve().parents[2]
CASES=[(10004,9304),(10002,9302),(10001,9301),(10003,9303),(10005,9305),(10006,9306)]
rows=[]
for map_seed,fixture_seed in CASES:
 if Path('/tmp/predator-intake-stop').exists():break
 command=['systemd-run','--user','--scope','-p','MemoryMax=3G','-p','MemoryHigh=2G','-p','MemorySwapMax=0','--',sys.executable,
          str(ROOT/'research/simple_chase/run_streaming_v2.py'),'--policy','simple_chase.policy_v10_approach_lane:SimpleChase',
          '--seconds','180','--map-seed',str(map_seed),'--fixture-seed',str(fixture_seed),'--station-bait','--native-width','320']
 completed=subprocess.run(command,cwd=ROOT)
 rows.append({'map_seed':map_seed,'fixture_seed':fixture_seed,'exit_code':completed.returncode})
 (ROOT/'results/simple_chase/probe-v10-execution.json').write_text(json.dumps({'cases':CASES,'executed':rows},indent=2)+'\n')
 if completed.returncode:break
