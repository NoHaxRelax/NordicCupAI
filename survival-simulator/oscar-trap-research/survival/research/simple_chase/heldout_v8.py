"""Eight held-out map/fixture pairs, fixed before results; do not tune mid-batch."""
from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).resolve().parents[2]
CASES=[(10000+i,9300+i) for i in range(1,9)]
rows=[]
for map_seed,fixture_seed in CASES:
 if Path('/tmp/predator-intake-stop').exists():break
 command=['systemd-run','--user','--scope','-p','MemoryMax=3G','-p','MemoryHigh=2G','-p','MemorySwapMax=0','--',sys.executable,
          str(ROOT/'research/simple_chase/run_streaming_v2.py'),'--policy','simple_chase.policy_v8_latency_spacing:SimpleChase',
          '--seconds','120','--map-seed',str(map_seed),'--fixture-seed',str(fixture_seed),'--station-bait','--native-width','320']
 completed=subprocess.run(command,cwd=ROOT)
 rows.append({'map_seed':map_seed,'fixture_seed':fixture_seed,'exit_code':completed.returncode})
 if completed.returncode:break
(ROOT/'results/simple_chase/heldout-v8-execution.json').write_text(json.dumps({'cases':CASES,'executed':rows},indent=2)+'\n')
