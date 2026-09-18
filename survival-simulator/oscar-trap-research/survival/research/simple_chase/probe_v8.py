"""Predeclared sequential probes; each child has an OS-enforced memory cap."""
from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).resolve().parents[2]
CASES=[(10000,9204),(5101,9203),(5101,9201)]
rows=[]
for map_seed,fixture_seed in CASES:
 if Path('/tmp/predator-intake-stop').exists():break
 command=['systemd-run','--user','--scope','-p','MemoryMax=3G','-p','MemoryHigh=2G','-p','MemorySwapMax=0','--',sys.executable,
          str(ROOT/'research/simple_chase/run_streaming_v2.py'),'--policy','simple_chase.policy_v8_latency_spacing:SimpleChase',
          '--seconds','120','--map-seed',str(map_seed),'--fixture-seed',str(fixture_seed),'--station-bait','--native-width','400']
 completed=subprocess.run(command,cwd=ROOT)
 rows.append({'map_seed':map_seed,'fixture_seed':fixture_seed,'exit_code':completed.returncode})
 if completed.returncode:break
(ROOT/'results/simple_chase/probe-v8-execution.json').write_text(json.dumps({'cases':CASES,'executed':rows},indent=2)+'\n')
