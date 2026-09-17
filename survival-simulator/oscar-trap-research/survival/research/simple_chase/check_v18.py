"""Four fitted repairs plus four fresh map/fixture pairs for frozen v18. Never pool the groups."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import subprocess,sys,json
ROOT=Path(__file__).resolve().parents[2]
CASES=[(10032,9632),(10038,9638),(10040,9640),(10045,9645)]+[(10000+i,9700+i) for i in range(48,52)]
OUT=ROOT/'results/simple_chase/heldout-v18-execution.json'
rows=[]
def run(case):
 if Path('/tmp/predator-intake-stop').exists():return {'map_seed':case[0],'fixture_seed':case[1],'not_started':'stop sentinel'}
 command=['systemd-run','--user','--scope','-p','MemoryMax=3G','-p','MemoryHigh=2G','-p','MemorySwapMax=0','--',sys.executable,
          str(ROOT/'research/simple_chase/run_streaming_v2.py'),'--policy','simple_chase.policy_v18_escape_lookahead:SimpleChase',
          '--seconds','180','--map-seed',str(case[0]),'--fixture-seed',str(case[1]),'--station-bait','--native-width','320']
 with (ROOT/f'results/simple_chase/heldout-v18-m{case[0]}.log').open('w') as log:
  result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 return {'map_seed':case[0],'fixture_seed':case[1],'exit_code':result.returncode}
OUT.write_text(json.dumps({'policy':'v18 frozen','cases':CASES,'executed':rows},indent=2)+'\n')
with ThreadPoolExecutor(max_workers=4) as pool:
 for future in as_completed([pool.submit(run,c) for c in CASES]):
  row=future.result();rows.append(row);print(json.dumps(row),flush=True)
  OUT.write_text(json.dumps({'policy':'v18 frozen','cases':CASES,'executed':rows},indent=2)+'\n')
