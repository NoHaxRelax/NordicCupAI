from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import subprocess,sys,json
ROOT=Path(__file__).resolve().parents[2]
def run(seed):
 if Path('/tmp/predator-intake-stop').exists():return dict(map_seed=seed,not_started=True)
 cmd=['systemd-run','--user','--scope','-p','MemoryMax=3G','-p','MemoryHigh=2G','-p','MemorySwapMax=0','--',sys.executable,str(ROOT/'research/terrain_guide/run.py'),'--policy','terrain_guide.policy_v25_release:ReleaseGuide','--seconds','300','--map-seed',str(seed),'--fixture-seed',str(seed+10000),'--station-bait','--native-width','320']
 with (ROOT/f'results/terrain_guide/m{seed}-v25.log').open('w') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 return dict(map_seed=seed,fixture_seed=seed+10000,returncode=r.returncode)
rows=[]
with ThreadPoolExecutor(max_workers=2) as pool:
 for f in as_completed([pool.submit(run,s) for s in [10138,10139]]):
  rows.append(f.result());print(rows[-1],flush=True)
  (ROOT/'results/terrain_guide/v25-execution.json').write_text(json.dumps(rows,indent=2)+'\n')
