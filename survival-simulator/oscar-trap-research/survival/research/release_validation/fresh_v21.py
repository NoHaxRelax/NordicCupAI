"""Four frozen fresh pairs for integrated replaceable-site / surviving-guide release."""
from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).resolve().parents[2]
rows=[]
for map_seed in range(10144,10148):
 if Path('/tmp/predator-intake-stop').exists():break
 fixture_seed=map_seed+10000
 cmd=['systemd-run','--user','--scope','-p','MemoryMax=3G','-p','MemoryHigh=2G','-p','MemorySwapMax=0','--',sys.executable,str(ROOT/'research/release_validation/run.py'),'--policy','release_validation.policy_v21_release:ReleaseGuide','--seconds','300','--map-seed',str(map_seed),'--fixture-seed',str(fixture_seed),'--station-bait','--native-width','320']
 with (ROOT/f'results/release_validation/m{map_seed}-v21-fresh.log').open('w') as f:r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 rows.append(dict(map_seed=map_seed,fixture_seed=fixture_seed,returncode=r.returncode))
 (ROOT/'results/release_validation/v21-fresh-execution.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows[-1],flush=True)
