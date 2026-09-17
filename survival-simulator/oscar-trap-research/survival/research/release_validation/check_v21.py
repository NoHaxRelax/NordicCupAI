"""Four predeclared fitted regressions for the integrated replaceable/release candidate."""
from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).resolve().parents[2]
rows=[]
for map_seed,fixture_seed in [(10028,9528),(10031,9531),(10048,9748),(10038,9638)]:
 if Path('/tmp/predator-intake-stop').exists():break
 command=['systemd-run','--user','--scope','-p','MemoryMax=3G','-p','MemoryHigh=2G','-p','MemorySwapMax=0','--',sys.executable,str(ROOT/'research/release_validation/run.py'),'--policy','release_validation.policy_v21_release:ReleaseGuide','--seconds','300','--map-seed',str(map_seed),'--fixture-seed',str(fixture_seed),'--station-bait','--native-width','320']
 with (ROOT/f'results/release_validation/m{map_seed}-v21.log').open('w') as log:r=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 rows.append(dict(map_seed=map_seed,fixture_seed=fixture_seed,returncode=r.returncode));print(rows[-1],flush=True)
 (ROOT/'results/release_validation/v21-execution.json').write_text(json.dumps(rows,indent=2)+'\n')
