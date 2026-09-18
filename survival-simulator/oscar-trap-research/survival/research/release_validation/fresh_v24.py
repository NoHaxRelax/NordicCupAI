"""Frozen fresh sixteen; development sample, separate from reserved59 protocol."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'research'), str(ROOT/'research/reliability_eval')]
from reliability_eval.run_frozen import policy_dependency_manifest
POLICY = 'release_validation.policy_v24_release:ReleaseGuide'
OUT = ROOT/'results/release_validation/v24_fresh16'
OUT.mkdir(parents=True, exist_ok=True)
frozen = policy_dependency_manifest(POLICY.split(':')[0])
plan = dict(policy=POLICY, cases=[dict(map_seed=m, fixture_seed=m+10000) for m in range(10160,10176)],
            seconds=300, policy_dependencies=frozen, limitations='fresh development16; never pooled with fitted repairs or prior policy versions')
plan_path = OUT/'PLAN.json'
if plan_path.exists():
    raise SystemExit('immutable batch already exists')
plan_path.write_text(json.dumps(plan, indent=2)+'\n')
def one(case):
    if Path('/tmp/predator-intake-stop').exists(): return dict(**case, status='not_started_stop')
    if policy_dependency_manifest(POLICY.split(':')[0]) != frozen: raise RuntimeError('frozen policy changed')
    cmd=['systemd-run','--user','--scope','-p','MemoryMax=3G','-p','MemoryHigh=2G','-p','MemorySwapMax=0','--',sys.executable,
         str(ROOT/'research/release_validation/run.py'),'--policy',POLICY,'--seconds','300','--map-seed',str(case['map_seed']),
         '--fixture-seed',str(case['fixture_seed']),'--station-bait','--native-width','320']
    with (OUT/f"m{case['map_seed']}.log").open('x') as f:
        r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    row=dict(**case,returncode=r.returncode,dependencies_unchanged=policy_dependency_manifest(POLICY.split(':')[0])==frozen)
    (OUT/f"m{case['map_seed']}.execution.json").write_text(json.dumps(row,indent=2)+'\n')
    print(row,flush=True)
    return row
with ThreadPoolExecutor(max_workers=2) as pool:
    rows=list(pool.map(one,plan['cases']))
(OUT/'execution.json').write_text(json.dumps(rows,indent=2)+'\n')
