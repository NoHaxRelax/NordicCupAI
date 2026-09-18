"""Frozen 16-case V27 EmergencyGuides batch with capped native workers."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,subprocess,sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research'),str(ROOT/'research/reliability_eval')]
from reliability_eval.run_frozen import policy_dependency_manifest

POLICY='integrated_guide.policy_v27:EmergencyGuides'
OUT=ROOT/'results/short_overlap_sol/v27_emergency_fresh16'
CASES=[dict(map_seed=m,fixture_seed=m+10000) for m in range(10320,10336)]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=policy_dependency_manifest('integrated_guide.policy_v27')
    plan=dict(policy=POLICY,cases=CASES,seconds=300,workers=2,dependencies=frozen,
        scope='predeclared fresh validation; all unsupported/errors count',
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    plan_path=OUT/'PLAN.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise RuntimeError('existing plan differs')
    else:plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    def one(case):
        if policy_dependency_manifest('integrated_guide.policy_v27')!=frozen:
            raise RuntimeError('frozen dependencies changed')
        before=set((ROOT/'results/simple_chase').glob(f"*-m{case['map_seed']}-f{case['fixture_seed']}-*.json"))
        cmd=['systemd-run','--user','--scope','-p','MemoryMax=3G','-p','MemoryHigh=2G','-p','MemorySwapMax=0','--',sys.executable,
            str(ROOT/'research/simple_chase/run_streaming_v2.py'),'--policy',POLICY,'--seconds','300',
            '--map-seed',str(case['map_seed']),'--fixture-seed',str(case['fixture_seed']),'--station-bait','--native-width','320']
        log_path=OUT/f"m{case['map_seed']}.log"
        with log_path.open('x') as log:result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        after=set((ROOT/'results/simple_chase').glob(f"*-m{case['map_seed']}-f{case['fixture_seed']}-*.json"))
        receipts=[]
        for path in after-before:
            try:data=json.loads(path.read_text())
            except Exception:continue
            if data.get('policy')==POLICY:receipts.append(str(path.relative_to(ROOT)))
        row=dict(**case,returncode=result.returncode,receipts=sorted(receipts),
            dependencies_unchanged=policy_dependency_manifest('integrated_guide.policy_v27')==frozen)
        (OUT/f"m{case['map_seed']}.execution.json").write_text(json.dumps(row,indent=2)+'\n')
        print(json.dumps(row),flush=True);return row
    with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(one,CASES))
    (OUT/'execution.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
