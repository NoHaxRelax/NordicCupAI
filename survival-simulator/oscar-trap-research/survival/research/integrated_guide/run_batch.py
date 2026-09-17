"""Freeze a development batch before launching bounded native workers."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research'),str(ROOT/'research/reliability_eval')]
from reliability_eval.run_frozen import policy_dependency_manifest

def main(policy, first_map, count, label, workers=2):
    out=ROOT/'results/integrated_guide'/label
    out.mkdir(parents=True,exist_ok=True)
    plan_path=out/'PLAN.json'
    if plan_path.exists():raise SystemExit('immutable development batch already exists')
    frozen=policy_dependency_manifest(policy.split(':')[0])
    plan=dict(policy=policy,cases=[dict(map_seed=m,fixture_seed=m+10000) for m in range(first_map,first_map+count)],
              seconds=300,workers=workers,dependencies=frozen,scope='fresh development batch; separate policy and fixture distribution from all prior trials',
              runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    def one(case):
        if Path('/tmp/predator-intake-stop').exists():return dict(**case,status='not_started_stop')
        if policy_dependency_manifest(policy.split(':')[0])!=frozen:raise RuntimeError('frozen policy dependencies changed')
        before=set((ROOT/'results/simple_chase').glob(f"*-m{case['map_seed']}-f{case['fixture_seed']}-*.json"))
        cmd=['systemd-run','--user','--scope','-p','MemoryMax=3G','-p','MemoryHigh=2G','-p','MemorySwapMax=0','--',sys.executable,
             str(ROOT/'research/simple_chase/run_streaming_v2.py'),'--policy',policy,'--seconds','300',
             '--map-seed',str(case['map_seed']),'--fixture-seed',str(case['fixture_seed']),'--station-bait','--native-width','320']
        with (out/f"m{case['map_seed']}.log").open('x') as log:
            result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        after=set((ROOT/'results/simple_chase').glob(f"*-m{case['map_seed']}-f{case['fixture_seed']}-*.json"))
        receipts=[]
        for path in after-before:
            data=json.loads(path.read_text())
            if data.get('policy')==policy:receipts.append(str(path.relative_to(ROOT)))
        row=dict(**case,returncode=result.returncode,receipts=sorted(receipts),
                 dependencies_unchanged=policy_dependency_manifest(policy.split(':')[0])==frozen)
        (out/f"m{case['map_seed']}.execution.json").write_text(json.dumps(row,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        return row
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows=list(pool.map(one,plan['cases']))
    (out/'execution.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',required=True);p.add_argument('--first-map',type=int,required=True)
    p.add_argument('--count',type=int,required=True);p.add_argument('--label',required=True)
    p.add_argument('--workers',type=int,default=2)
    main(**vars(p.parse_args()))
