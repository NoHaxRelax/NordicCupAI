"""Paired full native games, no paid infrastructure actions. Separate immutable output per job."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fastsim',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--seeds',type=int,nargs='+',default=[204871,917263,605319,148027,730951,392681])
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    script=Path(__file__).with_name('fast_entrapment_game.py')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def run(seed,size):
        folder=a.out/f'{seed}-nursery{size}'; folder.mkdir()
        cmd=[sys.executable,str(script),'--fastsim',str(a.fastsim),'--out',str(folder),
             '--seed',str(seed),'--seconds','3000','--nursery-size',str(size),'--summary-only']
        with (folder/'process.log').open('w') as log:
            try: code=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode
            except subprocess.TimeoutExpired: return dict(seed=seed,nursery=size,error='timeout')
        if code or not (folder/'summary.json').exists():return dict(seed=seed,nursery=size,error=f'exit {code}')
        s=json.loads((folder/'summary.json').read_text())
        children={e['agent'] for e in s['events'] if e['kind']=='nursery_child_observed'}
        arrivals={e['agent'] for e in s['events'] if e['kind']=='bait_arrived'}
        return dict(seed=seed,nursery=size,status=s['status'],score=s['score'],seconds=s['sim_time'],
                    gaps=s['estimated_bait_gap_seconds_after_first_arrival'],metrics=s['policy_metrics'],
                    native=s['native_evaluation'], nursery_outcomes=dict(
                        births_requested=sum(e['kind']=='nursery_birth_requested' for e in s['events']),
                        children_observed=len(children), children_arriving_as_bait=len(children & arrivals)))
    rows=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(run,seed,size) for seed in a.seeds for size in (0,2)]
        for future in as_completed(jobs):
            row=future.result();rows.append(row)
            temp=a.out/'results.tmp';temp.write_text(json.dumps(rows,indent=2));temp.replace(a.out/'results.json')
            print(json.dumps(row),flush=True)
    if any('error' in row or row.get('status')!='complete' for row in rows):sys.exit(1)


if __name__=='__main__':main()
