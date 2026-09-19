"""Paired full native games for explicit colony/coordinator settings.

Each subprocess writes its exact source manifest and native outcome. Timeouts
are infrastructure failures, never silently interpreted as colony extinction.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--fastsim',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',required=True)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--timeout',type=float,default=1800.)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    specification=json.loads(args.configs.read_text())
    recipes=specification['variants']
    def run(seed,name,recipe):
        folder=args.out/f'{seed}-{name}'
        folder.mkdir(exist_ok=True)
        config=folder/'survival-settings.json'
        config.write_text(json.dumps(recipe['survival']))
        command=[sys.executable,str(ROOT/'scripts/fast_entrapment_game.py'),
                 '--fastsim',str(args.fastsim.resolve()),'--seed',str(seed),
                 '--seconds','3000','--out',str(folder),'--summary-only',
                 '--survival-config',str(config),*specification.get('common_args',[]),*recipe.get('args',[])]
        if (folder/'manifest.json').exists():
            return dict(seed=seed,variant=name,status='refused_existing_run')
        try:
            with (folder/'process.log').open('w') as log:
                completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},timeout=args.timeout)
            if completed.returncode:
                return dict(seed=seed,variant=name,status='process_error',returncode=completed.returncode)
        except subprocess.TimeoutExpired:
            return dict(seed=seed,variant=name,status='infrastructure_timeout')
        summary=json.loads((folder/'summary.json').read_text())
        return dict(seed=seed,variant=name,**{k:v for k,v in summary.items() if k not in ('events','history','seed')})
    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,seed,name,recipe) for seed in args.seeds for name,recipe in recipes.items()]
        for future in as_completed(futures):
            row=future.result();rows.append(row)
            temp=args.out/'results.tmp';temp.write_text(json.dumps(rows,indent=2));temp.replace(args.out/'results.json')
            print(json.dumps({k:row.get(k) for k in ('seed','variant','status','score','sim_time')}),flush=True)


if __name__=='__main__': main()
