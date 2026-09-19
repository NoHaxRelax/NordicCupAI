"""Freeze current code and record complete native games for evaluation/replay."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=10)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=2026091901)
    parser.add_argument('--seconds',type=float,default=3000.)
    parser.add_argument('--timeout',type=float,default=18000.)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if min(args.count,args.workers,args.seconds,args.timeout)<=0:parser.error('Counts and time limits must be positive')
    if args.output.exists():parser.error('Choose a new folder to preserve recorded games')
    args.output.mkdir(parents=True)
    source=args.output/'source';hashes={}
    for directory in ('src','models','scripts'):
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and path.suffix in ('.py','.json','.html'):
                relative=path.relative_to(ROOT);target=source/relative
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
                hashes[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    seeds=random.Random(args.seed).sample(range(2**31),args.count)
    write(args.output/'manifest.json',dict(config=vars(args)|{'output':str(args.output)},seeds=seeds,
        source_hashes=hashes,python=platform.python_version(),
        packages={p:importlib.metadata.version(p) for p in ('numpy','pygame','pydantic','shapely','scipy')}))
    env=os.environ|{k:'1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}

    def run(job):
        index,seed=job;folder=args.output/f'game-{index:03}-{seed}'
        started=time.monotonic()
        cmd=[sys.executable,str(source/'scripts/entrapment_game.py'),'--seed',str(seed),'--seconds',str(args.seconds),'--out',str(folder)]
        timed_out=False
        with (args.output/f'game-{index:03}.log').open('w') as log:
            try:p=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
            except subprocess.TimeoutExpired:timed_out=True
        summary_path=folder/'summary.json'
        summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
        result={k:v for k,v in summary.items() if k not in ('history','events')}
        result.update(index=index,seed=seed,folder=folder.name,wall_seconds=round(time.monotonic()-started,2))
        if timed_out:result['status']='worker_timeout'
        elif p.returncode:result.update(status='worker_error',returncode=p.returncode)
        elif result.get('status')!='complete':result['status']='incomplete'
        write(args.output/f'game-{index:03}-result.json',result)
        return result

    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,j) for j in enumerate(seeds)]):
            result=future.result();results.append(result)
            write(args.output/'results.json',sorted(results,key=lambda r:r['index']))
            print(json.dumps(result),flush=True)


if __name__=='__main__':main()
