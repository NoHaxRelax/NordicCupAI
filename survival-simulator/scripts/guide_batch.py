"""Parallel native-map evaluation with deterministic random seeds and checkpoints.

python scripts/guide_batch.py --maps 1000 --workers 32 --shards 2 --shard 0
Run the other shard with --shard 1 and identical --seed. No policy changes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--seed', type=int, default=9182026)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--timeout', type=float, default=180.)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--multi', action='store_true', help='30 preloaded predators plus one new delivery; rear entrance must stay clear')
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or min(args.maps, args.workers) < 1:
        parser.error('Invalid shard or worker/map count')
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    seeds = rng.sample(range(2**31), args.maps)
    jobs = [dict(index=i, seed=s, encounter_seed=rng.randrange(2**31)) for i,s in enumerate(seeds)]
    sources = sorted({p for directory in ('src', 'models', 'scripts')
                      for p in (ROOT/directory).rglob('*')
                      if p.is_file() and p.suffix in ('.py', '.json', '.html')})
    manifest = dict(config=vars(args) | {'output': str(args.output)}, jobs=jobs,
                    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    manifest_path = args.output/'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if (previous['source_hashes'] != manifest['source_hashes'] or previous['jobs'] != jobs
                or any(previous['config'].get(k) != manifest['config'].get(k)
                       for k in ('multi', 'seconds', 'timeout'))):
            parser.error('Existing batch has different code, seeds, or settings; use a new output directory.')
    source_root = args.output/'source'
    for source in sources:
        target = source_root/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    env = os.environ | {key:'1' for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}

    def execute(job):
        folder = args.output/f"case-{job['index']:04}"
        folder.mkdir(exist_ok=True)
        result_file = folder/'result.json'
        if result_file.exists():
            return json.loads(result_file.read_text())
        command = [sys.executable, str(source_root/'scripts/guide_lab.py'), '--bulk',
                   '--seed', str(job['seed']), '--encounter-seed', str(job['encounter_seed']),
                   '--seconds', str(args.seconds), '--output', str(folder)]
        if args.multi:
            command = [sys.executable,str(source_root/'scripts/guide_multi.py'),'--bulk','--deliveries','1',
                       '--seed',str(job['seed']),'--encounter-seed',str(job['encounter_seed']),'--output',str(folder)]
        started = time.monotonic()
        try:
            with (folder/'worker.log').open('w') as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                           env=env, timeout=args.timeout)
            paths = list(folder.glob(('multi-*' if args.multi else 'map-*')+'/summary.json'))
            result = json.loads(paths[0].read_text()) if paths else dict(outcome='worker_error', returncode=completed.returncode)
            if result['outcome']=='running':
                result['outcome']='worker_error'
            if args.multi and not paths and 'No usable site at index 0; map has 0 eligible sites' in (folder/'worker.log').read_text():
                result['outcome']='no_usable_bait_site'
            if result['outcome']=='setup_error' and 'No usable site at index 0; map has 0 eligible sites' in result.get('error',''):
                result['outcome']='no_usable_bait_site'
        except subprocess.TimeoutExpired:
            result = dict(outcome='worker_timeout')
        result.update(job, wall_seconds=round(time.monotonic()-started,3))
        result_file.write_text(json.dumps(result, indent=2))
        return result

    selected = [j for j in jobs if j['index'] % args.shards == args.shard]
    started = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        with (args.output/'results.jsonl').open('w') as log:
            for future in as_completed([pool.submit(execute,j) for j in selected]):
                result = future.result()
                results.append(result)
                log.write(json.dumps(result)+'\n')
                log.flush()
                if len(results)%10==0 or len(results)==len(selected):
                    outcomes = {r['outcome']:sum(x['outcome']==r['outcome'] for x in results) for r in results}
                    print(json.dumps(dict(completed=len(results), total=len(selected),
                                          elapsed=round(time.monotonic()-started,1), outcomes=outcomes)), flush=True)
    (args.output/'results.json').write_text(json.dumps(sorted(results,key=lambda x:x['index']),indent=2))


if __name__ == '__main__':
    main()
