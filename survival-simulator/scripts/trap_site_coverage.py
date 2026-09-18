"""Screen candidate traps, then measure selected sites on independent encounters.

Two distinct outputs: held-out attempt success and fraction of random maps with
an empirically >50% site. Wilson lower bounds distinguish evidence from noise.
All native runs use 30 preloaded predators plus one full-start-energy guide.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def interval(k, n):
    if not n: return [0., 1.]
    z = 1.96
    p = k/n
    centre = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [centre-half, centre+half]


def inventory(seed):
    sys.path.insert(0, str(ROOT))
    import guide_lab as lab
    from models.entrapment.observed_trap_sites import available_sites, our_sites
    core = lab.SimulationCore(seed=seed, starting_agents=0, starting_predators=0)
    env = core.env
    static = dict(width=env.width, height=env.height,
                  obstacles=[(o.x,o.y,o.width,o.height) for o in env.obstacles])
    return dict(seed=seed, regular=available_sites(static), corners=our_sites(static,corner_only=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps',type=int,default=20)
    parser.add_argument('--seed',type=int,default=209182026)
    parser.add_argument('--sites',type=int,default=3)
    parser.add_argument('--corner-sites',type=int,default=2)
    parser.add_argument('--screen-attempts',type=int,default=6)
    parser.add_argument('--validation-attempts',type=int,default=20)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--timeout',type=float,default=300.)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inventory',type=int)
    args = parser.parse_args()
    if args.inventory is not None:
        write(args.output,inventory(args.inventory))
        return
    if min(args.maps,args.sites,args.screen_attempts,args.validation_attempts,args.workers)<1:
        parser.error('Counts must be positive')
    if args.output.exists(): parser.error('Use a new output directory to preserve frozen runs')
    args.output.mkdir(parents=True)
    snapshot = args.output/'source'
    sources = {str(p.relative_to(ROOT)):p for directory in ('src','models','scripts')
               for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.html')}
    for relative, source in sources.items():
        target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    rng=random.Random(args.seed)
    seeds=rng.sample(range(2**31),args.maps)
    manifest=dict(config=vars(args)|{'output':str(args.output),'multi':True},
                  map_seeds=seeds, source_hashes={p:hashlib.sha256(s.read_bytes()).hexdigest() for p,s in sources.items()})
    write(args.output/'manifest.json',manifest)
    env=os.environ|{k:'1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}
    inventories=[]
    for index,seed in enumerate(seeds):
        path=args.output/f'inventory-{index:04}.json'
        subprocess.run([sys.executable,str(snapshot/'scripts/trap_site_coverage.py'),
                        '--inventory',str(seed),'--output',str(path)],env=env,check=True,
                       stdout=subprocess.DEVNULL)
        inventories.append(json.loads(path.read_text()))
    jobs=[]; selected=[]
    for map_index,inv in enumerate(inventories):
        candidates=[]
        for corner,key,limit in ((False,'regular',args.sites),(True,'corners',args.corner_sites)):
            for site_index,site in enumerate(inv[key][:limit]):
                if any(math.dist(site['goal'],c['goal'])<.01 for c in candidates):continue
                candidates.append(dict(corner_only=corner,site_index=site_index,goal=site['goal'],
                                       site_kind=site.get('site_kind','crevice')))
        # Same encounter seeds across sites on a map; valid spawn acceptance can
        # differ with trap location. Independent seed stream for validation.
        encounters=[rng.randrange(2**31) for _ in range(args.screen_attempts)]
        validation=[rng.randrange(2**31) for _ in range(args.validation_attempts)]
        selected.append(dict(map_index=map_index,seed=inv['seed'],candidates=candidates,
                             validation_seeds=validation))
        for candidate in candidates:
            for encounter in encounters:
                jobs.append(dict(index=len(jobs),map_index=map_index,seed=inv['seed'],
                                 encounter_seed=encounter,phase='screen',**candidate))
    write(args.output/'selection.json',selected)
    results=[]
    started=time.monotonic()

    def execute(job):
        folder=args.output/f"case-{job['index']:04}";folder.mkdir()
        command=[sys.executable,str(snapshot/'scripts/guide_multi.py'),'--bulk','--deliveries','1',
                 '--seed',str(job['seed']),'--encounter-seed',str(job['encounter_seed']),
                 '--site',str(job['site_index']),'--output',str(folder)]
        if job['corner_only']:command.append('--corner-only')
        begin=time.monotonic()
        try:
            with (folder/'worker.log').open('w') as handle:
                process=subprocess.run(command,env=env,stdout=handle,stderr=subprocess.STDOUT,timeout=args.timeout)
            paths=list(folder.glob('multi-*/summary.json'))
            row=json.loads(paths[0].read_text()) if paths else dict(outcome='worker_error',returncode=process.returncode)
            if row['outcome']=='running':row['outcome']='worker_error'
        except subprocess.TimeoutExpired:row=dict(outcome='worker_timeout')
        row.update(job,wall_seconds=round(time.monotonic()-begin,3))
        write(folder/'result.json',row)
        return row

    def run_phase(tasks):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(execute,j) for j in tasks]):
                results.append(future.result())
                write(args.output/'results.json',sorted(results,key=lambda x:x['index']))
                if len(results)%10==0:
                    print(json.dumps(dict(completed=len(results),elapsed=round(time.monotonic()-started),phase=results[-1]['phase'])),flush=True)

    run_phase(jobs)
    validation_jobs=[]
    for row in selected:
        for candidate in row['candidates']:
            cases=[r for r in results if r['map_index']==row['map_index']
                   and r['site_index']==candidate['site_index'] and r['corner_only']==candidate['corner_only']]
            candidate['screen_passes']=sum(r['outcome']=='delivery_pass' for r in cases)
        if not row['candidates']:continue
        best=max(row['candidates'],key=lambda c:c['screen_passes'])
        row['selected']=best
        for encounter in row['validation_seeds']:
            validation_jobs.append(dict(index=len(jobs)+len(validation_jobs),map_index=row['map_index'],
                                        seed=row['seed'],encounter_seed=encounter,phase='validation',**best))
    write(args.output/'selection.json',selected)
    run_phase(validation_jobs)
    for row in selected:
        cases=[r for r in results if r['map_index']==row['map_index'] and r['phase']=='validation']
        n=len(cases); k=sum(r['outcome']=='delivery_pass' for r in cases)
        row['validation']=dict(successes=k,attempts=n,rate=k/n if n else None,
                               wilson_95=interval(k,n),empirical_above_half=n>0 and k/n>.5,
                               lower_bound_above_half=n>0 and interval(k,n)[0]>.5)
    valid=[r for r in results if r['phase']=='validation']
    report=dict(maps=args.maps,maps_with_candidate=sum(bool(r['candidates']) for r in selected),
                held_out_attempts=len(valid),held_out_successes=sum(r['outcome']=='delivery_pass' for r in valid),
                maps_empirically_above_half=sum(r['validation']['empirical_above_half'] for r in selected),
                maps_lower_bound_above_half=sum(r['validation']['lower_bound_above_half'] for r in selected),
                by_map=selected,note='Candidate search capped; failure to find a good site does not prove none exists. Selection uses screen attempts; metrics use independent validation encounters. Per-site intervals are not simultaneous confidence bounds.')
    write(args.output/'coverage.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='by_map'},indent=2),flush=True)


if __name__=='__main__':main()
