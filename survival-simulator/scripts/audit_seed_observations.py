"""Natural-map observation audit, NOT a seed-recovery success-rate benchmark.

Tests the real seed only as an evaluator assertion that public constraints retain
it. An independent fixed search prefix estimates false candidates. Hidden geometry
is used solely to assess extraction error, never to construct constraints.
"""
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,concurrent.futures,json,pathlib,sys,time,random,subprocess,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def run(job):
    seed,out,ticks,prefix,full=job
    from fastsim.fastpolicy import PolicySimulationCore
    from src.utils.DTOs import ActionRequest
    from models.seed_shadow.public_terrain import TerrainSamples
    from models.seed_shadow.public_geometry import PublicGeometry
    path=pathlib.Path(out)/str(seed);path.mkdir(parents=True,exist_ok=True)
    if (path/'result.json').exists():return json.loads((path/'result.json').read_text())
    start=time.monotonic();sim=PolicySimulationCore(seed=seed)
    config=json.loads((ROOT/'docs/localfood20/pod-7/expanded_local_food-winner.json').read_text())['config']
    sim.policy_init(0,config);terrain=TerrainSamples();geometry=PublicGeometry()
    error=0.;checks=0;exception=None
    try:
        for tick in range(ticks):
            actions=[(aid,ActionRequest(agent_id=aid,move_distance=d,move_direction=v,turn_angle=t,spawn_agent=z)) for aid,d,v,t,z in sim.policy_act()]
            body=sim.step(actions);terrain.observe(body);geometry.observe(body,terrain)
            truth={a.agent_id:a for a in sim.env.agents}
            for aid,(x,y,h) in terrain.last_poses.items():
                if aid in truth:
                    error=max(error,abs(x-truth[aid].x),abs(y-truth[aid].y));checks+=1
            if not body['num_agents']:break
    except Exception:
        exception=traceback.format_exc()
    collection_seconds=time.monotonic()-start
    samples=path/'samples.txt'
    samples.write_text(''.join(f'{x} {y} {label}\n' for x,y,label in terrain.rows()))
    (path/'geometry.json').write_text(json.dumps(geometry.evidence()))
    scanner=str(ROOT/'models/seed_shadow/build/scan')
    retained=False;false_candidates=None;scan_seconds=0.
    if terrain.points:
        test=subprocess.run([scanner,str(samples),str(seed),str(seed+1)],capture_output=True,text=True)
        retained=str(seed) in test.stdout.split()
        if test.returncode:
            exception=(exception or '')+f' scanner failed: {test.returncode}: {test.stderr}'
        else:
            test=subprocess.run([scanner,str(samples),'0',str(prefix)],capture_output=True,text=True,check=True)
            false_candidates=sum(int(s)!=seed for s in test.stdout.split())
            scan_seconds=json.loads(test.stderr)['seconds']
    full_seconds=None;score=None;duration=None
    if full and not exception:
        sim.run_policy(3000)
        full_seconds=time.monotonic()-start-scan_seconds
        score=sim.env.score;duration=sim.env.time
    result=dict(seed=seed,frames=tick+1,samples=len(terrain.points),pose_checks=checks,
        max_pose_error=error,true_seed_retained=retained,false_candidates_in_prefix=false_candidates,
        prefix_size=prefix,collection_seconds=collection_seconds,scan_seconds=scan_seconds,
        observed_rectangles=len(geometry.rectangles()),observed_fruit_positions=len(geometry.fruit_positions),
        full_game_seconds=full_seconds,baseline_score=score,duration=duration,exception=exception,
        recovery_test=False)
    (path/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True)
    p.add_argument('--maps',type=int,default=1000);p.add_argument('--workers',type=int,default=32)
    p.add_argument('--ticks',type=int,default=1800);p.add_argument('--prefix',type=int,default=65536)
    p.add_argument('--full-game',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    seeds=random.Random(2026092007).sample(range(2**32),a.maps)
    (a.out/'manifest.json').write_text(json.dumps(dict(seeds=seeds,ticks=a.ticks,prefix=a.prefix,full_game=a.full_game),indent=2))
    jobs=[(s,str(a.out.resolve()),a.ticks,a.prefix,a.full_game) for s in seeds]
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,j) for j in jobs]
        for future in concurrent.futures.as_completed(futures):
            result=future.result();results.append(result)
            status=dict(done=len(results),total=len(seeds),retained=sum(r['true_seed_retained'] for r in results),
                bad_pose_maps=sum(r['max_pose_error']>1e-7 for r in results),failures=sum(r['exception'] is not None for r in results))
            (a.out/'progress.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
    (a.out/'results.json').write_text(json.dumps(sorted(results,key=lambda r:r['seed']),indent=2))
