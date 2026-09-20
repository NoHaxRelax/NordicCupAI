"""Offline bounded-domain recovery test. The bot never receives the test seed.

This is a plumbing test, NOT evidence of feasible full uint32 live recovery.
Hidden state is read only by the harness to audit recovered poses/world accuracy.
"""
import os
os.environ['SDL_VIDEODRIVER']='dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
import argparse,json,pathlib,sys,time,subprocess,gzip
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from fastsim import SimulationCore as NativeCore
from fastsim.fastpolicy import PolicySimulationCore
from src.core import SimulationCore as PythonCore
from src.utils.DTOs import ActionRequest
from models.seed_shadow.replay import ShadowJournal,close
from models.seed_shadow.public_terrain import frame_poses
from check_seed_replay import world

p=argparse.ArgumentParser()
p.add_argument('--seed',type=int,default=12345)
p.add_argument('--end',type=int,default=65536)
p.add_argument('--ticks',type=int,default=500)
p.add_argument('--python',action='store_true')
p.add_argument('--collect-only',action='store_true')
p.add_argument('--out',type=pathlib.Path,required=True)
args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
policy=PolicySimulationCore(seed=args.seed)
cfg=json.loads((ROOT/'docs/localfood20/pod-7/expanded_local_food-winner.json').read_text())['config']
policy.policy_init(0,cfg)
real=PythonCore(seed=args.seed) if args.python else policy
journal=ShadowJournal()
pose_checks=0;max_pose_error=0.;start=time.monotonic()
for tick in range(args.ticks):
    acts=[(aid,ActionRequest(agent_id=aid,move_distance=d,move_direction=v,turn_angle=t,spawn_agent=z)) for aid,d,v,t,z in policy.policy_act()]
    response=real.step(acts)
    if real is not policy:policy.step(acts)
    journal.record(acts,response)
    truth={a.agent_id:a for a in real.env.agents}
    for aid,(x,y,h) in journal.terrain.last_poses.items():
        if aid in truth:
            pose_checks+=1
            max_pose_error=max(max_pose_error,abs(x-truth[aid].x),abs(y-truth[aid].y))
    if not real.env.agents:break
samples=args.out/'samples.txt'
samples.write_text(''.join(f'{x} {y} {label}\n' for x,y,label in journal.terrain.rows()))
with gzip.open(args.out/'public-journal.jsonl.gz','wt') as f:
    for frame in journal.frames:
        f.write(json.dumps(frame,separators=(',',':'))+'\n')
if args.collect_only:
    result=dict(source_engine='unmodified_python' if args.python else 'native',frames=len(journal.frames),samples=len(journal.terrain.points),pose_checks=pose_checks,max_pose_error=max_pose_error)
    (args.out/'collection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    raise SystemExit
scan=subprocess.run([str(ROOT/'models/seed_shadow/build/scan'),str(samples),'0',str(args.end)],capture_output=True,text=True,check=True)
candidates=list(map(int,scan.stdout.split()))
(args.out/'candidates.json').write_text(json.dumps(candidates))
scan_stats=json.loads(scan.stderr)
replay_start=time.monotonic()
recovered=journal.recover(candidates,NativeCore,complete_search=True)
result=dict(test_seed=args.seed,source_engine='unmodified_python' if args.python else 'native',
    domain=[0,args.end],full_uint32_search=args.end==2**32,frames=len(journal.frames),
    samples=len(journal.terrain.points),pose_checks=pose_checks,max_pose_error=max_pose_error,
    scanner=scan_stats,recovered_seed=recovered,status=journal.status,
    recovery_correct=recovered==args.seed,real_seed_in_candidates=args.seed in candidates,
    replay_seconds=time.monotonic()-replay_start,total_seconds=time.monotonic()-start,
    full_world_within_tolerance=close(world(real),world(journal.shadow)) if recovered is not None else None)
(args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
