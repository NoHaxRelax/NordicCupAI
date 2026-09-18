"""Predeclared held-out perturbations, identical worker policies and source RNG seeds.
Policies change source RNG consumption, so these are paired seeds, not guaranteed
identical future fruit streams. Run after renewal_screen; no tuning on these cases.
"""
from run import run,save
import argparse,json,math
BASE=dict(relay=True,forage=True,newborn_ready=140,switch_energy=120,dwell=10,renew=True)
CASES=[dict(seed=101,heading=-.7,gap=45,reserve_angle=math.pi,reserve_gap=70),
       dict(seed=102,heading=.4,gap=80,reserve_angle=2.2,reserve_gap=100),
       dict(seed=103,heading=1.8,gap=55,reserve_angle=-1.4,reserve_gap=80),
       dict(seed=104,heading=math.pi,gap=90,reserve_angle=math.pi/2,reserve_gap=65),
       dict(seed=105,heading=-2.2,gap=110,reserve_angle=.8,reserve_gap=120),
       dict(seed=106,heading=-1.3,gap=50,reserve_angle=2.8,reserve_gap=150)]
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['heldout','workers','multiple','replays'],default='heldout');args=p.parse_args()
if args.phase=='heldout':
    cases=[dict(**scenario,energy=energy,policy=policy,food='trees',workers=2,worker_near=True,seconds=120,**BASE) for scenario in CASES for energy in (150,500) for policy in ('none','orbit_single','orbit')]
elif args.phase=='workers':
    cases=[dict(seed=seed,energy=150,policy=policy,food='trees',workers=4-(2 if policy=='orbit' else 1 if policy=='orbit_single' else 0),worker_near=True,seconds=120,**BASE) for seed in (201,202,203) for policy in ('none','orbit_single','orbit')]
elif args.phase=='multiple':
    cases=[dict(seed=seed,energy=energy,policy=policy,predators=2,pairs=2,food='trees',workers=2,worker_near=True,seconds=120,**BASE) for seed in (301,302,303) for energy in (150,500) for policy in ('none','orbit_single','orbit')]
else:
    cases=[
        dict(**CASES[2],energy=150,policy='orbit',food='trees',workers=2,worker_near=True,seconds=120,native_render=True,record_every=10,record='ordinary150-handoff-starvation',**BASE),
        dict(**CASES[0],energy=500,policy='orbit',food='trees',workers=2,worker_near=True,seconds=120,native_render=True,record_every=10,record='prepared500-newborn-handoff',**BASE),
        dict(seed=0,energy=150,policy='orbit',food='trees',workers=2,worker_near=True,seconds=120,native_render=True,record_every=10,record='ordinary150-food-exhaustion',**BASE),
        dict(seed=301,energy=500,policy='orbit',predators=2,pairs=2,food='trees',workers=2,worker_near=True,seconds=120,native_render=True,record_every=10,record='two-predator-pairs',**BASE)]
runs=[]
for case in cases:
    r=run(**case);runs.append(r);save(args.phase,runs)
    print(json.dumps({k:v for k,v in r.items() if k not in ('trace','events')}),flush=True)
