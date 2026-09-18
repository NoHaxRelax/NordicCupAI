"""Separate stations for two predators, with four paid 150-energy baits."""
from common import *
from run import run

rows=[]
for width in [44,50,56]:
    for heading in [0,.35]:
        c=dict(width=width,seed=233,crew=4,predators=2,seconds=65,
            site=dict(center=[800,400],angle=0,width=width),
            second_station=dict(center=[800,800],angle=0,width=width),heading=heading,collision_model=True)
        r=run(c);rows.append(r);print(width,heading,r.get('elapsed_s'),r.get('joint_fraction'),flush=True)
m={m['seed']:m['sites'] for m in json.loads((OUT/'sites-1-40.json').read_text())['data']}
for heading in [0,.35]:
    c=dict(native=True,seed=37,crew=4,predators=2,seconds=90,site=m[37][10],
        second_station=m[37][8],heading=heading,collision_model=True)
    r=run(c);rows.append(r);print('native',heading,r.get('skipped'),r.get('elapsed_s'),r.get('joint_fraction'),flush=True)
save('twins',rows)
