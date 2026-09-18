"""Constrain along-bank following to the surveyed corridor, a single change."""
from common import *
from run import run
import argparse

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',action='store_true');a=p.parse_args()
    maps={m['seed']:m['sites'] for file in ['sites-1-40','sites-41-80']
        for m in json.loads((OUT/(file+'.json')).read_text())['data']}
    rows=[]
    cases=[]
    for seed,i in [(37,0),(37,9),(58,0),(70,5),(37,8),(37,10)]:
        for heading in [0,.35]:
            cases.append(dict(native=True,seed=seed,site=maps[seed][i],site_index=i,
                heading=heading,seconds=90,collision_model=True,anchor_window=15))
    for heading in [0,.35]:
        cases.append(dict(native=True,seed=37,site=maps[37][10],second_station=maps[37][8],
            predators=2,crew=4,heading=heading,seconds=90,collision_model=True,anchor_window=15))
    for c in cases:
        r=run(c);rows.append(r)
        print(c['seed'],c.get('site_index','twin'),c['heading'],r.get('elapsed_s'),r.get('joint_fraction'),flush=True)
        save('corridor',rows)
