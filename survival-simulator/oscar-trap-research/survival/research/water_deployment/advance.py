"""Predeclared follow-up cases; selection comes from the geometry survey."""
from common import *
from run import run
import argparse

def cases_for(mode):
    maps=json.loads((OUT/'sites-1-40.json').read_text())['data']
    lookup={m['seed']:m['sites'] for m in maps}
    held=json.loads((OUT/'sites-41-80.json').read_text())['data']
    if mode=='heldout':
        for m in held:
            # Include the least obstructed candidate per unseen map, even if
            # it fails the clearance qualification. Preserve blocked starts.
            if m['sites']:
                i,s=min(enumerate(m['sites']),key=lambda q:q[1]['obstructed_samples'])
                for heading in [0,.35]:
                    yield dict(native=True,seed=m['seed'],site=s,site_index=i,
                        collision_model=True,heading=heading,seconds=90)
    elif mode in ['acquisition','dogleg']:
        for seed,index in [(23,2),(23,3),(37,8),(37,10),(37,11)]:
            for gap,heading in ([(30,.15)] if mode=='dogleg' else [(30,.15),(45,-.25)]):
                yield dict(native=True,seed=seed,site=lookup[seed][index],site_index=index,
                    collision_model=True,seconds=75,outside=gap,heading=heading,safety_launch=mode=='dogleg')
    elif mode in ['native-renewal','native-renewal-v3']:
        for seed,index in [(23,2),(23,3),(37,8),(37,10),(37,11)]:
            for relay in (['renewal-urgent'] if mode=='native-renewal-v3' else ['none','reserve','renewal']):
                yield dict(native=True,seed=seed,site=lookup[seed][index],site_index=index,
                    collision_model=True,seconds=150,crew=4,relay=relay)
    elif mode in ['renewal','renewal-v3']:
        for seed in [173,177,181,197,211,223]:
            yield dict(seed=seed,width=50,seconds=180,crew=4,relay='renewal-urgent' if mode=='renewal-v3' else 'renewal',orchard=True)
    elif mode=='multi':
        for width in [44,50,56]:
            for heading in [0,.35]:
                yield dict(seed=197,width=width,seconds=70,crew=4,relay='reserve',predators=2,heading=heading)
    elif mode=='native-record':
        yield dict(native=True,seed=37,site=lookup[37][8],site_index=8,
            collision_model=True,seconds=80)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['heldout','acquisition','dogleg','native-renewal','native-renewal-v3','renewal','renewal-v3','multi','native-record'])
    a=p.parse_args();rows=[]
    for c in cases_for(a.mode):
        r=run(c,record='native-hold.replay.json' if a.mode=='native-record' else None);rows.append(r)
        print(c.get('seed'),c.get('site_index'),c.get('relay'),c.get('outside'),
            r.get('skipped'),r.get('elapsed_s'),r.get('joint_fraction'),len(r.get('handoffs',[])),len(r.get('births',[])),flush=True)
        save(a.mode,rows)
