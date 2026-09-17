"""Complete mapped approach + native-food replacement trials."""
from common import *
from run import run
import argparse

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',action='store_true');a=p.parse_args()
    maps={m['seed']:m['sites'] for file in ['sites-1-40','sites-41-80']
        for m in json.loads((OUT/(file+'.json')).read_text())['data']}
    rows=[]
    for seed,i in ([(37,8)] if a.record else [(37,8),(37,10),(23,2),(70,5)]):
        c=dict(native=True,seed=seed,site=maps[seed][i],site_index=i,crew=4,
            relay='reserve',outside=45,heading=-.25,collision_model=True,seconds=150,association_gate=True)
        r=run(c,record='approach-relay-native.replay.json' if a.record else None)
        rows.append(r);print(seed,i,r.get('elapsed_s'),r.get('after_acquisition_joint_fraction'),flush=True)
        save('approach-relay-record' if a.record else 'approach-relay',rows)
