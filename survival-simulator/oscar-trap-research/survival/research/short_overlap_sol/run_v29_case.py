import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research'))
from simple_chase import run_streaming_v2 as harness
harness.OUT=ROOT/'results'/'short_overlap_sol'/'v29_terrain_emergency'
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--map-seed',type=int,required=True);p.add_argument('--fixture-seed',type=int,required=True);p.add_argument('--policy',choices=('guide','emergency'),default='emergency');a=p.parse_args()
 cls='Guide' if a.policy=='guide' else 'EmergencyGuides'
 path,result=harness.run(policy_spec=f'short_overlap_sol.policy_v29_terrain_emergency:{cls}',map_seed=a.map_seed,fixture_seed=a.fixture_seed,seconds=300,predators=1,station_bait=True,native_render=True,native_width=320,mode='direct_guide')
 print(json.dumps({k:result.get(k) for k in ('map_seed','fixture_seed','seconds','guide_alive','child_births','physical_entered','guided_delivered','physical_losses_after_delivery','success')}),flush=True);print(path)
