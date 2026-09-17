"""Run one fixed v19 early-safety diagnostic with native every-frame recording."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'))
from simple_chase import run_streaming_v2 as harness
harness.OUT=ROOT/'results'/'short_overlap_sol'/'v24_terrain_lead'
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--map-seed',type=int,required=True);p.add_argument('--fixture-seed',type=int,required=True);a=p.parse_args()
 path,result=harness.run(policy_spec='short_overlap_sol.policy_v24_terrain_lead:SimpleChase',map_seed=a.map_seed,fixture_seed=a.fixture_seed,seconds=180,predators=1,station_bait=True,native_render=True,native_width=320,mode='direct_guide')
 print(json.dumps({k:result.get(k) for k in ('map_seed','fixture_seed','seconds','guide_alive','physical_entered','guided_delivered','physical_losses_after_delivery','success')}),flush=True);print(path)
