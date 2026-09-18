import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research'))
from simple_chase import run_streaming_v2 as harness
harness.OUT=ROOT/'results'/'short_overlap_sol'/'v22_prediction_probe'
if __name__=='__main__':
 harness.run(policy_spec='short_overlap_sol.policy_v22_prediction_probe:SimpleChase',map_seed=10138,fixture_seed=20138,seconds=2,predators=1,station_bait=True,native_render=True,native_width=320,mode='direct_guide')
