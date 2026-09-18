"""Run the stock guide_multi protocol on seed 1989373803's diagonal pocket.

Only guide_lab.setup/site selection and the rear evaluation predicate are
overridden. The guide policy and guide_multi simulation loop are imported
unchanged from the repository.
"""
from __future__ import annotations
import argparse, json, math, os, shutil, sys
from pathlib import Path

os.environ.setdefault('SDL_VIDEODRIVER','dummy'); os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import guide_lab as lab
import guide_multi as multi
from src.core import SimulationCore
from src.elements.agent import Agent
from src.elements.predator import Predator

MAP_SEED=1989373803
GOAL=(498.0,211.0)
# Radius-11 reachable point in the main/front component, selected for high
# obstacle clearance while remaining comfortably inside bait hearing range.
FRONT=(512.0,184.5)
REAR=(498.5463418043246,283.5842882865205)
ENCOUNTER_SEEDS=[1382370471,854619562,1643999355,2085883922,1374628056,
    772838390,672263659,2078492351,326676940,2133973059,812073905,
    52806821,1619300663,925977651,1495413983,1261436949,45208059,
    2128649714,355737998,504886942]

def site_contract():
    dx,dy=GOAL[0]-FRONT[0],GOAL[1]-FRONT[1];n=math.hypot(dx,dy);inward=(dx/n,dy/n);cross=(-inward[1],inward[0])
    return dict(site_kind='generic_corner_safe_region',goal=list(GOAL),handoff=list(FRONT),mouth=list(GOAL),
                inward=list(inward),cross=list(cross),gap=None,overlap=0.,far=list(FRONT),hold=list(FRONT),runup=list(FRONT),
                approach_lane_offset=0.,offset_approach=False,obstacle_indices=[8,19],axis=None,
                other_mouth=list(REAR),replacement_entry=list(REAR),second_access_clear=True,
                second_access_radius=5.01,bait_depth=None,boundary_indices=[],geometric_replacement_access_only=True,
                minimum_predator_distance=15.36,
                rear_contract='no predator centre within 15.05 of replacement segment from replacement_entry to goal')

def setup(seed,encounter_seed,site_index,*,corner_only=False):
    if seed!=MAP_SEED or site_index!=0:raise ValueError('override supports only map seed 1989373803/site0')
    core=SimulationCore(seed=seed,starting_agents=0,starting_predators=0);env=core.env;site=site_contract()
    bait=Agent(*GOAL,rng=env.rng);bait.agent_id=0;bait.energy=bait.max_energy
    unused_guide=Agent(0,0,rng=env.rng);unused_guide.agent_id=1;unused_guide.energy=unused_guide.max_energy
    unused_predator=Predator(0,0,rng=env.rng);unused_predator.energy=unused_predator.max_energy;unused_predator.resting=False
    env.agents=[bait,unused_guide];env.agents_dict={0:bait,1:unused_guide};env._next_agent_id=2;env.predators=[unused_predator]
    env._update_agent_grid();env._update_predator_grid()
    return core,site,1,bait,unused_guide,unused_predator

def segment_distance(point,a=REAR,b=GOAL):
    vx,vy=b[0]-a[0],b[1]-a[1];wx,wy=point[0]-a[0],point[1]-a[1]
    t=max(0.,min(1.,(wx*vx+wy*vy)/(vx*vx+vy*vy)))
    return math.dist(point,(a[0]+t*vx,a[1]+t*vy))

def replacement_side(point,site):
    # Generic corners have no meaningful axial channel. The physical rear
    # contract is contact clearance along the complete replacement segment.
    return segment_distance(point)<15.05

def run_one(encounter_seed,output,bulk):
    lab.setup=setup;multi.lab.setup=setup;multi.replacement_side=replacement_side
    args=argparse.Namespace(seed=MAP_SEED,encounter_seed=encounter_seed,site=0,deliveries=1,bulk=bulk,
        vision_delivery=False,corner_only=False,replace_bait=True,output=output)
    folder=multi.run(args)
    shutil.copy2(__file__,folder/'corner_override.py')
    return folder

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--encounter-seed',type=int)
    p.add_argument('--batch',action='store_true');p.add_argument('--bulk',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    seeds=ENCOUNTER_SEEDS if a.batch else [a.encounter_seed if a.encounter_seed is not None else ENCOUNTER_SEEDS[0]]
    manifest=dict(map_seed=MAP_SEED,encounter_seeds=seeds,site=site_contract(),runner=str(Path(__file__).resolve()))
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    folders=[]
    for seed in seeds:folders.append(run_one(seed,a.output,a.bulk))
    rows=[]
    for folder in folders:
        summary=json.loads((folder/'summary.json').read_text());rows.append(dict(folder=folder.name,encounter_seed=summary['encounter_seed'],
            outcome=summary['outcome'],initial_min_held=summary['initial_min_held'],final_hold_min=summary['final_hold_min'],
            replacement=summary.get('replacement'),replacement_side_final_period=summary['replacement_side_final_period']))
    (a.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows),passes=sum(r['outcome']=='delivery_pass' for r in rows),rows=rows),indent=2))
if __name__=='__main__':main()
