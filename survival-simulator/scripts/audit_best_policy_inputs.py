"""Synthetic public-input audit only: no simulator or validation service is run."""
import copy,json,math,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.experiment_actor import ExperimentActor
from models.notrap_config import validate

def main():
    state=dict(agent_id=1,energy=500.,biome='forest',age=0.,speed=10.,sprint_speed=20.,hearing_radius=50.,vision_angle=math.pi/3,vision_range=200.,max_energy=500.,observations=[dict(type='Tree',distance=80.,angle=.2),dict(type='Fruit',distance=40.,angle=.1),dict(type='Predator',distance=65.,angle=.4,rel_dir=0.),dict(type='Edge',coords=[[-100.,-100.],[100.,-100.]])])
    rows=[]
    for path in sorted((ROOT/'models/best_policies').glob('rank*.json')):
        config=json.loads(path.read_text());validate(config)
        with ExperimentActor('orchard_evasion',config,policy_seed=0) as clean, ExperimentActor('orchard_evasion',config,policy_seed=0) as poisoned:
            count=0
            for i in range(30):
                visible=copy.deepcopy(state);visible['age']=i*.1
                hidden=copy.deepcopy(visible);hidden.update(x=9999,y=-1234,direction=123,world_seed=99999,max_age=123,unseen_fruit=object(),world=object())
                for obs in hidden['observations']:obs.update(world_position=[555,666],world_seed=23,energy=9999,age=888,max_age=9999)
                hidden['observations'][0]['id']=12345
                hidden['observations'][1]['id']=12345
                hidden['observations'][2]['id']=12345
                hidden['observations'][3]['world_coords']=object()
                assert clean([visible],i*.1)==poisoned([hidden],i*.1)
                assert clean.last_audit['env_loaded'] is False and poisoned.last_audit['env_loaded'] is False
                count+=1
            rows.append(dict(config=path.name,synthetic_steps=count,actions_identical=True,audit=clean.audit))
    out=dict(commit='6e49081d845dae6f650b4e3d365f383ea4f31ee8',simulation_run=False,competition_validation_attempt=False,results=rows)
    (ROOT/'docs/nikolaj_validation/input-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
