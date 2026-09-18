"""Meaningful model/contract/replay checks, no competition validation API."""
from experiment import fixture,add_agent,add_pred,observe,ROOT,OUT
from controller import chase,BaitController
import random,math,numpy as np,gzip,json

def checks():
    rng=random.Random(991);worst=0.
    for i in range(200):
        e=fixture();p=add_pred(e,energy=200 if i%2 else 39,heading=rng.uniform(-math.pi,math.pi))
        ang=rng.uniform(-math.pi,math.pi);gap=rng.uniform(20,190)
        a=add_agent(e,p.x+gap*math.cos(ang),p.y+gap*math.sin(ang));a.direction=rng.uniform(-math.pi,math.pi)
        speed=15 if p.energy>=40 else 11
        obs=[dict(type='Agent',id=a.agent_id,distance=gap,angle=(ang-p.direction+math.pi)%(2*math.pi)-math.pi,rel_dir=(ang+math.pi-a.direction+math.pi)%(2*math.pi)-math.pi)]
        q,h=chase(np.array([p.x,p.y]),p.direction,np.array([a.x,a.y]),a.direction,np.array(float(speed)))
        action=p.step(obs);e.update_entity_position(p,action['move'],action['direction'],e._get_local_obstacles(p));e.update_entity_direction(p,action.get('turn',0))
        err=max(math.dist((p.x,p.y),q),abs((p.direction-h+math.pi)%(2*math.pi)-math.pi));worst=max(worst,err)
        assert err<1e-8,(i,err)
    # Cached observation DTOs, no shared engine references, one finite action.
    e=fixture();a=add_agent(e,800,600,150);p=add_pred(e,740,600)
    state=observe(e)[0];clean=json.loads(json.dumps(state));ctrl=BaitController()
    action,_=ctrl.step(clean,0)
    assert clean==state
    assert 0<=action.move_distance<=a.sprint_speed
    assert all(math.isfinite(v) for v in [action.move_distance,action.move_direction,action.turn_angle])
    files=[]
    for f in sorted(OUT.rglob('*.json.gz')):
        data=json.load(gzip.open(f,'rt'))
        assert data['format']=='survival-replay' and data['version']==1
        assert data['world']['width']==1600 and data['world']['height']==1200
        ts=[x['t'] for x in data['frames']];assert all(y>x for x,y in zip(ts,ts[1:]))
        files.append(dict(file=str(f.relative_to(OUT)),frames=len(ts),duration=data['summary']['duration']))
    result=dict(predictor_cases=200,max_error=worst,contract='passed',replays=files)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':checks()
