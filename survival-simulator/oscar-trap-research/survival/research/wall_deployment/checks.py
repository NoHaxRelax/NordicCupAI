"""Focused checks of sensor registration, collision correction and action timing."""
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predator_control import controlled_env, add_agent, add_predator
from src.elements.obstacle import Obstacle
from controller import WallPolicy, Pose, Group, rot, sub, add


def observe(env):
    for a in env.agents:
        agents, fruits, trees, obstacles, predators, edges = env._get_local_objects(a)
        env.agent_observations[a.agent_id] = a.observe(agents,fruits,trees,obstacles,edges,predators)
    return [env.get_agent_state(a.agent_id) for a in env.agents]


def main():
    env = controlled_env()
    wall = Obstacle(800,560,30,80)
    env.obstacles=[wall]
    env.edges.update((min(a,b),max(a,b)) for a,b in wall.edges)
    a=add_agent(env,837,600,150); b=add_agent(env,862,617,150)
    a.direction=2.6; b.direction=-.4
    p=add_predator(env,788,600,heading=0)
    env._update_spatial_grid()
    states = observe(env)
    policy=WallPolicy()
    policy.time=.1
    policy._register({s['agent_id']:s for s in states})
    assert policy.group_for[0] is policy.group_for[1]
    expected=rot((25,17),-a.direction)
    assert math.dist(policy.poses[1].p,expected)<1e-8
    assert abs(math.remainder(policy.poses[1].theta-(b.direction-a.direction),2*math.pi))<1e-8
    original=(b.x,b.y)
    b.x,b.y=a.x,a.y
    env._update_spatial_grid()
    colocated=WallPolicy()
    colocated._register({s['agent_id']:s for s in observe(env)})
    assert math.dist(colocated.poses[0].p,colocated.poses[1].p)<1e-8
    assert abs(math.remainder(colocated.poses[1].theta-(b.direction-a.direction),2*math.pi))<1e-8
    b.x,b.y=original
    env._update_spatial_grid()
    for aid,s in enumerate(states): policy._map(aid,s)
    group=policy.group_for[0]
    assert group.edges
    # Simulate a wrong odometry prediction. The same static edge must correct it
    # solely from its currently sensed relative endpoints.
    policy.poses[0].p=add(policy.poses[0].p,(12.,-4.))
    policy._map(0,states[0])
    assert math.dist(policy.poses[0].p,(0,0))<1e-8
    assert not policy._blocked(group, (0.,0.), (0.,0.))
    # A newborn within the 6-unit planning buffer can move outward. Crossing
    # that observed edge remains forbidden.
    test_group=Group(edges=[((0.,-40.),(0.,40.))])
    assert not policy._blocked(test_group,(3.,0.),(10.,0.))
    assert policy._blocked(test_group,(3.,0.),(-10.,0.))
    predators=policy._predators(group,{s['agent_id']:s for s in states})
    policy._recognize(group,{s['agent_id']:s for s in states},predators)
    assert group.trap and group.trap.holder==0
    # Seeded irregular headings do not expose world coordinates to policy.
    rows=[]
    for tick in range(100):
        states=observe(env)
        actions=policy.act(states,.1+tick*.1)
        assert len(actions)==len(env.agents)==len({a['agent_id'] for a in actions})
        for action in actions:
            env.agent_step(**action)
        env.non_agent_step(.1)
    assert 0 in env.agents_dict
    assert p.x<800 and abs(p.y-600)<50
    rows.append('relative registration with unequal headings')
    rows.append('colocated newborn heading from mutual zero-gap observations')
    rows.append('12,-4 odometry drift corrected by native static edge')
    rows.append('newborn can leave clearance buffer but cannot cross wall')
    rows.append('wall recognized with no supplied anchor or world coordinates')
    rows.append('100 native ticks retain holder with one action per live agent')
    result=dict(passed=rows,policy_metrics=policy.metrics)
    output=Path(__file__).resolve().parents[2]/'results/wall_deployment/checks.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
