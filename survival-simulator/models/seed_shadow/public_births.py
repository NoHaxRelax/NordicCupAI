"""Ordered RNG constraints within an observed, unambiguous reproduction batch.

Input is solely public before/after responses and the agent's executed actions.
Draw offsets are relative to this batch, NOT to the entire RNG stream. Environment
updates between batches consume unknown draws. Do not concatenate these blocks.
"""
import math
from .public_terrain import agent_states

TRAITS=(('speed',20.),('sprint_speed',40.),('max_energy',1000.),
        ('hearing_radius',100.),('vision_range',400.),('vision_angle',math.pi/2))


def unknown(reason):
    return dict(interval=[0.,1.],reason=reason)


def uniform(a,b,value,error=1e-9):
    return dict(interval=[max(0.,(value-error-a)/(b-a)),min(1.,(value+error-a)/(b-a))])


def birth_block(before,actions,after,terrain):
    old={a['agent_id']:a for a in agent_states(before)}
    new={a['agent_id']:a for a in agent_states(after) if a['agent_id'] not in old}
    requested=[]
    for aid,action in actions:
        spawn=action.get('spawn_agent') if isinstance(action,dict) else action.spawn_agent
        if spawn and aid in old:requested.append(aid)
    if not requested or len(requested)!=len(new) or len(set(requested))!=len(requested):
        return None  # Failed/hidden births make action-to-child matching ambiguous.
    draws=[];pairs=[]
    for parent_id,child_id in zip(requested,sorted(new)):
        parent,child=old[parent_id],new[child_id]
        if child['age']>.10000001:
            return None
        pp=terrain.last_poses.get(parent_id);cp=terrain.last_poses.get(child_id)
        angle=unknown('unobserved or rejected birth position')
        distance=unknown('unobserved or rejected birth position')
        if pp and cp:
            dx,dy=cp[0]-pp[0],cp[1]-pp[1];d=math.hypot(dx,dy)
            if 10-1e-7<=d<=30+1e-7:
                a=math.atan2(dy,dx)%(2*math.pi)
                if 1e-8<a<2*math.pi-1e-8:angle=uniform(0,2*math.pi,a)
                distance=uniform(10,30,d)
        offset=len(draws);draws.extend([angle,distance])
        for name,cap in TRAITS:
            a,b=parent[name],child[name]
            if a==b:
                if b==cap:return None  # Clipped mutation indistinguishable from none.
                draws.append(dict(interval=[.1,1.],reason='no observed mutation'))
            else:
                draws.append(dict(interval=[0.,.1],reason='observed mutation'))
                if b==cap:
                    draws.append(dict(interval=[max(0.,cap/a-.5),1.],reason='clipped mutation'))
                else:
                    draws.append(uniform(.5,1.5,b/a,error=1e-12))
        draws.append(unknown('temporary Agent constructor heading'))
        draws.append(unknown('temporary Agent constructor maximum age'))
        h=cp[2]%(2*math.pi) if cp else None
        draws.append(uniform(0,2*math.pi,h) if h is not None and 1e-8<h<2*math.pi-1e-8
                     else unknown('unobserved birth heading'))
        draws.append(unknown('child maximum age'))
        pairs.append(dict(parent=parent_id,child=child_id,draw_offset=offset))
    return dict(time=after.get('sim_time'),pairs=pairs,draws=draws,
                global_draw_offset=None,inter_batch_draw_count_unknown=True)
