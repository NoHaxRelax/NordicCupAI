"""Bounded, observation-only attempts to leave a predator facing a corner."""
import math
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from models.entrapment.guide_lookahead import predator_step, wrap, TERRAIN


def steer(action, state, sightings, desired_heading, memory, now, *, tolerance=math.radians(20), strict=False):
    """Return an action and local-coordinate diagnostics; never takes world state."""
    target = min(sightings, key=lambda o: o['distance'])
    p = (target['distance']*math.cos(target['angle']), target['distance']*math.sin(target['angle']))
    heading = wrap(math.atan2(-p[1],-p[0])-target.get('rel_dir',0.))
    distance = math.hypot(*p)
    exposed = distance <= 60. or (distance <= 250. and abs(target.get('rel_dir',math.pi)) <= math.pi/6)
    if distance < 85. and not memory.get('phase'):
        return None, {}
    if not exposed and memory.get('phase') not in ('align','release'):
        return None, {}
    if 'started' not in memory:
        memory.update(started=now, phase='align', energy_start=state['energy'])
    error = abs(wrap(heading-desired_heading))
    exhausted = now-memory['started'] > 2.5 or state['energy'] < state['max_energy']*.32
    if exhausted:
        memory['phase'] = 'abort'
    elif error <= tolerance:
        memory['phase'] = 'release'
    elif strict and memory['phase']=='release' and exposed:
        memory['phase'] = 'align'
    if memory['phase']=='abort':
        # Ordinary escape takes over; never continue an expensive turning attempt.
        memory['phase']='done'
        return None, dict(phase='aborted',heading_error=error)
    edges = list(dict.fromkeys(tuple(tuple(p) for p in o['coords']) for o in state['observations'] if o['type']=='Edge'))
    walls = unary_union([LineString(e) for e in edges])
    def free(q): return walls.is_empty or Point(q).distance(walls) >= 10.01
    def clear(q):
        return walls.is_empty or LineString([(0.,0.),q]).distance(walls) >= min(5.01,Point(0.,0.).distance(walls))-1e-6
    def visible(p,h,q):
        d=math.dist(p,q)
        return d<=60. or (d<=250. and abs(wrap(math.atan2(q[1]-p[1],q[0]-p[0])-h))<=math.pi/6
                          and not LineString([p,q]).intersects(walls))
    modifier = TERRAIN[state['biome']]
    cap = state['sprint_speed'] if state['energy']>=state['max_energy']/5 else min(state['speed'],state['sprint_speed'])
    # Sample the missing native predator move. Hidden energy and biome are not read.
    projected=[]
    for speed in (11.,15.):
        for terrain in set((modifier,1.)):
            for bias in (-1.,1.):
                pp,hh=predator_step(p,heading,(0.,0.),0.,(1e8,1e8),free,visible,speed,bias,terrain)
                projected.append((pp,hh,speed,terrain,bias))
    away=math.atan2(-p[1],-p[0])
    angles=[action.move_direction,away,away-math.pi/2,away+math.pi/2]+[i*math.tau/32 for i in range(32)]
    best=None
    for length in sorted(set((min(state['speed'],cap),cap))):
        for angle in angles:
            q=(length*modifier*math.cos(angle),length*modifier*math.sin(angle))
            if not clear(q): continue
            turn=math.atan2(p[1]-q[1],p[0]-q[0])
            captures=0; minimum=math.inf; detects=0; heading_error=0.; spacing=0.
            path=None
            for pp,hh,speed,terrain,bias in projected:
                np,nh=predator_step(pp,hh,q,turn,(1e8,1e8),free,visible,speed,bias,terrain)
                gap=math.dist(np,q); minimum=min(minimum,gap); captures+=gap<15.
                detects+=visible(np,nh,q)
                heading_error+=abs(wrap(nh-desired_heading))
                spacing+=max(0.,95.-gap,gap-125.)
                path=[p,pp,np]
            # Other visible predators remain hazards even when one is being steered.
            for other in sightings:
                if other is target: continue
                op=(other['distance']*math.cos(other['angle']),other['distance']*math.sin(other['angle']))
                oh=math.atan2(-op[1],-op[0])-other.get('rel_dir',0.)
                op,oh=predator_step(op,oh,(0.,0.),0.,(1e8,1e8),free,visible)
                op,oh=predator_step(op,oh,q,turn,(1e8,1e8),free,visible)
                gap=math.dist(op,q);minimum=min(minimum,gap);captures+=gap<15.
            heading_error/=len(projected);spacing/=len(projected)
            cost=.05*min(length,state['speed'])+.5*max(0.,length-state['speed'])+abs(wrap(turn))/math.tau
            # Safety first. Alignment is a short soft objective, not a lure assignment.
            objective=(detects,heading_error,math.dist(q,(action.move_distance*modifier*math.cos(action.move_direction),
                                                       action.move_distance*modifier*math.sin(action.move_direction)))) if memory['phase']=='release' else (heading_error+.06*spacing, cost)
            rank=(captures>0,captures,-minimum if captures else max(0.,35.-minimum),*objective,cost)
            if best is None or rank<best[0]: best=(rank,length,angle,turn,path,minimum,detects)
    if best is None: return None, {}
    _,length,angle,turn,path,minimum,detects=best
    if not exposed and distance>80.:
        memory['phase']='done'
    return action.model_copy(update=dict(move_distance=length,move_direction=angle,turn_angle=turn,spawn_agent=False)),dict(
        phase=memory['phase'],target_position=list(p),target_heading=heading,desired_heading=desired_heading,
        heading_error=error,alignment_tolerance=tolerance,aligned_exit=memory['phase']=='done' and error<=tolerance,observed_distance=distance,minimum_projected_distance=minimum,
        sensed_after_move_scenarios=detects,elapsed=now-memory['started'],energy_spent=memory['energy_start']-state['energy'],
        predator_path=path,guide_path=[(0.,0.),(length*modifier*math.cos(angle),length*modifier*math.sin(angle))])
