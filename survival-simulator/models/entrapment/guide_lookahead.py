"""Three-tick guide search with public predator turning and capture rules.

Preferred spacing is a soft 100-120-unit target. Only the game's actual
15-unit contact after an enemy move is a predicted capture, not closeness
during our move. Unknown enemy speed/pivot outcomes are sampled, not read.
"""
from dataclasses import dataclass
import math
from shapely.geometry import LineString, Point

TERRAIN = {'forest': 1., 'grassland': 1., 'desert': .8, 'swamp': .5, 'river': .3}
HORIZON = 3
BEAM = 4
CAPTURE_RADIUS = 15.
PREDATOR_MOTION = tuple((cap, terrain) for cap in (15., 11.) for terrain in (1., .8, .5, .3))


def wrap(angle):
    return math.atan2(math.sin(angle),math.cos(angle))


def chase_step(p, target, clear=None):
    """Coarse corridor forecast for an intentional delivery hold."""
    distance = math.dist(p,target)
    if distance < 1e-9:
        return p
    length = min(15.,distance)
    angle = math.atan2(target[1]-p[1],target[0]-p[0])
    for offset in (0.,-math.pi/18,math.pi/18,-math.pi/9,math.pi/9):
        q = (p[0]+length*math.cos(angle+offset),p[1]+length*math.sin(angle+offset))
        if clear is None or clear(p,q):
            return q
    return p


def predator_step(p, heading, guide, guide_heading, bait, free, visible,
                  speed_cap=15., pivot_bias=0., terrain=1.):
    """Public bounded-turn/watched-strafe rules; move BEFORE heading update.

    Speed cap and terrain sample the unknown walk/sprint and biome gates. Resting,
    unseen competing prey and wandering are not exactly predictable.
    """
    target, target_heading = guide, guide_heading
    if math.dist(p,bait)<math.dist(p,guide) and visible(p,heading,bait):
        target, target_heading = bait, 0.
    distance = math.dist(p,target)
    angle = wrap(math.atan2(target[1]-p[1],target[0]-p[0])-heading)
    if visible(p,heading,target):
        looking = wrap(math.atan2(p[1]-target[1],p[0]-target[0])-target_heading)
        if abs(looking)>math.pi/2 or distance<90.:
            turn = max(-.3,min(.3,angle*.5)) if abs(angle)>.05 else 0.
            direction = turn if abs(angle)>.05 else angle
            requested = min(15.,distance)
        else:
            sign = pivot_bias if abs(looking)<1e-7 else (-1. if looking>0. else 1.)
            direction = angle+sign*math.pi/4
            turn = math.atan2(distance*math.sin(angle)-15.*math.sin(direction),
                              distance*math.cos(angle)-15.*math.cos(direction))
            requested = 15.
    else:
        # Exact wandering needs hidden random draws and edge visibility.
        requested, direction, turn = 11.,0.,0.
    length = min(requested,speed_cap)*terrain
    absolute = heading+direction
    result = p
    for i in range(36):
        adjusted = absolute+math.pi/18*((i+1)//2)*(-1)**i
        q = (p[0]+length*math.cos(adjusted),p[1]+length*math.sin(adjusted))
        if free(q):
            result = q
            break
    return result,wrap(heading+turn)


@dataclass(slots=True)
class Node:
    point: tuple
    heading: float
    energy: float
    predators: list
    predator_headings: list
    first: dict | None
    path: list
    headings: list
    predator_paths: list
    commands: list
    survival: int = HORIZON+1
    separation: float = math.inf
    spent: float = 0.


def search(action,bait,agent,memory,positions,target_index,geometry,walls,
           predator_geometry,to_fixed,to_local):
    modifier = TERRAIN[agent['biome']]
    current_modifier = modifier
    debug = memory.get('debug')
    route = debug.get('route',{}) if isinstance(debug,dict) else {}
    waypoints = route.get('waypoints') or [(
        HORIZON*action['move_distance']*modifier*math.cos(action['move_direction']),
        HORIZON*action['move_distance']*modifier*math.sin(action['move_direction']))]
    following_distance = memory.get('_preferred_predator_distance',(100.,120.))
    reacquiring = bool(memory.get('_reacquire_close',False) and memory.get('_predator_not_following',False))
    # The following gap is useful only after contact is established. Keeping
    # 100 units away from a predator looking elsewhere prevents reacquisition.
    preferred_min,preferred_max = (0.,55.) if reacquiring else following_distance
    samples = [((0.,0.),modifier,0.)] + [
        (tuple(p),TERRAIN[biome],uncertainty)
        for p,biome,uncertainty in memory.get('_terrain_samples',())]

    def terrain_at(point):
        # These are visited points, not known biome polygons. Close samples
        # guide the nominal forecast; separate slowdown branches cover gaps.
        p,factor,uncertainty = min(samples,key=lambda row: math.dist(point,row[0]))
        return factor if math.dist(point,p)+uncertainty<=12. else current_modifier
    observations = [o for o in agent['observations'] if o['type']=='Predator']
    initial_headings = [wrap(math.atan2(-p[1],-p[0])-o.get('rel_dir',0.))
                        for p,o in zip(positions,observations)]
    forecast = list(positions)
    target = positions[target_index]
    tick = memory.get('_following_call_tick',0)
    previous = memory.get('_forecast_observation')
    memory['_forecast_observation'] = (tick,to_fixed(target))
    lag_known = False
    if previous is not None and tick-previous[0]==1:
        old = to_local(previous[1])
        velocity = (target[0]-old[0],target[1]-old[1])
        q = (target[0]+velocity[0],target[1]+velocity[1])
        if math.hypot(*velocity)<=16. and predator_geometry.free(to_fixed(q)):
            forecast[target_index] = q
            lag_known = True

    def clear(a,b):
        fa,fb = to_fixed(a),to_fixed(b)
        if geometry.clear(fa,fb):
            return True
        x0,y0,x1,y1 = geometry.bounds
        if not (x0<=fb[0]<=x1 and y0<=fb[1]<=y1):
            return False
        path = LineString([fa,fb])
        return (not path.intersects(walls) and
                path.distance(walls)>=min(5.01,Point(fa).distance(walls))-1e-6)

    def free(p):
        return predator_geometry.free(to_fixed(p))

    def visible(p,h,g):
        distance = math.dist(p,g)
        if distance<=60.:
            return True
        return (distance<=250. and abs(wrap(math.atan2(g[1]-p[1],g[0]-p[0])-h))<=math.pi/6
                and not walls.intersects(LineString([to_fixed(p),to_fixed(g)])))

    nodes = [Node((0.,0.),0.,agent['energy'],forecast,initial_headings,None,
                  [(0.,0.)],[0.],[[p] for p in forecast],[])]
    expanded = 0
    for depth in range(HORIZON):
        choices = []
        for node in nodes:
            g,p = node.point,node.predators[target_index]
            modifier = terrain_at(g)
            goal = next((w for w in waypoints if math.dist(w,g)>3.),waypoints[-1])
            bearing = math.atan2(goal[1]-g[1],goal[0]-g[0])
            away = math.atan2(g[1]-p[1],g[0]-p[0])
            angles = [bearing,bearing-math.pi/4,bearing+math.pi/4,away,away-math.pi/2,away+math.pi/2]
            angles += [i*math.tau/12 for i in range(12)]
            angles = list(dict.fromkeys(round(wrap(a),8) for a in angles))
            cap = agent['sprint_speed'] if node.energy>=agent['max_energy']/5 else agent['speed']
            lengths = {min(agent['speed'],cap),cap}
            if node.energy < agent['max_energy']/5+3*(.05*agent['speed']+.5*max(0.,cap-agent['speed'])+1.):
                lengths.add((min(agent['speed'],cap)+cap)/2)
            commands = [(0.,bearing)]
            for length in sorted(lengths):
                commands += [(min(length,math.dist(goal,g)/modifier) if abs(wrap(a-bearing))<1e-7 else length,a)
                             for a in angles]
            for length,angle in commands:
                q = (g[0]+length*modifier*math.cos(angle),g[1]+length*modifier*math.sin(angle))
                if length and not clear(g,q):
                    continue
                heading = math.atan2(p[1]-q[1],p[0]-q[0])
                turning = abs(wrap(heading-node.heading))
                cost = .05*min(length,agent['speed'])+.5*max(0.,length-agent['speed'])+turning/math.tau+.15
                age = agent['age']+.1*depth
                if age>=60.: cost+=.01*age
                if cost>=node.energy:
                    continue
                preds,headings = [],[]
                for old_p,h in zip(node.predators,node.predator_headings):
                    new_p,new_h = predator_step(old_p,h,q,heading,bait,free,visible)
                    preds.append(new_p);headings.append(new_h)
                # The native engine tests contact after the enemy's move,
                # not against swept agent/predator paths.
                separation = min(math.dist(q,p) for p in preds)
                survival = min(node.survival,depth if separation<CAPTURE_RADIUS else HORIZON+1)
                contact = visible(preds[target_index],headings[target_index],q)
                distance = math.dist(q,preds[target_index])
                spacing_error = max(0.,preferred_min-distance,distance-preferred_max)
                # Spacing is a preference, never a forbidden region. The last
                # approach prioritizes reaching delivery so the handoff works.
                spacing_weight = 0. if math.dist(q,bait)<=80. else 1.5
                route_cost = math.dist(q,goal)+spacing_weight*spacing_error
                rank = (-survival,-min(node.separation,separation) if survival<=depth else 0.,
                        not contact,route_cost,node.spent+cost)
                first = node.first or dict(move_distance=length,move_direction=angle,turn_angle=heading)
                child = Node(q,heading,node.energy-cost,preds,headings,first,node.path+[q],
                    node.headings+[heading],[points+[p] for points,p in zip(node.predator_paths,preds)],
                    node.commands+[(length,angle)],survival,min(node.separation,separation),node.spent+cost)
                choices.append((rank,child));expanded+=1
        if not choices:
            return None
        choices.sort(key=lambda x:x[0])
        nodes,seen = [],set()
        for rank,node in choices:
            key = (round(node.first['move_distance'],2),round(node.first['move_direction'],3))
            if key in seen: continue
            seen.add(key);nodes.append(node)
            if len(nodes)>=BEAM: break

    # Check surviving paths with possible physical speeds and pivot signs.
    # The nominal search avoids a full branching game-tree explosion.
    checked = []
    for order,node in enumerate(nodes):
        captures = 0; worst_tick = HORIZON+1; minimum = math.inf; scenarios = 0
        for cap,terrain in PREDATOR_MOTION:
            for bias in (-1.,0.,1.):
                # Current biome is observed. Future starts may land in river,
                # while the predator remains on faster ground. Replay actual
                # commands with that slowdown, rather than nominal endpoints.
                for slow_at in (HORIZON,1,2):
                    preds,headings = list(forecast),list(initial_headings)
                    q=(0.,0.); captured=False
                    for depth,(length,angle) in enumerate(node.commands):
                        factor = .3 if depth>=slow_at else terrain_at(q)
                        new_q=(q[0]+length*factor*math.cos(angle),q[1]+length*factor*math.sin(angle))
                        # If this branch hits a wall, model staying put; native
                        # deflection direction remains an additional uncertainty.
                        if clear(q,new_q): q=new_q
                        p=preds[target_index]; h=math.atan2(p[1]-q[1],p[0]-q[0])
                        for i in range(len(preds)):
                            preds[i],headings[i] = predator_step(preds[i],headings[i],q,h,bait,free,visible,cap,bias,terrain)
                        separation = min(math.dist(q,p) for p in preds)
                        minimum = min(minimum,separation)
                        if separation<CAPTURE_RADIUS:
                            captured=True;worst_tick=min(worst_tick,depth);break
                    captures+=captured;scenarios+=1
        checked.append(((captures>0,-worst_tick,captures,order),node,captures,scenarios,minimum))
    _,winner,captures,scenarios,minimum = min(checked,key=lambda x:x[0])
    return winner.first,dict(horizon_ticks=HORIZON,horizon_seconds=.3,
        capture_radius=CAPTURE_RADIUS,extra_predator_clearance=0.,preferred_distance=[preferred_min,preferred_max],
        reacquiring_contact=reacquiring,following_distance=list(following_distance),
        predicted_safe=captures==0,capture_scenarios=captures,sampled_scenarios=scenarios,
        minimum_separation=round(minimum,2),observation_lag_estimated=lag_known,expanded=expanded,
        terrain_samples=len(samples),terrain_scenarios=['observed_samples','river_next_tick','river_in_two_ticks'],
        guide_path=winner.path,predator_path=winner.predator_paths[target_index],
        note='Public turning/capture rules; sampled unknown speed/pivot. Rest, wandering, lag remain uncertain.')
