"""Bounded three-tick guide search from ordinary sightings and observed walls.

Predator motion is an intentionally fast chase approximation (15 units/tick),
not hidden engine state. Rest, unknown terrain, target switches and native
collision deflections make the forecast uncertain. Replan on every observation.
"""
import math
from shapely.geometry import LineString, Point

TERRAIN = {'forest': 1., 'grassland': 1., 'desert': .8, 'swamp': .5, 'river': .3}
HORIZON = 3
BEAM = 6


def segment_distance(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    t = max(0., min(1., ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/max(1e-12, dx*dx+dy*dy)))
    return math.hypot(p[0]-a[0]-t*dx, p[1]-a[1]-t*dy)


def chase_step(p, target, clear=None):
    """Fastest physical chase, stopping at observed walls if supplied."""
    distance = math.dist(p, target)
    if distance < 1e-9:
        return p
    length = min(15., distance)
    direction = math.atan2(target[1]-p[1], target[0]-p[0])
    # Native movement tries nearby deflections when its requested move hits a
    # wall. This bounded subset estimates the useful near-forward alternatives.
    for offset in (0., -math.pi/18, math.pi/18, -math.pi/9, math.pi/9):
        q = (p[0]+length*math.cos(direction+offset), p[1]+length*math.sin(direction+offset))
        if clear is None or clear(p, q):
            return q
    return p


def search(action, bait, agent, memory, positions, target_index, geometry, walls,
           predator_geometry, to_fixed, to_local):
    """Return the first action plus a predicted route, or None if no move fits."""
    target = positions[target_index]
    modifier = TERRAIN[agent['biome']]
    debug = memory.get('debug')
    route = debug.get('route', {}) if isinstance(debug, dict) else {}
    waypoints = route.get('waypoints') or [(
        HORIZON*action['move_distance']*modifier*math.cos(action['move_direction']),
        HORIZON*action['move_distance']*modifier*math.sin(action['move_direction']))]
    tick = memory.get('_following_call_tick', 0)
    previous = memory.get('_forecast_observation')
    memory['_forecast_observation'] = (tick, to_fixed(target))
    forecast = list(positions)
    lag_known = False
    if previous is not None and tick-previous[0] == 1:
        old = to_local(previous[1])
        velocity = (target[0]-old[0], target[1]-old[1])
        if math.hypot(*velocity) <= 16.:
            q = (target[0]+velocity[0], target[1]+velocity[1])
            if math.hypot(*q) > 18. and predator_geometry.free(to_fixed(q)):
                forecast[target_index] = q
                lag_known = True
    held = [math.dist(p, bait) <= 40. for p in positions]

    def clear(a, b):
        fa, fb = to_fixed(a), to_fixed(b)
        if geometry.clear(fa, fb):
            return True
        # Localization/collision can put us in padding. Permit moving out,
        # without crossing an observed wall or decreasing physical clearance.
        x0, y0, x1, y1 = geometry.bounds
        if not (x0 <= fb[0] <= x1 and y0 <= fb[1] <= y1):
            return False
        path = LineString([fa, fb])
        return (not path.intersects(walls) and
                path.distance(walls) >= min(5.01, Point(fa).distance(walls))-1e-6)

    def pred_clear(a, b):
        return predator_geometry.clear(to_fixed(a), to_fixed(b))

    # point, energy, predator points, first action, agent path, predator paths,
    # earliest predicted capture (larger is better), worst separation, cost.
    nodes = [((0., 0.), agent['energy'], forecast, None, [(0., 0.)],
              [[p] for p in forecast], HORIZON+1, math.inf, 0.)]
    expanded = 0
    for depth in range(HORIZON):
        choices = []
        for g, energy, preds, first, path, pred_paths, survival, clearance, spent in nodes:
            goal = next((p for p in waypoints if math.dist(p, g) > 3.), waypoints[-1])
            bearing = math.atan2(goal[1]-g[1], goal[0]-g[0])
            p = preds[target_index]
            away = math.atan2(g[1]-p[1], g[0]-p[0])
            angles = [bearing, bearing-math.pi/4, bearing+math.pi/4,
                      away, away-math.pi/2, away+math.pi/2]
            angles += [i*math.tau/12 for i in range(12)]
            cap = agent['sprint_speed'] if energy >= agent['max_energy']/5 else agent['speed']
            lengths = {min(agent['speed'], cap), cap, (min(agent['speed'], cap)+cap)/2}
            commands = [(0., bearing)]
            for length in sorted(lengths):
                commands += [(min(length, math.dist(goal,g)/modifier) if a == bearing else length, a)
                             for a in angles]
            for length, angle in commands:
                q = (g[0]+length*modifier*math.cos(angle), g[1]+length*modifier*math.sin(angle))
                if length and not clear(g, q):
                    continue
                turn = math.atan2(p[1]-q[1], p[0]-q[0])
                old_heading = 0. if len(path) < 2 else math.atan2(
                    pred_paths[target_index][-2][1]-g[1], pred_paths[target_index][-2][0]-g[0])
                turning = abs(math.atan2(math.sin(turn-old_heading), math.cos(turn-old_heading)))
                cost = .05*min(length,agent['speed'])+.5*max(0.,length-agent['speed'])+turning/math.tau
                cost += .15 + (.01*(agent['age']+.1*depth) if agent['age']+.1*depth >= 60. else 0.)
                if cost >= energy:
                    continue
                new_preds, margin = [], math.inf
                for i, old_p in enumerate(preds):
                    bait_closer = held[i] and math.dist(old_p, bait)+5. < math.dist(old_p,q)
                    destination = bait if bait_closer else q
                    new_p = chase_step(old_p, destination, pred_clear)
                    new_preds.append(new_p)
                    # Include the entire agent move and the subsequent enemy
                    # move, not only distances at the ends of the tick.
                    margin = min(margin, segment_distance(old_p,g,q), segment_distance(q,old_p,new_p))
                required = 21. if lag_known else 33.
                survives = min(survival, depth if margin < required else HORIZON+1)
                new_path = path+[q]
                new_pred_paths = [points+[p] for points,p in zip(pred_paths,new_preds)]
                distance = math.dist(q,new_preds[target_index])
                # The chase estimate faces its target. Hearing contact is the
                # reliable inner region; sight needs an unobstructed segment.
                visible = (distance <= 235. and not walls.intersects(
                    LineString([to_fixed(new_preds[target_index]),to_fixed(q)])))
                lost = distance > 55. and not visible
                remaining = math.dist(q,goal)
                total_cost = spent+cost
                rank = (-survives, -min(clearance,margin) if survives <= depth else 0.,
                        lost, max(0.,distance-235.) if lost else 0., remaining, total_cost)
                first_action = first or dict(move_distance=length,move_direction=angle,turn_angle=turn)
                node = (q,energy-cost,new_preds,first_action,new_path,new_pred_paths,
                        survives,min(clearance,margin),total_cost)
                choices.append((rank,node)); expanded += 1
        if not choices:
            return None
        choices.sort(key=lambda item:item[0])
        # Keep different first moves so the second/third tick can rescue an
        # initially less attractive route around the predator.
        nodes, seen = [], set()
        for _,node in choices:
            first = node[3]
            key = (round(first['move_distance'],2),round(first['move_direction'],3))
            if key in seen:
                continue
            seen.add(key); nodes.append(node)
            if len(nodes) >= BEAM:
                break
    winner = nodes[0]
    return winner[3], dict(horizon_ticks=HORIZON, horizon_seconds=.3,
        predicted_safe=winner[6] > HORIZON-1, minimum_separation=round(winner[7],2),
        observation_lag_estimated=lag_known, expanded=expanded,
        guide_path=winner[4], predator_path=winner[5][target_index],
        note='Observed positions; fastest-chase approximation, unknown rest/terrain/targets.')
