"""Local survival/contact steering using sensed predators and known walls only."""
import math

from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from models.guide_pathfinding import RoutePlanner, fixed_frame

HEARING_TARGET = 55.0  # Five units inside the native 60-unit hearing circle.
VISION_TARGET = 235.0
HALF_CONE_TARGET = math.radians(25)  # Native half angle is 30 degrees.
SAFE_DISTANCE = 48.0  # 15 contact + 15 stale step + 15 next step + 3 margin.
TERRAIN = {'forest': 1., 'grassland': 1., 'desert': .8, 'swamp': .5, 'river': .3}


def prioritize(action, bait, edges, agent, memory):
    predators = [p for p in agent['observations'] if p['type'] == 'Predator']
    if not predators:
        return action
    target = min(predators, key=lambda p: p['distance'])
    positions = [(p['distance'] * math.cos(p['angle']),
                  p['distance'] * math.sin(p['angle'])) for p in predators]
    p = positions[predators.index(target)]
    # rel_dir is the bearing FROM the predator TO us minus its heading.
    heading = (math.atan2(-p[1], -p[0]) - target['rel_dir']
               if 'rel_dir' in target else None)
    to_fixed, _ = fixed_frame(bait, edges)
    if '_steering_geometry' not in memory:
        fixed_edges = [(to_fixed(a), to_fixed(b)) for a, b in edges]
        memory['_steering_geometry'] = (
            RoutePlanner(fixed_edges, clearance=5.5),
            unary_union([LineString(e) for e in fixed_edges]))
    geometry, walls = memory['_steering_geometry']
    origin = to_fixed((0., 0.))
    modifier = TERRAIN[agent['biome']]
    energy = agent.get('energy', math.inf)
    cap = (agent['speed'] if energy < agent.get('max_energy', 500.) / 5
           else agent['sprint_speed'])
    desired_length = min(action['move_distance'], cap) * modifier
    desired = (desired_length * math.cos(action['move_direction']),
               desired_length * math.sin(action['move_direction']))
    angles = [action['move_direction'], math.atan2(-p[1], -p[0])]
    angles += [i * math.tau / 72 for i in range(72)]
    candidates = [(0., 0.)]
    for length in sorted({min(action['move_distance'], cap),
                          min(agent['speed'] / 2, cap), min(agent['speed'], cap), cap}):
        if length > 0:
            candidates.extend((length, angle) for angle in angles)

    best = None
    for length, angle in candidates:
        q = (length * modifier * math.cos(angle), length * modifier * math.sin(angle))
        end = to_fixed(q)
        if length and not geometry.clear(origin, end):
            continue  # Avoid native collision deflections invalidating our choice.
        distances = [math.dist(q, other) for other in positions]
        path = LineString([(0., 0.), q]) if length else Point(0., 0.)
        no_approach = all(path.distance(Point(other)) >= min(math.hypot(*other), SAFE_DISTANCE) - 1e-6
                          for other in positions)
        clearance = min(distances)
        turn = math.atan2(p[1] - q[1], p[0] - q[0])
        cost = (min(length, agent['speed']) * .05
                + max(0., length - agent['speed']) * .5
                + min(math.pi, abs(turn)) / math.tau)
        affordable = cost + 1. < energy
        safe = clearance >= SAFE_DISTANCE and no_approach and affordable
        distance = math.dist(p, q)
        hearing_error = max(0., distance - HEARING_TARGET)
        vision_error = math.inf
        if heading is not None and not walls.intersects(LineString([to_fixed(p), end])):
            bearing = math.atan2(q[1] - p[1], q[0] - p[0]) - heading
            bearing = abs(math.atan2(math.sin(bearing), math.cos(bearing)))
            vision_error = max(0., distance - VISION_TARGET) + distance * max(0., bearing - HALF_CONE_TARGET)
        contact_error = min(hearing_error, vision_error)
        # Lexicographic priorities: safe first, contact second, route third.
        # If every option is unsafe, maximize separation before other concerns.
        rank = (not affordable, not safe, -clearance if not safe else 0.,
                contact_error, math.dist(q, desired), cost)
        if best is None or rank < best[0]:
            best = (rank, dict(move_distance=length, move_direction=angle, turn_angle=turn),
                    dict(safe=safe, clearance=round(clearance, 2),
                         contact_error=round(contact_error, 2),
                         detectable=contact_error == 0.))
    debug = memory.get('debug')
    if not isinstance(debug, dict):
        debug = {'mode': debug}
        memory['debug'] = debug
    debug['steering'] = best[2]
    return best[1]
