"""Local survival/contact steering using sensed predators and known walls only."""
import math

from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from models.entrapment.guide_pathfinding import RoutePlanner, fixed_frame

HEARING_TARGET = 55.0  # Five units inside the native 60-unit hearing circle.
VISION_TARGET = 235.0
HALF_CONE_TARGET = math.radians(25)  # Native half angle is 30 degrees.
SAFE_DISTANCE = 48.0  # 15 contact + 15 stale step + 15 next step + 3 margin.
BAIT_BUFFER = 5.0
CONTACT_BUFFER = 18.0  # Native contact radius 15 plus three units.
TRAPPED_RADIUS = 40.0
TERRAIN = {'forest': 1., 'grassland': 1., 'desert': .8, 'swamp': .5, 'river': .3}


def prioritize(action, bait, edges, agent, memory, target=None):
    predators = [p for p in agent['observations'] if p['type'] == 'Predator']
    if not predators:
        return action
    target = target if target is not None else min(predators, key=lambda p: p['distance'])
    positions = [(p['distance'] * math.cos(p['angle']),
                  p['distance'] * math.sin(p['angle'])) for p in predators]
    # Observation-only approximation of the held group. Relax clearance only
    # while the bait remains closer by a buffer; never permit physical contact
    # along the proposed move. Incoming predators retain normal clearance.
    required = [min(SAFE_DISTANCE, max(CONTACT_BUFFER, math.dist(other, bait)+BAIT_BUFFER))
                if math.dist(other, bait) <= TRAPPED_RADIUS else SAFE_DISTANCE
                for other in positions]
    p = positions[predators.index(target)]
    # rel_dir is the bearing FROM the predator TO us minus its heading.
    heading = (math.atan2(-p[1], -p[0]) - target['rel_dir']
               if 'rel_dir' in target else None)
    to_fixed, _ = fixed_frame(bait, edges)
    if '_steering_geometry' not in memory:
        fixed_edges = [(to_fixed(a), to_fixed(b)) for a, b in edges]
        memory['_steering_geometry'] = (
            RoutePlanner(fixed_edges, clearance=5.5, exclusion_radius=memory.get('_trap_exclusion_radius', 0.)),
            unary_union([LineString(e) for e in fixed_edges]))
    geometry, walls = memory['_steering_geometry']
    origin = to_fixed((0., 0.))
    modifier = TERRAIN[agent['biome']]
    energy = agent.get('energy', math.inf)
    cap = (agent['speed'] if energy < agent.get('max_energy', 500.) / 5
           else agent['sprint_speed'])
    native_cap = cap
    cap = min(cap, memory.get('_pace_move_cap', cap))
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
    paced_candidates = candidates
    # First evaluate the energy-saving pace. Only if every paced choice is
    # unsafe may steering spend an emergency sprint to preserve the guide.
    phases = [paced_candidates]
    if cap < native_cap:
        phases.append([(native_cap, angle) for angle in angles])
    emergency_override = False
    for phase_index, phase in enumerate(phases):
      if phase_index and best is not None and best[2]['safe']:
        break
      if phase_index:
        emergency_override = True
      for length, angle in phase:
        q = (length * modifier * math.cos(angle), length * modifier * math.sin(angle))
        end = to_fixed(q)
        if length and not geometry.clear(origin, end):
            continue  # Avoid native collision deflections invalidating our choice.
        distances = [math.dist(q, other) for other in positions]
        path = LineString([(0., 0.), q]) if length else Point(0., 0.)
        no_approach = all(path.distance(Point(other)) >= min(math.hypot(*other), margin) - 1e-6
                          for other, margin in zip(positions, required))
        clearance = min(distances)
        turn = math.atan2(p[1] - q[1], p[0] - q[0])
        cost = (min(length, agent['speed']) * .05
                + max(0., length - agent['speed']) * .5
                + min(math.pi, abs(turn)) / math.tau)
        affordable = cost + 1. < energy
        safe = all(d >= margin for d, margin in zip(distances, required)) and no_approach and affordable
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
        # On the final approach, take a safe clear route movement rather than
        # hovering to maintain contact with an already bait-held predator.
        final_approach = math.hypot(*bait) <= HEARING_TARGET + agent['sprint_speed']
        progress = math.dist(q, desired)
        rank = (not affordable, not safe, -clearance if not safe else 0.,
                progress if final_approach else contact_error,
                contact_error if final_approach else progress, cost)
        if best is None or rank < best[0]:
            best = (rank, dict(move_distance=length, move_direction=angle, turn_angle=turn),
                    dict(safe=safe, clearance=round(clearance, 2),
                         relaxed_predators=sum(m < SAFE_DISTANCE for m in required),
                         contact_error=round(contact_error, 2),
                         detectable=contact_error == 0.))
    debug = memory.get('debug')
    if not isinstance(debug, dict):
        debug = {'mode': debug}
        memory['debug'] = debug
    debug['steering'] = best[2]
    debug['steering']['actual_move_cap'] = cap
    debug['steering']['native_move_cap'] = native_cap
    debug['steering']['emergency_sprint_considered'] = emergency_override
    debug['steering']['selected_move_distance'] = best[1]['move_distance']
    # `_guide`'s mode describes its requested pace, before safety steering and
    # the native low-energy walking cap. Show the action actually returned.
    debug['steering'].update(
        selected_move=round(best[1]['move_distance'], 3),
        sprint_selected=best[1]['move_distance'] > agent['speed'],
        movement_cap=cap,
        energy=round(energy, 3),
        low_energy_walk_cap=energy < agent.get('max_energy', 500.) / 5,
    )
    return best[1]
