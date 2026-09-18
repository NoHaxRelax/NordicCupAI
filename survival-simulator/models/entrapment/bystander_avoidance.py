"""Keep non-guides outside observed predator sensing and the occupied trap.

Consumes ordinary local observations and an optional locally estimated bait
position. Unknown predators are not available to this filter. This is a local
escape/avoidance layer, not a guarantee against a moving predator seeing us.
"""
import math

from shapely.geometry import LineString, Point

TERRAIN = {'forest': 1., 'grassland': 1., 'desert': .8, 'swamp': .5, 'river': .3}


def avoid_predators(action, state, bait=None):
    predators = [o for o in state['observations'] if o['type'] == 'Predator']
    if not predators and (bait is None or math.hypot(*bait) > 125.):
        return action, False
    walls = [LineString(o['coords']) for o in state['observations'] if o['type'] == 'Edge']
    positions = [(o['distance']*math.cos(o['angle']), o['distance']*math.sin(o['angle']))
                 for o in predators]
    headings = [math.atan2(-p[1], -p[0])-o['rel_dir'] if 'rel_dir' in o else None
                for p, o in zip(positions, predators)]
    cap = state['sprint_speed'] if state['energy'] >= state['max_energy']/5 else state['speed']
    modifier = TERRAIN[state['biome']]
    desired_length = min(cap, action.move_distance)
    desired = (desired_length*modifier*math.cos(action.move_direction),
               desired_length*modifier*math.sin(action.move_direction))

    def assess(length, direction):
        q = (length*modifier*math.cos(direction), length*modifier*math.sin(direction))
        path = LineString([(0., 0.), q]) if length else Point(0., 0.)
        if length and any(path.distance(w) < min(5.05, Point(0., 0.).distance(w))-1e-6 for w in walls):
            return None
        contact = 0.
        hearing = 0.
        vision = 0.
        for p, heading in zip(positions, headings):
            distance = math.dist(p, q)
            # Leave room for one native 15-unit predator move after our step.
            contact = max(contact, max(0., 31.-path.distance(Point(p))))
            hearing = max(hearing, max(0., 76.-distance))
            if distance <= 265.:
                if heading is None:
                    vision = max(vision, 265.-distance)
                else:
                    bearing = math.atan2(q[1]-p[1], q[0]-p[0])-heading
                    bearing = abs(math.atan2(math.sin(bearing), math.cos(bearing)))
                    if bearing < math.radians(40):
                        vision = max(vision, min(265.-distance, distance*(math.radians(40)-bearing)))
        # A retained predator may lie up to 40 units from bait: reserve its
        # 60-unit hearing radius plus margin without needing hidden positions.
        trap = max(0., 105.-math.dist(q, bait)) if bait is not None else 0.
        crosses_trap = bait is not None and path.distance(Point(bait)) < min(105., math.hypot(*bait))-1e-6
        rank = (contact > 0., contact, crosses_trap, max(hearing, trap), vision,
                math.dist(q, desired), length)
        return rank, q

    original = assess(desired_length, action.move_direction)
    if original is not None and not any(original[0][:5]):
        return action, False
    best = None
    for length in sorted({0., min(state['speed'], cap), cap, desired_length}):
        angles = [action.move_direction] if not length else [action.move_direction]+[i*math.tau/48 for i in range(48)]
        for direction in angles:
            value = assess(length, direction)
            if value is not None and (best is None or value[0] < best[0]):
                best = (value[0], length, direction)
    if best is None:
        return action.model_copy(update={'move_distance': 0., 'spawn_agent': False}), True
    _, length, direction = best
    turn = min(predators, key=lambda p: p['distance'])['angle'] if predators else action.turn_angle
    return action.model_copy(update=dict(move_distance=length, move_direction=direction,
                                         turn_angle=turn, spawn_agent=False)), True
