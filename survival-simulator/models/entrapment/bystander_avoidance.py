"""Keep non-guides outside observed predator sensing and the occupied trap.

Consumes ordinary local observations and an optional locally estimated bait
position. Unknown predators are not available to this filter. This is a local
escape/avoidance layer, not a guarantee against a moving predator seeing us.
"""
import math

from shapely.geometry import LineString, Point

TERRAIN = {'forest': 1., 'grassland': 1., 'desert': .8, 'swamp': .5, 'river': .3}


def avoid_predators(action, state, bait=None, shared_predators=(), guided_paths=()):
    predators = [o for o in state['observations'] if o['type'] == 'Predator'] + list(shared_predators)
    if not predators and not guided_paths and (bait is None or math.hypot(*bait) > 125.):
        return action, False
    # Native ray casting repeats the same whole wall segment many times.
    # Exact deduplication preserves geometry and avoids repeating every check.
    edge_coords = dict.fromkeys(tuple(tuple(p) for p in o['coords'])
                               for o in state['observations'] if o['type']=='Edge')
    walls = [LineString(coords) for coords in edge_coords]
    positions = [(o['distance']*math.cos(o['angle']), o['distance']*math.sin(o['angle']))
                 for o in predators]
    headings = [math.atan2(-p[1], -p[0])-o['rel_dir'] if 'rel_dir' in o else None
                for p, o in zip(positions, predators)]
    corridors = [LineString(points) for points in guided_paths if len(points) >= 2]
    # Forecasts are traffic hints, not extra simultaneously present predators.
    # Mixing these points into current sightings can make fleeing the far end
    # of a forecast path send an agent straight into the actual predator.
    forecast_positions, forecast_headings = [], []
    for points in guided_paths:
        for previous, point in zip(points, points[1:]):
            if math.dist(previous,point) > .1:
                forecast_positions.append(point)
                forecast_headings.append(math.atan2(point[1]-previous[1],point[0]-previous[0]))
    cap = state['sprint_speed'] if state['energy'] >= state['max_energy']/5 else min(state['speed'],state['sprint_speed'])
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
            # Hearing crosses walls; vision does not.
            sight_blocked = any(LineString([p, q]).intersects(w) for w in walls)
            if distance <= 265. and not sight_blocked:
                if heading is None:
                    vision = max(vision, 265.-distance)
                else:
                    bearing = math.atan2(q[1]-p[1], q[0]-p[0])-heading
                    bearing = abs(math.atan2(math.sin(bearing), math.cos(bearing)))
                    if bearing < math.radians(40):
                        vision = max(vision, min(265.-distance, distance*(math.radians(40)-bearing)))
        forecast_contact = forecast_hearing = forecast_vision = 0.
        for corridor in corridors:
            distance = path.distance(corridor)
            forecast_contact = max(forecast_contact, max(0., 31.-distance))
            forecast_hearing = max(forecast_hearing, max(0., 76.-distance))
        for p, heading in zip(forecast_positions, forecast_headings):
            distance = math.dist(p,q)
            if distance <= 265. and not any(LineString([p,q]).intersects(w) for w in walls):
                bearing = abs(math.atan2(math.sin(math.atan2(q[1]-p[1],q[0]-p[0])-heading),
                                         math.cos(math.atan2(q[1]-p[1],q[0]-p[0])-heading)))
                if bearing < math.radians(40):
                    forecast_vision = max(forecast_vision,min(265.-distance,distance*(math.radians(40)-bearing)))
        # A retained predator may lie up to 40 units from bait: reserve its
        # 60-unit hearing radius plus margin without needing hidden positions.
        trap = max(0., 105.-math.dist(q, bait)) if bait is not None else 0.
        crosses_trap = bait is not None and path.distance(Point(bait)) < min(105., math.hypot(*bait))-1e-6
        # Escape from current sightings has strict priority over the trap's
        # keep-out zone and an uncertain guide corridor. Outside current danger,
        # those forecasts still prevent agents from entering approaching traffic.
        rank = (contact > 0., contact, hearing, vision,
                crosses_trap, trap, forecast_contact, forecast_hearing, forecast_vision,
                math.dist(q, desired), length)
        return rank, q

    original = assess(desired_length, action.move_direction)
    here = assess(0., 0.)
    exposed = here is not None and any(here[0][:4])
    if not exposed and original is not None and not any(original[0][:9]):
        return action, False
    best = None
    lengths = {cap} if exposed else {0., min(state['speed'], cap), cap, desired_length}
    for length in sorted(lengths):
        angles = [action.move_direction] if not length else [action.move_direction]+[i*math.tau/48 for i in range(48)]
        for direction in angles:
            value = assess(length, direction)
            if value is not None and (best is None or value[0] < best[0]):
                best = (value[0], length, direction)
    if best is None:
        return action.model_copy(update={'move_distance': 0.}), True
    _, length, direction = best
    turn = min(predators, key=lambda p: p['distance'])['angle'] if predators else action.turn_angle
    return action.model_copy(update=dict(move_distance=length, move_direction=direction,
                                         turn_angle=turn)), True
