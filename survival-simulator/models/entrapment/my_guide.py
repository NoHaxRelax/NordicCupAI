"""Edit guide() here, then rerun scripts/guide_lab.py with the same seed.

All coordinates are relative to YOUR agent's current position and heading:
  (0, 0) = you; +x = forward; +y = right; angles are radians, clockwise.
The local frame rotates when you turn. Do not retain local points in memory
across ticks without transforming them. Memory is reset for every new run.

bait: actual stationary bait centre, in your local frame.
edges: every map edge as ((start_x, start_y), (end_x, end_y)), same frame.
agent: ordinary simulator status/observations (energy, speed, biome, etc.).
context: tick, dt, and a local handoff point outside the bait's gap.
memory: your persistent dictionary; memory['debug'] is shown in the replay.

The harness supplies perfect static-map localization for bait/edges/handoff.
Predator positions, energy, targets, and resting state are NOT supplied except
for what your agent actually senses through agent['observations'].

Return move_distance, move_direction, turn_angle. No agent ID is needed.
Movement happens BEFORE turning. move_direction is relative to the OLD heading:
  0 = forward; pi = backward; pi/2 = right; -pi/2 = left.
distance <= agent['speed'] walks; greater values sprint, up to sprint_speed.
You may call breakpoint() here, or use the runner's --break-at TICK option.
"""

import math
from shapely.geometry import LineString
from shapely.ops import unary_union
from models.entrapment.guide_pathfinding import fixed_frame, navigation_plan
from models.entrapment.predator_following import predator_is_not_following
from models.entrapment.guide_steering import prioritize, HEARING_TARGET


LOST_WAIT_TICKS = 5  # Hold for 0.5 seconds before returning to last contact.
REACQUIRE_ARRIVAL_DISTANCE = 12.0
DELIVERY_ARRIVAL_DISTANCE = 1.0


def _walking_action(plan, agent, turn):
    modifier = {'forest': 1., 'grassland': 1., 'desert': .8, 'swamp': .5, 'river': .3}[agent['biome']]
    return dict(move_distance=min(agent['speed'], plan['dist_to_point']/modifier),
                move_direction=plan['dir'], turn_angle=turn)


def _return_to_predator(bait, edges, agent, memory, to_local, turn):
    memory['_lost_ticks'] = memory.get('_lost_ticks', 0) + 1
    lost_ticks = memory['_lost_ticks']
    last = memory.get('_last_predator_fixed')
    if last is None:
        memory['debug'] = dict(mode='wait_without_previous_contact', lost_ticks=lost_ticks)
        return dict(move_distance=0., move_direction=0., turn_angle=turn)
    target = to_local(last)
    if lost_ticks <= LOST_WAIT_TICKS:
        # Preserve the requested short wait before committing to recovery.
        memory['debug'] = dict(mode='wait_for_predator', lost_ticks=lost_ticks,
                               wait_ticks=LOST_WAIT_TICKS, last_predator=target)
        return dict(move_distance=0., move_direction=0.,
                    turn_angle=turn)

    # Separate route cache keeps the delivery route available after reacquisition.
    recovery = memory.setdefault('_recovery_navigation', {})
    distance = math.hypot(*target)
    if distance <= REACQUIRE_ARRIVAL_DISTANCE:
        memory['debug'] = dict(mode='wait_at_last_predator_position',
                               lost_ticks=lost_ticks, last_predator=target)
        return dict(move_distance=0., move_direction=0., turn_angle=turn)
    plan = navigation_plan(bait, edges, target, recovery)
    fallback = False
    if plan is None:
        # The predator may have stood within our conservative wall margin.
        # Return to where we last actually sensed it instead of crossing a wall.
        target = to_local(memory['_last_contact_position_fixed'])
        plan = navigation_plan(bait, edges, target, recovery)
        fallback = True
    if plan is None or plan['dist'] <= REACQUIRE_ARRIVAL_DISTANCE:
        memory['debug'] = dict(mode='wait_recovery_blocked_or_arrived',
                               lost_ticks=lost_ticks, last_predator=to_local(last))
        return dict(move_distance=0., move_direction=0., turn_angle=turn)

    last_local = to_local(last)
    memory['debug'] = dict(mode='return_to_last_contact' if fallback else 'return_to_predator',
                           lost_ticks=lost_ticks, last_predator=last_local, route=plan)
    return _walking_action(plan, agent, turn)

def guide(bait, edges, agent, context, memory):
    """Guide safely en route; stand at delivery while the predator follows."""
    context = dict(context)
    if 'target_predator' not in context:
        context['target_predator'] = _select_target(bait, edges, agent, context, memory)
    if context.get('vision_delivery') and 'mouth' in context:
        mouth = context['mouth']
        inward = (bait[0]-mouth[0], bait[1]-mouth[1])
        depth = max(1e-9, math.hypot(*inward))
        inward = tuple(v/depth for v in inward)
        cross = (-inward[1], inward[0])
        offset = sum((context['handoff'][k]-mouth[k])*cross[k] for k in range(2))
        context['handoff'] = tuple(mouth[k]-90.*inward[k]+offset*cross[k] for k in range(2))
        memory['_trap_exclusion_radius'] = 85.
    action = _guide(bait, edges, agent, context, memory)
    if isinstance(memory.get('debug'), dict) and memory['debug'].get('mode') == 'hold_at_delivery':
        # Intentional handoff: survival steering must not pull us away when
        # the following predator approaches. Being caught here is allowed.
        return action
    return prioritize(action, bait, edges, agent, memory, target=context['target_predator'])


def _bait_visible_for_handoff(predator, bait, edges, memory):
    """Conservative sight alignment from ordinary bearing/heading observations."""
    p = (predator['distance']*math.cos(predator['angle']),
         predator['distance']*math.sin(predator['angle']))
    distance = math.dist(p, bait)
    if distance <= 55.:
        return True
    if distance > 230. or 'rel_dir' not in predator:
        return False
    toward_guide = math.atan2(-p[1], -p[0])
    heading = toward_guide-predator['rel_dir']
    toward_bait = math.atan2(bait[1]-p[1], bait[0]-p[0])
    wrap = lambda a: abs(math.atan2(math.sin(a), math.cos(a)))
    if max(wrap(heading-toward_bait),wrap(toward_guide-toward_bait)) > math.radians(5):
        return False
    to_fixed, _ = fixed_frame(bait, edges)
    if '_handoff_visibility_walls' not in memory:
        memory['_handoff_visibility_walls'] = unary_union([
            LineString([to_fixed(a),to_fixed(b)]) for a,b in edges])
    return not memory['_handoff_visibility_walls'].intersects(LineString([to_fixed(p),to_fixed(bait)]))


def _select_target(bait, edges, agent, context, memory):
    """Associate ordinary sightings by motion; never acquire the held crowd.

    Native observations have no predator IDs. Ambiguous overlap can therefore
    never establish identity, but a nearby held predator must not replace a
    missing newcomer simply because it is closest to the guide.
    """
    observations = [o for o in agent['observations'] if o['type'] == 'Predator']
    to_fixed, _ = fixed_frame(bait, edges)
    samples = [(o, to_fixed((o['distance'] * math.cos(o['angle']),
                            o['distance'] * math.sin(o['angle'])))) for o in observations]
    tick = context.get('tick', 0)
    previous = memory.get('_target_track')
    if previous is None:
        choices = [(o, p) for o, p in samples if math.hypot(*p) > 40.]
        selected = min(choices, key=lambda row: row[0]['distance'], default=None)
    else:
        elapsed = max(1, tick - previous['tick'])
        choices = [(o, p) for o, p in samples
                   if math.dist(p, previous['position']) <= 15.75 * min(elapsed, 5)
                   and (math.hypot(*p) > 40. or math.hypot(*previous['position']) <= 55.)]
        selected = min(choices, key=lambda row: math.dist(row[1], previous['position']), default=None)
    if selected is None:
        return None
    observation, position = selected
    memory['_target_track'] = dict(position=position, tick=tick)
    return observation

def _guide(bait, edges, agent, context, memory):
    """Follow a predator-width A* route while looking at the predator."""
    predators = [o for o in agent['observations'] if o['type'] == 'Predator']
    # Colony coordinator may associate a particular ordinary observation with
    # this guide. Omitted context preserves the evaluated single-guide policy.
    predator = (context['target_predator'] if 'target_predator' in context else
                min(predators, key=lambda o: o['distance']) if predators else None)
    # Keep the current heading until contact; track the observed bearing in all modes.
    turn = predator['angle'] if predator is not None else 0.
    to_fixed, to_local = fixed_frame(bait, edges)
    not_following = predator_is_not_following(predator, bait, edges, agent, context, memory)

    # Save only ordinary observations, in the static frame rather than the
    # guide's rotating local frame. A fresh sighting resumes delivery immediately.
    if predator is not None:
        memory['_last_predator_fixed'] = to_fixed((predator['distance']*math.cos(predator['angle']),
                                                   predator['distance']*math.sin(predator['angle'])))
        memory['_last_contact_position_fixed'] = to_fixed((0., 0.))
    if predator is None or not_following:
        action = _return_to_predator(bait, edges, agent, memory, to_local, turn)
        memory['debug']['following_check'] = memory['_following_debug']
        return action
    if memory.get('_lost_ticks', 0):
        memory['_reacquisitions'] = memory.get('_reacquisitions', 0) + 1
        memory.pop('_recovery_navigation', None)
    memory['_lost_ticks'] = 0

    delivery_ready = (math.hypot(*bait) <= HEARING_TARGET
                      or math.hypot(*context['handoff']) <= DELIVERY_ARRIVAL_DISTANCE)
    if context.get('vision_delivery'):
        delivery_ready = (math.hypot(*context['handoff']) <= 2.
                          and _bait_visible_for_handoff(predator, bait, edges, memory))
    if delivery_ready:
        memory['debug'] = dict(mode='hold_at_delivery',
                               handoff_distance=math.hypot(*context['handoff']),
                               bait_distance=math.hypot(*bait),
                               predator_distance=predator['distance'],
                               following_check=memory['_following_debug'])
        return dict(move_distance=0., move_direction=0., turn_angle=turn)

    # A* routes around edges to the handoff, with predator radius + margin.
    # The cached plan survives translation/rotation of the local input frame.
    guide_plan = navigation_plan(bait, edges, context['handoff'], memory)
    if guide_plan is None:
        memory['debug'] = 'No predator-width route found; hold and watch.'
        return dict(move_distance=0., move_direction=0., turn_angle=turn)
    distance_to_target = guide_plan['dist']

    # Look at the predator
    preferred_min,preferred_max = memory.get('_preferred_predator_distance',(100.,120.))
    speed = (agent['sprint_speed'] if predator['distance'] < preferred_min else
             agent['speed'] if predator['distance'] <= preferred_max else 0.)
    memory['debug'] = {
        'predator_distance': round(predator['distance'], 2),
        'handoff_distance': round(distance_to_target, 2),
        'mode': 'sprint' if speed > agent['speed'] else 'walk',
        'route': guide_plan,
        'reacquisitions': memory.get('_reacquisitions', 0),
        'following_check': memory['_following_debug'],
    }
    # Path lengths are physical units; native move_distance is charged before
    # terrain slowing. Stop at the waypoint without overshooting it.
    terrain = {'forest': 1., 'grassland': 1., 'desert': .8, 'swamp': .5, 'river': .3}
    modifier = terrain[agent['biome']]
    return dict(
        move_distance=min(speed, guide_plan['dist_to_point']/modifier),
        move_direction=guide_plan['dir'],
        turn_angle=turn,
    )
