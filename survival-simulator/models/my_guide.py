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
from models.guide_pathfinding import fixed_frame, navigation_plan
from models.predator_following import predator_is_not_following
from models.guide_steering import prioritize, HEARING_TARGET


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
    action = _guide(bait, edges, agent, context, memory)
    if isinstance(memory.get('debug'), dict) and memory['debug'].get('mode') == 'hold_at_delivery':
        # Intentional handoff: survival steering must not pull us away when
        # the following predator approaches. Being caught here is allowed.
        return action
    return prioritize(action, bait, edges, agent, memory)


def _guide(bait, edges, agent, context, memory):
    """Follow a predator-width A* route while looking at the predator."""
    predators = [o for o in agent['observations'] if o['type'] == 'Predator']
    predator = min(predators, key=lambda o: o['distance']) if predators else None
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

    if (math.hypot(*bait) <= HEARING_TARGET
            or math.hypot(*context['handoff']) <= DELIVERY_ARRIVAL_DISTANCE):
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
    speed = agent['sprint_speed'] if predator['distance'] < 100 else agent['speed']
    # TODO Also don't move if predator is more than 120 away e.g. we can hyperparameterize this.
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
