"""Check whether a predator sighting supports being its detectable prey.

This is public sensing geometry, not access to the predator's target ID. The
sighting is delayed by a movement, so the next guide tick must still verify
following and protect against capture.
"""
import math
from shapely.geometry import LineString


def detectable_observer(observation, state, edges):
    distance = observation['distance']
    if distance <= 60.: return True
    heading_error = observation.get('rel_dir')
    if heading_error is None or distance > 250.: return False
    if abs(math.atan2(math.sin(heading_error),math.cos(heading_error))) > math.pi/6:
        return False
    # Outside our hearing radius the sighting itself establishes clear LOS.
    if distance > state['hearing_radius']: return True
    point = (distance*math.cos(observation['angle']),distance*math.sin(observation['angle']))
    sight = LineString([(0.,0.),point])
    return not any(sight.intersects(LineString(edge)) for edge in edges)


def recover_guide_contacts(policy, states, to_local):
    """Repair a lost assignment when a guide visibly encounters another track.

    Positions and ownership are inferred from ordinary shared sightings. Only
    isolated, detectable contacts well outside bait sensing qualify. An active
    owner gives way only to an observer clearly closer despite sighting delay.
    """
    released = set()
    for lost in list(policy.tracks.values()):
        aid = lost.guide_id
        if (aid not in states or aid in lost.observers or aid in released
                or lost.group != policy.site_group or lost.completed_at is not None):
            continue
        state = states[aid]
        pose = policy.estimator.poses[aid]
        if pose.group_id != lost.group or pose.uncertainty > 8.:
            continue
        # A rescue guide must still be able to sprint after this decision.
        if state['energy'] < state['max_energy']/5 + .5*state['sprint_speed'] + 1.:
            continue
        sightings = [o for o in state['observations'] if o['type']=='Predator']
        if len(sightings)!=1:
            continue
        sighting = sightings[0]
        p = (sighting['distance']*math.cos(sighting['angle']),
             sighting['distance']*math.sin(sighting['angle']))
        # A possibly held predator must not pull a second guide into the crowd.
        if math.dist(p,to_local(pose,policy.site['goal'])) <= 60.+15.+pose.uncertainty:
            continue
        edges = [(to_local(pose,e.start),to_local(pose,e.end))
                 for e in policy.estimator.groups[pose.group_id].edges]
        if not detectable_observer(sighting,state,edges):
            continue
        recipients = [t for t in policy.tracks.values() if t is not lost
                      and t.group==lost.group and aid in t.observers
                      and t.observers[aid]==sighting and t.completed_at is None]
        if len(recipients)!=1:
            continue
        recipient = recipients[0]
        previous = recipient.guide_id
        if previous in states:
            owner_pose = policy.estimator.poses[previous]
            owner_sighting = recipient.observers.get(previous)
            if (owner_sighting is None or owner_pose.group_id!=pose.group_id
                    or owner_pose.uncertainty>8.):
                continue
            margin = 2*15.+pose.uncertainty+owner_pose.uncertainty
            if sighting['distance']+margin >= owner_sighting['distance']:
                continue
            released.add(previous)
            policy.metrics['guide_handovers'] += 1
        lost.guide_id = None
        lost.memory = {}
        lost.completed_at = None
        recipient.guide_id = aid
        recipient.memory = {}
        recipient.edges = [(tuple(e.start),tuple(e.end))
                           for e in policy.estimator.groups[recipient.group].edges]
        recipient.guide_last_saw = policy.now
        policy.metrics['guide_contact_recoveries'] = policy.metrics.get('guide_contact_recoveries',0)+1
        policy.event('guide_contact_recovered',agent=aid,previous=previous,
                     old_track=lost.key,track=recipient.key,
                     observed_distance=sighting['distance'],
                     reason='sole_detectable_contact_closer_than_current_owner')
    return released
