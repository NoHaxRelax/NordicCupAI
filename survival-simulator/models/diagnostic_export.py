"""Outbound policy evidence only. No simulator imports or evaluator inputs."""
import math


def safe(value, depth=0):
    if depth > 9:
        return '<depth limit>'
    if isinstance(value, dict):
        return {str(k): safe(v, depth+1) for k, v in list(value.items())[:256]}
    if isinstance(value, (tuple, list)):
        return [safe(v, depth+1) for v in value[:256]]
    if hasattr(value, 'tolist'):
        return safe(value.tolist(), depth+1)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)[:256]


def export(policy):
    states = dict(phase='exploration' if policy.site is None else 'orchard_and_entrapment',
                  roles=policy.roles, bait=policy.bait, incoming=policy.incoming,
                  decisions=getattr(policy, 'decision_trace', {}),
                  events=policy.events[-16:], estimates={}, orchard={}, guides={})
    for aid in list(policy.roles)[:256]:
        pose = policy.estimator.poses.get(aid)
        if pose is not None:
            states['estimates'][aid] = dict(position=pose.position, heading=pose.heading,
                uncertainty=pose.uncertainty, group=pose.group_id)
        mind = policy.orchard.minds.get(aid)
        if mind is not None:
            states['orchard'][aid] = {key:getattr(mind, key, None) for key in ('fruit', 'post', 'old', 'heir_done')}
    for track in policy.tracks.values():
        if track.guide_id is not None:
            states['guides'][track.guide_id] = dict(track=track.key, debug=track.memory.get('debug'))
    return safe(states)
