"""Agent motion primitives: turn an intended world-frame move into an action.

Engine order per tick: move (direction relative to the current heading), then
turn. A requested distance is multiplied by the biome modifier of the starting
pixel; we compensate so ``goto`` covers real ground.
"""
from __future__ import annotations

import math

from .geometry import wrap, sub, dist, heading_of, add, mul, unit
from .world import AgentView


def action(agent_id, move=0.0, direction=0.0, turn=0.0, spawn=False):
    return dict(agent_id=agent_id, move_distance=float(move), move_direction=float(direction),
                turn_angle=float(turn), spawn_agent=bool(spawn))


def step_toward(a: AgentView, target, real_distance, face=None):
    """Move ``real_distance`` ground units toward ``target`` (capped by sprint speed and energy)
    while turning to face ``face`` (a world point) or the travel direction."""
    d = dist(a.p, target)
    if d < 1e-6:
        real_distance = 0.0
    real_distance = min(real_distance, d)
    requested = real_distance / max(a.move_modifier, 1e-6)
    cap = a.sprint_speed if a.can_sprint else a.walk
    requested = min(requested, cap)
    direction = wrap(heading_of(sub(target, a.p)) - a.heading) if requested > 0 else 0.0
    if face is None:
        turn = direction if requested > 0 else 0.0
    else:
        turn = wrap(heading_of(sub(face, a.p)) - a.heading)
    return action(a.id, requested, direction, turn)


def face_point(a: AgentView, point, offset=0.0):
    """Turn only, to face a point with an optional angular offset."""
    return action(a.id, 0.0, 0.0, wrap(heading_of(sub(point, a.p)) + offset - a.heading))


def hold(a: AgentView):
    return action(a.id)


def speed_for(a: AgentView, want_sprint):
    """Ground speed available this tick: sprint when allowed and wanted, else walk."""
    if want_sprint and a.can_sprint:
        return a.sprint_speed * a.move_modifier
    return a.walk * a.move_modifier


def next_waypoint(p, waypoints, reach=8.0):
    """Drop reached waypoints; return the current target or None."""
    while waypoints and dist(p, waypoints[0]) <= reach and len(waypoints) > 1:
        waypoints.pop(0)
    return waypoints[0] if waypoints else None
