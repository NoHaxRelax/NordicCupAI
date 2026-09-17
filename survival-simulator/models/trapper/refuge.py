"""Refuge flight: a chased agent runs into the nearest narrow gap instead of just away.

A gap passage (11-19 wide) admits an agent but not a predator, so it is a safe
room. The society already decides to flee; this module only redirects that
flight to a gap mouth when one is reachable before the predator closes in, and
walks the agent inside. The first agent inside becomes the station's bait; the
predator that followed it piles up at the mouth and is held.

Feasibility (per tick): the agent must reach the passage before the gap to the
predator closes below contact. A predator charges at 15/tick (11 when tired);
walking costs us 5/tick of gap, sprinting gains 5/tick.
"""
from __future__ import annotations

import math

from .geometry import add, sub, mul, dot, dist, unit, wrap, heading_of, polar, path_clear, los_clear
from .motion import action, step_toward, hold, speed_for
from .paths import plan
from .sites import Site
from .world import WorldState, AgentView, PredatorView, PRED_CHARGE_RANGE

MOUTH_OUT = 45.0        # approach point outside the mouth, on the axis
INSIDE_STEP = 8.0       # spacing of inside slots along the passage


def inside_slot(site: Site, k: int):
    return sub(site.holder, mul(site.normal, INSIDE_STEP * k))


def mouth_choice(world: WorldState, site: Site, held_pids, a: AgentView):
    """Which mouth to use: the front mouth unless predators sit there, else the far
    mouth if clear. Returns (mouth_point, normal_out, is_front) or None."""
    recent = world.recent_predators()
    front_busy = any(dist(p.p, site.front_mid) < 70 for p in recent)
    if not front_busy:
        return site.front_mid, site.normal, True
    if site.far_mouth is not None and site.far_mouth_open:
        far_busy = any(dist(p.p, site.far_mouth) < 70 for p in recent)
        if not far_busy:
            return site.far_mouth, mul(site.normal, -1.0), False
    return None


def plan_refuge(world: WorldState, a: AgentView, p: PredatorView, sites, held_pids):
    """Best reachable gap for this chased agent, or None. Returns (site, waypoints, mouth_is_front, cost)."""
    gap = dist(a.p, p.p)
    walk = a.walk * a.move_modifier
    best = None
    for site in sites:
        if site.kind != 'gap':
            continue
        choice = mouth_choice(world, site, held_pids, a)
        if choice is None:
            continue
        mouth, out, is_front = choice
        approach = add(mouth, mul(out, MOUTH_OUT))
        d_mouth = dist(a.p, mouth)
        if d_mouth > 520:
            continue
        # never run through the predator: the approach point must not be much closer to it than we are
        if dist(approach, p.p) < min(gap, 60.0) - 5:
            continue
        path = plan(world.rects, world.width, world.height, a.p, approach, radius=7.0, slow=world.biome_at)
        if path is None:
            continue
        length = sum(dist(path[i], path[i + 1]) for i in range(len(path) - 1)) + MOUTH_OUT + 10.0
        # time to get inside vs. time for the predator to reach us
        if a.can_sprint:
            speed = a.sprint_speed * a.move_modifier
            budget = (a.energy - a.max_energy / 5) / 5.5      # sprint ticks we can afford above the sprint floor
            ticks_sprint = min(length / max(speed, 1e-6), budget)
            remaining = max(0.0, length - ticks_sprint * speed)
            ticks = ticks_sprint + remaining / max(walk, 1e-6)
            closing = 15.0 * ticks - (ticks_sprint * speed + remaining)   # how much it closes if it charges all along
        else:
            ticks = length / max(walk, 1e-6)
            closing = 15.0 * ticks - length
        # on an estimated world the predator track can be a step or two stale and unseen predators
        # may sit near the mouth: demand a wider margin and a fresh track
        margin = 22.0 if world.complete_map else 45.0
        if not world.complete_map and not getattr(p, 'fresh', True):
            continue
        if gap - closing < margin and not (p.resting or p.still_ticks >= 3):
            continue
        cost = length + (0.0 if is_front else 60.0)
        # a long run is only worth it when the predator is close; otherwise let the society flee
        if cost > 300.0 and gap > 80.0:
            continue
        if best is None or cost < best[3]:
            best = (site, path[1:] + [mouth], is_front, cost)
    return best


class Refugee:
    """Per-tick action while running to and entering a gap."""

    def __init__(self, world: WorldState):
        self.world = world

    def act(self, a: AgentView, p: PredatorView | None, site: Site, waypoints, slot_k, occupied=False, side=None):
        """``occupied``: a bait already stands at the holder, so we cannot enter from the front:
        at the flyby point we sprint along ``side`` instead and the predator takes the bait."""
        w = self.world
        goal_inside = inside_slot(site, slot_k)
        if occupied and side is not None:
            out = dot(sub(a.p, site.front_mid), site.normal)
            lateral = abs(dot(sub(a.p, site.front_mid), (-site.normal[1], site.normal[0])))
            gap = dist(a.p, p.p) if p is not None else 999.0
            if out <= 16.0 and lateral < 40.0 and (gap <= 45.0 or lateral > 6.0):
                return step_toward(a, add(a.p, mul(side, 40.0)), speed_for(a, True)), 'refuge: flyby sprint', True
            if out <= 16.0 and lateral <= 6.0:
                return action(a.id, 0.0, 0.0, wrap(heading_of(sub(p.p, a.p)) - a.heading)) if p else hold(a), f'refuge: at the flyby point, gap {gap:.0f}', False
            waypoints[-1] = site.front
        # inside the passage already: walk to our slot and stop
        axis_in = mul(site.normal, -1.0)
        along = dot(sub(a.p, site.front_mid), axis_in)
        lateral = abs(dot(sub(a.p, site.front_mid), (-site.normal[1], site.normal[0])))
        inside = -0.5 <= along <= site.length + 0.5 and lateral <= site.thickness / 2 - 4.0
        if inside:
            if dist(a.p, goal_inside) < 0.6:
                return hold(a), 'refuge: inside, holding', True
            return step_toward(a, goal_inside, min(a.walk * a.move_modifier, dist(a.p, goal_inside)), face=goal_inside), 'refuge: settling inside', False
        gap = dist(a.p, p.p) if p is not None else 999.0
        while len(waypoints) > 1 and dist(a.p, waypoints[0]) < 6.0:
            waypoints.pop(0)
        wp = waypoints[0]
        sprint = p is not None and gap < 75 and a.can_sprint
        speed = speed_for(a, sprint)
        if len(waypoints) == 1:
            # final approach: aim exactly at the mouth centre line and cross it
            speed = min(speed, max(dist(a.p, wp) + 2.0, 4.0))
        act = step_toward(a, wp, speed, face=wp)
        return act, f'refuge: to {"mouth" if len(waypoints) == 1 else "approach"}, gap {gap:.0f}', False
