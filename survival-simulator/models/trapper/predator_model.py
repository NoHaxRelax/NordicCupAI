"""Exact clone of the engine's predator: what it observes, what it decides, how it moves.

Mirrors ``src/elements/predator.py`` (decision), ``src/elements/creature.py``
(observation) and ``src/elements/environment.py`` (movement, collision, bounds)
at commit acfc31a4. Used to predict a predator's next positions and to verify
which agent it targets. ``validate`` in scripts/trapper checks this against the
engine tick by tick.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from shapely.geometry import Point, Polygon

from .geometry import Rect, wrap
from .world import (PRED_CONE, PRED_HEARING, PRED_MAX_ENERGY, PREDATOR_RADIUS,
                    PRED_SPRINT, PRED_VISION, PRED_WALK)

CHUNK = 400


@dataclass
class PredState:
    x: float
    y: float
    heading: float
    energy: float = 102.0
    resting: bool = False
    speed: float = PRED_WALK
    sprint_speed: float = PRED_SPRINT
    size: float = PREDATOR_RADIUS
    max_energy: float = PRED_MAX_ENERGY
    hearing: float = PRED_HEARING
    vision: float = PRED_VISION
    cone: float = PRED_CONE

    @property
    def p(self):
        return (self.x, self.y)


def neighbouring_chunks(x, y):
    cx, cy = int(x // CHUNK), int(y // CHUNK)
    return {(cx + dx, cy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}


def local_edges(rects, x, y):
    """Edges of every rectangle that spans a chunk neighbouring (x, y), as the engine's grid does."""
    chunks = neighbouring_chunks(x, y)
    out = []
    for r in rects:
        for (a, b) in r.edges():
            min_c = (int(a[0] // CHUNK), int(a[1] // CHUNK))
            max_c = (int(b[0] // CHUNK), int(b[1] // CHUNK))
            hit = False
            for cx in range(min_c[0], max_c[0] + 1):
                for cy in range(min_c[1], max_c[1] + 1):
                    if (cx, cy) in chunks:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                out.append((a, b))
    return out


def local_rects(rects, x, y):
    chunks = neighbouring_chunks(x, y)
    out = []
    for r in rects:
        min_c = (int(r.x // CHUNK), int(r.y // CHUNK))
        max_c = (int(r.x2 // CHUNK), int(r.y2 // CHUNK))
        if any((cx, cy) in chunks for cx in range(min_c[0], max_c[0] + 1) for cy in range(min_c[1], max_c[1] + 1)):
            out.append(r)
    return out


def _compute_visibility(x, y, direction, cone, vision, edges):
    from src.utils.sensing import compute_visibility   # vendored engine function
    return compute_visibility(x, y, direction, cone, vision, edges)


def observe(pred: PredState, agents, edges):
    """Engine ``Creature.observe`` for a predator.

    ``agents`` is a list of (id, x, y, heading) tuples for agents in the
    neighbouring chunks (the caller filters, or passes all: extra far agents
    never pass the range tests). ``edges`` are the local edges.
    Returns the observation list with the same dict shapes as the engine.
    """
    observations = []
    vision_poly, hit_edges = _compute_visibility(pred.x, pred.y, pred.heading, pred.cone, pred.vision, edges)
    if agents:
        xs = np.fromiter((a[1] for a in agents), float)
        ys = np.fromiter((a[2] for a in agents), float)
        dx, dy = xs - pred.x, ys - pred.y
        distances = np.hypot(dx, dy)
        angles = (np.arctan2(dy, dx) - pred.heading + np.pi) % (2 * np.pi) - np.pi
        half = pred.cone / 2
        nearby = distances <= pred.hearing
        visible = (~nearby) & (distances <= pred.vision) & (np.abs(angles) <= half)
        for idx in np.where(nearby)[0]:
            a = agents[idx]
            rel = ((math.atan2(pred.y - a[2], pred.x - a[1]) - a[3] + math.pi) % (2 * math.pi)) - math.pi
            observations.append({"type": "Agent", "distance": float(distances[idx]), "angle": float(angles[idx]),
                                 "rel_dir": float(rel), "id": a[0]})
        cand = np.where(visible)[0]
        if len(cand):
            shape = Polygon([(pred.x, pred.y)] + list(vision_poly))
            for idx in cand:
                a = agents[idx]
                if shape.contains(Point(a[1], a[2])):
                    rel = ((math.atan2(pred.y - a[2], pred.x - a[1]) - a[3] + math.pi) % (2 * math.pi)) - math.pi
                    observations.append({"type": "Agent", "distance": float(distances[idx]), "angle": float(angles[idx]),
                                         "rel_dir": float(rel), "id": a[0]})
    cos_d, sin_d = math.cos(-pred.heading), math.sin(-pred.heading)
    for (sx, sy), (ex, ey) in hit_edges:
        dxs, dys, dxe, dye = sx - pred.x, sy - pred.y, ex - pred.x, ey - pred.y
        observations.append({"type": "Edge", "coords": ((dxs * cos_d - dys * sin_d, dxs * sin_d + dys * cos_d),
                                                        (dxe * cos_d - dye * sin_d, dxe * sin_d + dye * cos_d))})
    return observations


def target_id(observation):
    """Which agent the predator will chase this tick (nearest observed), or None."""
    agents = [o for o in observation if o.get("type") == "Agent"]
    if not agents:
        return None
    return min(agents, key=lambda o: o["distance"])["id"]


def decide(pred: PredState, observation, rng=None):
    """Engine ``Predator.step``. Returns (signals, mode) with mode in
    {'charge', 'pivot', 'edge', 'wander'}. Wander needs the engine rng; without it
    the turn is 0 (unknown small random turn)."""
    agents = [o for o in observation if o.get("type") == "Agent"]
    edges = [o["coords"] for o in observation if o.get("type") == "Edge"]
    signals = {}
    if agents:
        closest = min(agents, key=lambda f: f["distance"])
        d = closest["distance"]
        ang = closest["angle"]
        look = closest["rel_dir"]
        if abs(look) > np.pi / 2 or d < pred.hearing * 1.5:
            turn = max(-0.3, min(0.3, ang * 0.5))
            if abs(ang) > 0.05:
                signals["turn"] = turn
                signals["move"] = min(pred.sprint_speed, d)
                signals["direction"] = turn
            else:
                signals["move"] = min(pred.sprint_speed, d)
                signals["direction"] = ang
            return signals, "charge"
        pivot = -np.sign(look)
        move_dir = ang + pivot * np.pi / 4
        signals["move"] = pred.sprint_speed
        signals["direction"] = move_dir
        dx_m = pred.sprint_speed * np.cos(move_dir)
        dy_m = pred.sprint_speed * np.sin(move_dir)
        x_a = d * np.cos(ang)
        y_a = d * np.sin(ang)
        signals["turn"] = float(np.arctan2(y_a - dy_m, x_a - dx_m))
        return signals, "pivot"
    if edges:
        def closest_point(edge):
            (x1, y1), (x2, y2) = edge
            dx, dy = x2 - x1, y2 - y1
            t = (-(x1 * dx + y1 * dy)) / (dx * dx + dy * dy)
            t = max(0.0, min(1.0, t))
            return (x1 + t * dx, y1 + t * dy)
        pts = [closest_point(e) for e in edges]
        c = min(pts, key=lambda p: np.hypot(p[0], p[1]))
        dist_c = max(np.hypot(c[0], c[1]) - pred.size, 2.0)
        a = np.arctan2(c[1], c[0])
        a = (a + np.pi) % (2 * np.pi) - np.pi
        turn = -np.pi / dist_c if a > 0 else np.pi / dist_c
        signals["turn"] = float(turn)
        signals["move"] = pred.speed
        signals["direction"] = float(turn)
        return signals, "edge"
    turn = rng.uniform(-0.1, 0.1) if rng is not None else 0.0
    signals["turn"] = turn
    signals["move"] = pred.speed
    signals["direction"] = None
    return signals, "wander"


def in_obstacle(p, radius, rects):
    return any(r.contains(p, radius) for r in rects)


def move(pred: PredState, distance, direction, modifier, rects, width, height):
    """Engine ``update_entity_position``: energy, cap, biome modifier, collision deflection, bounds.
    ``rects`` must be the local rectangles (neighbouring chunks of the pre-move position)."""
    if distance < 0:
        distance = 0.0
    if distance > pred.sprint_speed:
        distance = pred.sprint_speed
    if pred.energy < pred.max_energy / 5 and distance > pred.speed:
        distance = pred.speed
    if distance <= pred.speed:
        energy = pred.energy - distance * 0.05
    else:
        energy = pred.energy - (pred.speed * 0.05 + (distance - pred.speed) * 0.5)
    heading = pred.heading if direction is None else pred.heading + direction
    distance *= modifier
    px, py = pred.x, pred.y
    nx, ny = px + distance * math.cos(heading), py + distance * math.sin(heading)
    if in_obstacle((nx, ny), pred.size, rects):
        step = math.pi / 18
        placed = False
        for i in range(36):
            test = heading + step * ((i + 1) // 2) * (-1) ** i
            tx, ty = px + distance * math.cos(test), py + distance * math.sin(test)
            if not in_obstacle((tx, ty), pred.size, rects):
                nx, ny = tx, ty
                placed = True
                break
        if not placed:
            nx, ny = px, py
    nx = max(pred.size, min(width - pred.size, nx))
    ny = max(pred.size, min(height - pred.size, ny))
    return replace(pred, x=nx, y=ny, energy=energy)


def turn(pred: PredState, angle):
    return replace(pred, heading=pred.heading + angle, energy=pred.energy - min(math.pi, abs(angle)) / (2 * math.pi))


def step(pred: PredState, agents, rects, width, height, modifier_at, rng=None):
    """One full predator tick, including rest/wake. ``agents``: (id, x, y, heading) after
    their own move this tick. Returns (new_state, info) where info has target, mode,
    kills (agent ids within contact) and signals."""
    info = dict(target=None, mode=None, kills=[], signals=None)
    if pred.resting:
        if pred.energy > pred.max_energy * 0.5:
            pred = replace(pred, resting=False)
        else:
            return replace(pred, energy=pred.energy + 0.1 * 30), dict(info, mode="rest")
    lrects = local_rects(rects, pred.x, pred.y)
    edges = local_edges(rects, pred.x, pred.y)
    chunks = neighbouring_chunks(pred.x, pred.y)
    near = [a for a in agents if (int(a[1] // CHUNK), int(a[2] // CHUNK)) in chunks]
    obs = observe(pred, near, edges)
    info["target"] = target_id(obs)
    signals, mode = decide(pred, obs, rng)
    info["mode"] = mode
    info["signals"] = signals
    modifier = modifier_at((pred.x, pred.y))
    if "move" in signals:
        pred = move(pred, signals["move"], signals["direction"], modifier, lrects, width, height)
    if "turn" in signals:
        pred = turn(pred, signals["turn"])
    chunks = neighbouring_chunks(pred.x, pred.y)
    for a in agents:
        if (int(a[1] // CHUNK), int(a[2] // CHUNK)) in chunks and math.hypot(a[1] - pred.x, a[2] - pred.y) < pred.size + 5.0:
            info["kills"].append(a[0])
    if pred.energy <= 0:
        pred = replace(pred, resting=True)
    return pred, info


def rollout(pred: PredState, agent_plan, rects, width, height, modifier_at, ticks):
    """Predict ``ticks`` predator steps against ``agent_plan(t) -> [(id,x,y,heading)]``.
    Wander turns are unknown and taken as zero."""
    trace = []
    for t in range(ticks):
        pred, info = step(pred, agent_plan(t), rects, width, height, modifier_at)
        trace.append((pred, info))
    return trace


def sees(pred: PredState, agent_xyh, rects):
    """Whether the predator currently observes the agent (hearing or unobstructed cone)."""
    edges = local_edges(rects, pred.x, pred.y)
    obs = observe(pred, [(0, agent_xyh[0], agent_xyh[1], agent_xyh[2])], edges)
    return any(o.get("type") == "Agent" for o in obs)


def observed_agents(pred: PredState, agents, rects):
    """Ids of every agent the predator observes, nearest first."""
    edges = local_edges(rects, pred.x, pred.y)
    chunks = neighbouring_chunks(pred.x, pred.y)
    near = [a for a in agents if (int(a[1] // CHUNK), int(a[2] // CHUNK)) in chunks]
    obs = [o for o in observe(pred, near, edges) if o.get("type") == "Agent"]
    return [o["id"] for o in sorted(obs, key=lambda o: o["distance"])]
