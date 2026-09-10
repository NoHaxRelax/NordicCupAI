"""A hand-written survival policy over the captured payload.

Three jobs: the bar PPO must beat, a behaviour-cloning warm-start if we want
one, and the fallback we serve on day one if the trained policy misreads the
real environment. It reads the same JSON the endpoint receives, so it needs no
adapter.

    from survivalsim.heuristic import heuristic_action
    action = heuristic_action(agent_status_dict)      # -> one of ACTIONS
"""

from __future__ import annotations

import math

from .env import ACTIONS

# Tunables. The names say what they mean; the values are guesses for the real
# environment and are exactly what a day-one fit should adjust first.
FLEE_DIST = 18.0        # a predator closer than this drives the decision
FACING_TOL = 0.9        # rad: predator counts as "looking at us" inside this
EAT_REACH = 3.0         # our reconstruction's default tree reach
HUNGRY_FRAC = 0.85      # seek food below this energy fraction
TURN_TOL = 0.25         # rad: aligned enough to walk straight


def _turn_toward(angle: float) -> str:
    if abs(angle) <= TURN_TOL:
        return "FORWARD"
    return "LEFT" if angle > 0 else "RIGHT"


def _turn_away(angle: float, sprint: bool) -> str:
    """Face the opposite bearing, then run."""
    away = math.atan2(math.sin(angle + math.pi), math.cos(angle + math.pi))
    if abs(away) <= TURN_TOL * 2:
        return "SPRINT" if sprint else "FORWARD"
    return "LEFT" if away > 0 else "RIGHT"


def heuristic_action(a: dict) -> str:
    obs = a.get("observations", [])
    energy = float(a.get("energy", 0.0))
    max_e = max(float(a.get("max_energy", 1.0)), 1e-6)
    frac = energy / max_e

    trees = [e for e in obs if e.get("type") == "tree" and "distance" in e]
    preds = [e for e in obs if e.get("type") == "predator" and "distance" in e]

    # 1. A predator that is close AND facing us is the only emergency.
    if preds:
        p = min(preds, key=lambda e: e["distance"])
        facing_us = abs(math.atan2(math.sin(p.get("rel_dir", 0.0) - p["angle"] - math.pi),
                                   math.cos(p.get("rel_dir", 0.0) - p["angle"] - math.pi))) < FACING_TOL
        if p["distance"] < FLEE_DIST and (facing_us or p["distance"] < FLEE_DIST * 0.5):
            # sprint only when it is worth the energy: close, and we have some
            return _turn_away(p["angle"], sprint=(p["distance"] < FLEE_DIST * 0.6 and frac > 0.2))

    # 2. Eat if a tree is in reach and we are not full.
    if trees:
        t = min(trees, key=lambda e: e["distance"])
        if t["distance"] <= EAT_REACH and frac < 0.98:
            return "EAT"
        # 3. Otherwise walk to the nearest tree when hungry.
        if frac < HUNGRY_FRAC:
            return _turn_toward(t["angle"])

    # 4. Nothing worth doing: a slow sweep keeps new trees entering the cone
    #    without paying the movement drain.
    return "LEFT" if not trees else "NOTHING"


def act_all(payload: dict) -> dict[int, str]:
    return {int(a["agent_id"]): heuristic_action(a) for a in payload.get("agent_status", [])}


ACTION_INDEX = {n: i for i, n in enumerate(ACTIONS)}
