"""Deterministic independent full-map start sampling."""
from __future__ import annotations

import hashlib
import math
import random


class SamplingError(RuntimeError):
    pass


def _stream(seed: int, label: str, index: int) -> random.Random:
    material = f"real-map-intake-v1:{seed}:{label}:{index}".encode("ascii")
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return random.Random(derived)


def _free_for_native_spawn(env, point: tuple[float, float], radius: float,
                           kind: str) -> bool:
    x, y = point
    if x < radius or x > env.width - radius or y < radius or y > env.height - radius:
        return False
    if env._in_obstacle(point, radius=radius, obstacles=env.obstacles):
        return False
    # Upstream spawn methods use an asymmetric box check in addition to the
    # center/radius collision model. Honor that check so an accepted fixture is
    # never silently replaced with an engine-chosen random point.
    extent = 20 if kind == "agent" else radius
    return bool(env._is_position_free(x, y, extent, extent))


def sample_start(env, *, fixture_seed: int, kind: str, index: int,
                 occupied=(), attempts: int = 20_000) -> tuple[float, float, float]:
    """Sample one obstacle-valid start and heading from a role-specific stream.

    Streams depend only on fixture seed, creature kind, and slot index. Agent
    and predator starts therefore have no guide/follower pairing or shared
    offsets. Rejection only enforces native obstacle validity and no initial
    creature overlap.
    """
    if kind not in {"agent", "predator"}:
        raise ValueError("kind must be agent or predator")
    radius = 5.0 if kind == "agent" else 10.0
    rng = _stream(fixture_seed, kind, index)
    for _ in range(attempts):
        point = (rng.uniform(radius, env.width - radius),
                 rng.uniform(radius, env.height - radius))
        if not _free_for_native_spawn(env, point, radius, kind):
            continue
        if any(math.dist(point, other_point) <= radius + other_radius
               for other_point, other_radius in occupied):
            continue
        return point[0], point[1], rng.uniform(-math.pi, math.pi)
    raise SamplingError(f"could not place {kind} slot {index} in {attempts} attempts")


def sample_fixture(env, *, fixture_seed: int, agents: int, predators: int):
    """Return independently seeded starts distributed over the full valid map."""
    occupied = []
    agent_rows = []
    predator_rows = []
    for index in range(agents):
        x, y, heading = sample_start(
            env, fixture_seed=fixture_seed, kind="agent", index=index,
            occupied=occupied,
        )
        agent_rows.append({"slot": index, "x": x, "y": y, "heading": heading})
        occupied.append(((x, y), 5.0))
    for index in range(predators):
        x, y, heading = sample_start(
            env, fixture_seed=fixture_seed, kind="predator", index=index,
            occupied=occupied,
        )
        predator_rows.append({"slot": index, "x": x, "y": y, "heading": heading})
        occupied.append(((x, y), 10.0))
    return {"agents": agent_rows, "predators": predator_rows}

