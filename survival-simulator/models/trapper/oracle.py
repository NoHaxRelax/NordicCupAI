"""WorldState from the true engine state. Development only: the competition
server never exposes this. Stable predator ids come from object identity."""
from __future__ import annotations

import weakref

from .geometry import Rect
from .world import AgentView, FoodView, PredatorView, WorldState, MOVE_PENALTY


class OracleWorld:
    def __init__(self, env):
        self.env = env
        self._ids = weakref.WeakKeyDictionary()
        self._next = 0
        self._prev = {}          # pid -> (x, y)
        self._still = {}         # pid -> ticks without movement
        self.state: WorldState | None = None

    def pid(self, predator):
        if predator not in self._ids:
            self._ids[predator] = self._next
            self._next += 1
        return self._ids[predator]

    def update(self, states=None, sim_time=None) -> WorldState:
        env = self.env
        by_id = {s['agent_id']: s for s in (states or [])}
        agents = {}
        for a in env.agents:
            ix = min(max(int(a.x), 0), env.width - 1)
            iy = min(max(int(a.y), 0), env.height - 1)
            agents[a.agent_id] = AgentView(
                id=a.agent_id, x=a.x, y=a.y, heading=a.direction, energy=a.energy,
                max_energy=a.max_energy, age=a.age, speed=a.speed, sprint_speed=a.sprint_speed,
                hearing=a.hearing_radius, vision_range=a.vision_radius, vision_angle=a.cone_angle,
                biome=env.biome_map[ix, iy].type, state=by_id.get(a.agent_id, {}))
        predators = []
        for p in env.predators:
            pid = self.pid(p)
            prev = self._prev.get(pid)
            vx = vy = 0.0
            if prev is not None:
                vx, vy = p.x - prev[0], p.y - prev[1]
                self._still[pid] = self._still.get(pid, 0) + 1 if abs(vx) + abs(vy) < 1e-9 else 0
            self._prev[pid] = (p.x, p.y)
            predators.append(PredatorView(pid=pid, x=p.x, y=p.y, heading=p.direction, vx=vx, vy=vy,
                                          last_seen=env.time, still_ticks=self._still.get(pid, 0),
                                          resting=p.resting, energy=p.energy, fresh=True))
        rects = [Rect(o.x, o.y, o.width, o.height) for o in env.obstacles]
        fruits = [FoodView(f.x, f.y, f.age, f.energy) for f in env.fruits]
        trees = [FoodView(t.x, t.y, t.age) for t in env.trees]
        world = _OracleState(time=env.time, width=env.width, height=env.height, agents=agents,
                             predators=predators, rects=rects, fruits=fruits, trees=trees, complete_map=True)
        world._env = env
        self.state = world
        return world


class _OracleState(WorldState):
    _env = None

    def biome_at(self, p):
        env = self._env
        ix = min(max(int(p[0]), 0), env.width - 1)
        iy = min(max(int(p[1]), 0), env.height - 1)
        return MOVE_PENALTY.get(env.biome_map[ix, iy].type, 1.0)
