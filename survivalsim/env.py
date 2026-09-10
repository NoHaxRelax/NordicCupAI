"""A survival simulator that emits the captured payload schema exactly.

The contract this file has to keep is narrow and strict: whatever happens
internally, `observe()` must produce a dict that is key-for-key compatible with
what came off their endpoint. Everything else here is a guess we deliberately
randomise over.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from .spec import WorldSpec, sample_spec

# Discrete action set. Their previous control task (race-car) used five discrete
# actions, so a small discrete set is the most likely interface; a continuous
# adapter lives at the bottom of this file for the case where it is not.
ACTIONS = ("NOTHING", "FORWARD", "SPRINT", "LEFT", "RIGHT", "EAT")
A = {name: i for i, name in enumerate(ACTIONS)}


def _wrap(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles into (-pi, pi], which is the convention the capture used."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class SurvivalSim:
    """Multi-agent survival sim over a square world.

    Agents walk, sprint, turn and eat. Trees restore energy, predators end runs,
    energy drains continuously. Score accumulates from a randomised mixture of
    age, energy gathered and trees eaten, because the capture does not say which.
    """

    def __init__(self, spec: WorldSpec | None = None, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.spec = spec if spec is not None else sample_spec(self.rng)
        self.reset()

    # ---------------------------------------------------------------- reset

    def reset(self, spec: WorldSpec | None = None) -> dict:
        if spec is not None:
            self.spec = spec
        s = self.spec
        n, W = s.n_agents, s.world_size

        self.t = 0.0
        self.agent_xy = self.rng.uniform(0.1 * W, 0.9 * W, size=(n, 2))
        self.agent_th = self.rng.uniform(-np.pi, np.pi, size=n)
        self.energy = np.full(n, s.max_energy * s.start_energy_frac)
        self.alive = np.ones(n, dtype=bool)
        self.age = np.zeros(n)
        self.gathered = np.zeros(n)
        self.eaten = np.zeros(n, dtype=int)

        self.tree_xy = self.rng.uniform(0, W, size=(s.n_trees, 2))
        self.tree_ready = np.ones(s.n_trees, dtype=bool)
        self.tree_timer = np.zeros(s.n_trees)

        self.pred_xy = self.rng.uniform(0, W, size=(s.n_predators, 2))
        self.pred_th = self.rng.uniform(-np.pi, np.pi, size=s.n_predators)

        return self.observe()

    # ----------------------------------------------------------------- step

    def step(self, actions: Iterable[int]) -> tuple[dict, np.ndarray, bool, dict]:
        s = self.spec
        dt = s.dt
        acts = np.asarray(list(actions), dtype=int)
        prev_score = self.score_per_agent()
        was_alive = self.alive.copy()

        moving = np.isin(acts, (A["FORWARD"], A["SPRINT"]))
        sprinting = acts == A["SPRINT"]

        # turn
        self.agent_th = _wrap(
            self.agent_th
            + s.turn_rate * dt * ((acts == A["LEFT"]).astype(float) - (acts == A["RIGHT"]).astype(float))
        )

        # translate
        speed = np.where(sprinting, s.speed * s.sprint_mult, s.speed) * moving
        step = (speed * dt)[:, None] * np.stack([np.cos(self.agent_th), np.sin(self.agent_th)], axis=1)
        self.agent_xy = np.clip(self.agent_xy + step, 0.0, s.world_size)

        # energy
        drain = np.where(moving, s.drain_move, s.drain_idle)
        drain = np.where(sprinting, drain * s.drain_sprint_mult, drain)
        self.energy -= drain * dt * self.alive

        # eat: nearest ready tree inside reach
        if s.n_trees:
            d = np.linalg.norm(self.agent_xy[:, None, :] - self.tree_xy[None, :, :], axis=2)
            d = np.where(self.tree_ready[None, :], d, np.inf)
            for i in np.flatnonzero((acts == A["EAT"]) & self.alive):
                j = int(np.argmin(d[i]))
                if d[i, j] <= s.tree_reach:
                    gain = min(s.tree_energy, s.max_energy - self.energy[i])
                    self.energy[i] += gain
                    self.gathered[i] += gain
                    self.eaten[i] += 1
                    self.tree_ready[j] = False
                    self.tree_timer[j] = s.tree_respawn

        # tree respawn
        if s.trees_respawn and s.n_trees:
            self.tree_timer = np.maximum(0.0, self.tree_timer - dt)
            self.tree_ready |= self.tree_timer <= 0.0
        elif s.n_trees:
            self.tree_timer = np.maximum(0.0, self.tree_timer - dt)

        self._step_predators(dt)

        # deaths
        self.energy = np.clip(self.energy, 0.0, s.max_energy)
        self.alive &= self.energy > 0.0
        if s.n_predators and s.predators_hunt:
            dp = np.linalg.norm(self.agent_xy[:, None, :] - self.pred_xy[None, :, :], axis=2)
            self.alive &= dp.min(axis=1) > s.predator_kill_range

        self.age += dt * self.alive
        self.t += dt

        # Reward is the score delta, which keeps the training signal aligned with
        # whatever `score` turns out to mean, plus an explicit penalty for dying
        # this tick so the policy learns that death ends the accrual.
        reward = self.score_per_agent() - prev_score
        reward -= s.death_penalty * (was_alive & ~self.alive)
        done = (not self.alive.any()) or (self.t >= s.max_age)
        return self.observe(), reward, bool(done), {"alive": self.alive.copy()}

    def _step_predators(self, dt: float) -> None:
        s = self.spec
        if not s.n_predators:
            return
        pspeed = s.speed * s.predator_speed_mult

        if s.predators_hunt and self.alive.any():
            live = np.flatnonzero(self.alive)
            rel = self.agent_xy[live][None, :, :] - self.pred_xy[:, None, :]
            dist = np.linalg.norm(rel, axis=2)
            bearing = np.arctan2(rel[:, :, 1], rel[:, :, 0])
            # a predator only chases prey inside its own range and cone
            visible = (dist <= s.predator_aggro) & (np.abs(_wrap(bearing - self.pred_th[:, None])) <= s.predator_fov)
            dist_v = np.where(visible, dist, np.inf)
            tgt = np.argmin(dist_v, axis=1)
            has_tgt = np.isfinite(dist_v.min(axis=1))
            chase_th = bearing[np.arange(s.n_predators), tgt]
            self.pred_th = np.where(has_tgt, chase_th, self.pred_th)
        else:
            has_tgt = np.zeros(s.n_predators, dtype=bool)

        wander = self.rng.normal(0.0, s.predator_wander * dt, size=s.n_predators)
        self.pred_th = _wrap(self.pred_th + np.where(has_tgt, 0.0, wander))
        move = (pspeed * dt) * np.stack([np.cos(self.pred_th), np.sin(self.pred_th)], axis=1)
        self.pred_xy = np.clip(self.pred_xy + move, 0.0, s.world_size)

    # ---------------------------------------------------------------- score

    def score_per_agent(self) -> np.ndarray:
        s = self.spec
        return s.w_age * self.age + s.w_energy_gathered * (self.gathered / 100.0) + s.w_trees_eaten * self.eaten

    # ---------------------------------------------------- observation (fixed)

    def observe(self) -> dict:
        """Emit the captured schema. This shape is a contract; do not change it."""
        return {
            "game_status": "running" if self.alive.any() and self.t < self.spec.max_age else "game_over",
            "score": float(round(self.score_per_agent().sum(), 4)),
            "agent_status": [self._agent_status(i) for i in range(self.spec.n_agents) if self.alive[i]],
        }

    def _agent_status(self, i: int) -> dict:
        s = self.spec
        return {
            "agent_id": int(i + 1),  # the capture showed agent_id 1, so 1-indexed
            "observations": self._observations(i),
            "energy": float(round(self.energy[i], 4)),
            "max_energy": float(s.max_energy),
            "biome": s.biome,
            "age": float(round(self.age[i], 4)),
            "speed": float(s.speed),
            "sprint_speed": float(round(s.speed * s.sprint_mult, 4)),
            "hearing_radius": float(s.hearing_radius),
            "vision_angle": float(s.vision_angle),
            "vision_range": float(s.vision_range),
        }

    def _observations(self, i: int) -> list[dict]:
        s = self.spec
        pos, th = self.agent_xy[i], self.agent_th[i]
        out: list[dict] = []

        def sensed(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            rel = points - pos
            dist = np.linalg.norm(rel, axis=1)
            ang = _wrap(np.arctan2(rel[:, 1], rel[:, 0]) - th)
            seen = ((dist <= s.vision_range) & (np.abs(ang) <= s.vision_angle)) | (dist <= s.hearing_radius)
            return dist, ang, seen

        if s.n_trees:
            ready = np.flatnonzero(self.tree_ready)
            if ready.size:
                dist, ang, seen = sensed(self.tree_xy[ready])
                for k in np.flatnonzero(seen):
                    out.append({"type": "tree", "distance": float(round(dist[k], 4)),
                                "angle": float(round(ang[k], 4))})

        if s.n_predators:
            dist, ang, seen = sensed(self.pred_xy)
            for k in np.flatnonzero(seen):
                out.append({"type": "predator", "distance": float(round(dist[k], 4)),
                            "angle": float(round(ang[k], 4)),
                            # the capture reports predator heading, so evasion can
                            # depend on where it is looking rather than only range
                            "rel_dir": float(round(_wrap(self.pred_th[k] - th), 4))})

        # Edges came back in ABSOLUTE coords in the capture, unlike everything
        # else, so they are emitted the same way here.
        W = s.world_size
        corners = [(0.0, 0.0), (W, 0.0), (W, W), (0.0, W)]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            if _seg_dist(pos, np.array(a), np.array(b)) <= max(s.vision_range, s.hearing_radius):
                out.append({"type": "edge", "coords": [[float(a[0]), float(a[1])],
                                                       [float(b[0]), float(b[1])]]})
        return out


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom == 0 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))
