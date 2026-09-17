"""TrapperPolicy: the society baseline plus trap roles.

    policy = TrapperPolicy(seed, env=sim.env)        # oracle world (development)
    actions = policy(state['observations'], state['sim_time'])

The society decides for everyone each tick (so its per-agent memory stays
current); the manager then overrides the agents that hold a trap role or stand
inside a held zone, and the override is written back into the society's
odometry (``Mind.last_action``) so its dead reckoning follows what really
happened.
"""
from __future__ import annotations

from src.utils.DTOs import ActionRequest

from .manager import TrapManager
from .oracle import OracleWorld
from .estimator import EstimatedWorld
from .society_base import SocietyPolicy
from .world import WorldState


class TrapperPolicy:
    def __init__(self, seed=0, env=None, trap=True, society_kwargs=None, world_source=None, world='oracle', **trap_params):
        """world: 'oracle' (needs env) or 'estimator' (observations only, as on the server)."""
        self.seed = seed
        self.society = SocietyPolicy(seed=seed, **(society_kwargs or {}))
        self.trap = trap
        if world_source is not None:
            self.source = world_source
        elif world == 'estimator' or env is None:
            self.source = EstimatedWorld()
        else:
            self.source = OracleWorld(env)
        self.manager = TrapManager(**trap_params)
        self.world: WorldState | None = None
        self.decisions = {}
        self.last_decisions = {}

    @property
    def metrics(self):
        m = dict(self.society.metrics)
        m.update({f'trap_{k}': v for k, v in self.manager.metrics.items()})
        return m

    def __call__(self, states, sim_time):
        base = self.society(states, sim_time)
        self.decisions = {aid: str(self.society.decisions.get(aid)) for aid, _ in base}
        if not self.trap or self.source is None:
            self.last_decisions = {aid: dict(rule=r) for aid, r in self.decisions.items()}
            return base
        world = self.world = self.source.update(states, sim_time)
        fleeing = {aid for aid, _ in base if (self.society.decisions.get(aid) or ('',))[0] == 'flee'}
        overrides = self.manager.step(world, fleeing=fleeing)
        by_id = {s['agent_id']: s for s in states}
        out = []
        for aid, act in base:
            over = overrides.get(aid)
            why = None
            if over is None:
                a = world.agents.get(aid)
                if a is not None:
                    o = self.manager.society_override(world, a, fleeing=aid in fleeing)
                    if o is not None:
                        over = o
            if over is not None:
                d, why = over
                act = ActionRequest(agent_id=aid, move_distance=float(d['move_distance']), move_direction=float(d['move_direction']),
                                    turn_angle=float(d['turn_angle']), spawn_agent=bool(d.get('spawn_agent', False)))
                self._note(aid, act, by_id.get(aid))
                self.decisions[aid] = why
            out.append((aid, act))
        self.last_decisions = {aid: dict(rule=r) for aid, r in self.decisions.items()}
        self.manager.last_actions = {aid: act for aid, act in out}
        if hasattr(self.source, 'note_actions'):
            self.source.note_actions(out)
        return out

    def _note(self, aid, act: ActionRequest, s):
        """Write the override into the society's odometry (engine order: move then turn)."""
        m = self.society.minds.get(aid)
        if m is None or s is None:
            return
        m.last_action = (act.move_distance, act.move_direction, act.turn_angle, s['biome'], s['energy'],
                         s['speed'], s['sprint_speed'], s['max_energy'])
