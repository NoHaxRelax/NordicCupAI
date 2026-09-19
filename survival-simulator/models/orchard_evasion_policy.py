"""Coordinated orchard foraging plus own-sighting predator evasion. No trapping.

Second seed family for the no-trapping optimization campaign. The foraging,
population and lineage-shared-map behaviour is the vendored
models/survival/orchard_population.py; this layer adds the predator response it
lacks entirely.

Evasion defaults come from the `with_predators_best` config measured on
survival-simulator/oscar-overnight-cpp (configs/best-configs.json). That
campaign's own numbers, on its separate C++ engine port and therefore not
verified against this engine, ranked flee-close/own-sightings above both a
wider-radius shared-sighting variant (1239 s vs 923 s mean survival) and every
bait/guide trapping variant it tried (all negative against this same baseline).
Treat those as the reason these defaults were chosen, not as measured results
for this repository.

Sharing predator sightings across a group is deliberately not implemented:
upstream measured it worse than own-sightings-only, so `pred_share` accepts 0
and rejects anything else rather than silently ignoring the setting.

Inputs stay the public per-agent observation dictionaries and simulation time.
The threat query reads only observation entries of type 'Predator' with their
reported distance/angle; no engine state, world seed or true coordinates.
"""
from __future__ import annotations

import math

from models.survival.orchard_population import OrchardPolicy, wrap


class OrchardEvasionPolicy(OrchardPolicy):
    def __init__(self, seed=0, *, pred_mode=1, pred_r=70., pred_face_r=80., pred_sprint_r=40.,
                 pred_dodge_r=80., pred_dodge_ang=1.4, pred_share=0, pred_turn_max=1.0, **kw):
        if pred_share:
            raise ValueError('pred_share>0 (group-shared predator sightings) is not implemented')
        super().__init__(seed=seed, **kw)
        self.PRED = dict(mode=int(pred_mode), r=float(pred_r), face_r=float(pred_face_r),
                         sprint_r=float(pred_sprint_r), dodge_r=float(pred_dodge_r),
                         dodge_ang=float(pred_dodge_ang), turn_max=float(pred_turn_max))
        self.metrics.update(flee_ticks=0, face_ticks=0, sprint_ticks=0)

    def _threat(self, s):
        """Nearest predator in this agent's own observations, or None."""
        nearest = None
        for o in s['observations']:
            if o['type'] != 'Predator':
                continue
            if nearest is None or o['distance'] < nearest['distance']:
                nearest = o
        return nearest

    def _release_fruit(self, m):
        """Free a claimed fruit before fleeing/facing, or it sits reserved and
        unreachable by hungrier group-mates for as long as the threat lasts."""
        if m.fruit is not None:
            g = self.groups[m.group]
            if m.fruit in g.fruits:
                g.fruits[m.fruit].claimed = None
            m.fruit = None

    def _act(self, m, s, states):
        P = self.PRED
        if P['mode']:
            threat = self._threat(s)
            if threat is not None:
                distance, angle = threat['distance'], threat['angle']
                clamp = lambda a: max(-P['turn_max'], min(P['turn_max'], a))
                if distance <= P['r']:
                    away = wrap(angle+math.pi)
                    # A predator outruns an agent in a straight line, so break across
                    # its approach instead of directly away once it is this close.
                    if distance <= P['dodge_r']:
                        away = wrap(away+(P['dodge_ang'] if angle >= 0. else -P['dodge_ang']))
                    sprinting = distance <= P['sprint_r']
                    reach = s['sprint_speed'] if sprinting else min(s['speed'], s['sprint_speed'])
                    self.metrics['flee_ticks'] += 1
                    self.metrics['sprint_ticks'] += sprinting
                    self._release_fruit(m)
                    return reach, away, clamp(away), 'flee'
                if distance <= P['face_r']:
                    self.metrics['face_ticks'] += 1
                    self._release_fruit(m)
                    return 0., 0., clamp(angle), 'face'
        return super()._act(m, s, states)
