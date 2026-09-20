"""Negative-energy harvest layer (duplicate-action defect).

An agent about to be eaten is drained far below zero with repeated turn_angle=pi actions
(0.5 energy each, 0 px). It survives its own energy<=0 check only because the agent
immediately before it in env.agents dies in the same tick (list-mutation skip), so a
sacrificial neighbour (next-lower live id) is drained to just below zero at the same time.
The predator then eats a negative-energy agent: score -= energy/100 pays out, and the
predator absorbs the negative energy and sleeps for a long time.
"""
import math

PI = math.pi
TURN_COST = 0.5          # energy per turn_angle=pi action (moves 0 px)
KILL_RADIUS = 15.0       # predator.size (10) + agent.size (5)


def _drain_action(aid):
    return {"agent_id": aid, "move_distance": 0.0, "move_direction": 0.0, "turn_angle": PI, "spawn_agent": False}


class Harvester:
    def __init__(self, budget=20000, trigger=30.0, closing=6.0, max_harvests=8,
                 min_free=6, cooldown=50.0, enabled=True, sacrifice_mode="low_energy",
                 old_age=55.0):
        self.budget = int(budget); self.trigger = trigger; self.closing = closing
        self.max_harvests = int(max_harvests); self.min_free = int(min_free)
        self.cooldown = cooldown; self.enabled = enabled
        self.sacrifice_mode = sacrifice_mode; self.old_age = float(old_age)
        self.reset()

    def reset(self):
        self.prev_pred_d = {}; self.harvests = 0; self.last_t = -1e9
        self.log = []

    def _min_pred(self, state):
        d = min((float(o["distance"]) for o in state.get("observations", []) if str(o["type"]).lower() == "predator"), default=None)
        return d

    def apply(self, states, sim_time, actions):
        """Returns the action list to send (duplicates included)."""
        pred_d = {int(s["agent_id"]): self._min_pred(s) for s in states}
        energy = {int(s["agent_id"]): float(s["energy"]) for s in states}
        age = {int(s["agent_id"]): float(s.get("age", 0.0)) for s in states}
        prev, self.prev_pred_d = self.prev_pred_d, pred_d
        if not self.enabled or self.harvests >= self.max_harvests or sim_time - self.last_t < self.cooldown \
           or len(states) < self.min_free + 2:
            return actions
        live = sorted(energy)                                  # engine list order == ascending id
        pos = {aid: i for i, aid in enumerate(live)}
        cands = []
        for aid, d in pred_d.items():
            if d is None or d > self.trigger:
                continue
            was = prev.get(aid)
            if was is None or was - d < self.closing:          # predator must be actively closing
                continue
            i = pos[aid]
            if i == 0:
                continue                                       # no lower-id neighbour to sacrifice
            neighbour = live[i - 1]
            if pred_d.get(neighbour) is not None and pred_d[neighbour] <= KILL_RADIUS:
                continue                                       # neighbour may be eaten before its own check
            if self.sacrifice_mode == "old_chased":
                # All candidates are actively chased. Prefer an old chased agent
                # as the bait, then the oldest one, then the most imminent chase.
                cands.append((age[aid] < self.old_age, -age[aid], d, energy[neighbour], aid, neighbour))
            elif self.sacrifice_mode == "nearest_chased":
                # The predator observation has no identity, so the safest proxy for
                # the predator's actual target is the closest actively closing agent.
                cands.append((d, -(was - d), energy[neighbour], aid, neighbour))
            else:
                cands.append((energy[neighbour], d, aid, neighbour))
        if not cands:
            return actions
        chosen = min(cands)
        if self.sacrifice_mode == "old_chased":
            _, _, d, _, farm, doomed = chosen
        elif self.sacrifice_mode == "nearest_chased":
            d, _, _, farm, doomed = chosen
        else:
            _, d, farm, doomed = chosen                        # cheapest sacrifice
        n_doomed = int(math.ceil((energy[doomed] + 1.0) / TURN_COST))
        out = [a for a in actions if a["agent_id"] not in (farm, doomed)]
        out += [_drain_action(doomed)] * n_doomed
        out += [_drain_action(farm)] * self.budget
        self.harvests += 1; self.last_t = sim_time
        payout = (self.budget * TURN_COST - energy[farm]) / 100.0
        self.log.append({"t": round(sim_time, 1), "farm": farm, "doomed": doomed, "pred_d": round(d, 1),
                         "farm_e": round(energy[farm], 1), "farm_age": round(age[farm], 1),
                         "doomed_e": round(energy[doomed], 1), "doomed_age": round(age[doomed], 1),
                         "expected_score": round(payout, 2), "actions": len(out)})
        return out
