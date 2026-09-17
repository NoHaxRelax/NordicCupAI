"""One guide selecting among multiple statically deployed trap baits.

Harness contract: ``bait_sites`` maps each already deployed native bait agent
ID to the exact compatible site dictionary where setup placed it. The policy
chooses once, from the guide's DTO-localized pose, and never receives predator
truth. All nonchosen baits remain stationary and can retain earlier captures.
"""
from __future__ import annotations

import math

from integrated_guide.policy_v31_compact_fallback import Guide as BaseGuide


class Guide(BaseGuide):
    pass


def dispersed_sites(sites, limit=8):
    """Choose distinct corridors, greedily maximizing mouth separation."""
    distinct = []
    seen = set()
    for site in sites:
        corridor = tuple(sorted(site["obstacle_indices"]))
        if corridor not in seen:
            seen.add(corridor)
            distinct.append(site)
    if len(distinct) <= limit:
        return distinct
    selected = [distinct[0]]
    remaining = distinct[1:]
    while remaining and len(selected) < limit:
        row = max(remaining, key=lambda candidate: min(
            math.dist(candidate["mouth"], chosen["mouth"]) for chosen in selected))
        remaining.remove(row)
        selected.append(row)
    return selected


class MultiSiteGuide:
    """Static bait fleet with a single DTO-local choice by the live guide."""

    def __init__(self, static_map, bait_id=0, guide_id=1, *, bait_sites=None):
        if not bait_sites:
            raise ValueError("MultiSiteGuide requires setup-provided bait_sites mapping")
        self.static_map = static_map
        self.bait_sites = {int(aid): dict(site) for aid, site in bait_sites.items()}
        self.bait_id = int(bait_id)
        self.guide_id = int(guide_id)
        self.guide_role_ids = [self.guide_id]
        self.active_guide_id = self.guide_id
        self.selected_bait_id = None
        self.controller = Guide(static_map, bait_id=self.bait_id, guide_id=self.guide_id)
        self.site = self.controller.site
        self.bait_ready = False
        self.decisions = {}
        self.events = []

    def _select_from_public_pose(self, state, sim_time):
        pose = self.controller._localize(state)
        if pose is None:
            return
        bait_id, site = min(self.bait_sites.items(),
                            key=lambda row: math.dist(pose.p, row[1]["far"]))
        self.selected_bait_id = bait_id
        self.bait_id = bait_id
        self.site = site
        self.controller.bait_id = bait_id
        self.controller.site = site
        self.controller.runup = tuple(site["runup"])
        self.controller.route = []
        self.controller.phase = "localize"
        self.controller.engaged = False
        self.events.append({"time": round(sim_time, 1),
                            "kind": "dto_pose_nearest_static_site_selected",
                            "bait_id": bait_id,
                            "mouth": list(site["mouth"])})

    @staticmethod
    def _still(aid):
        return {"agent_id": aid, "move_distance": 0., "move_direction": 0.,
                "turn_angle": 0., "spawn_agent": False}

    def act(self, observations, sim_time):
        states = {row["agent_id"]: row for row in observations}
        guide_state = states.get(self.guide_id)
        if self.selected_bait_id is None and guide_state is not None:
            self._select_from_public_pose(guide_state, sim_time)
        actions = []
        self.decisions = {}
        for aid in sorted(states):
            if aid in self.bait_sites:
                actions.append(self._still(aid))
                self.decisions[aid] = {"rule": "stationary_static_site_bait",
                                      "selected": aid == self.selected_bait_id}
            elif aid == self.guide_id:
                action = self.controller.act([states[aid]], sim_time)[0]
                action["spawn_agent"] = False
                actions.append(action)
                self.decisions[aid] = dict(self.controller.decisions.get(aid, {}))
                self.decisions[aid]["selected_bait_id"] = self.selected_bait_id
            else:
                actions.append(self._still(aid))
                self.decisions[aid] = {"rule": "unassigned_agent_stationary"}
        return actions
