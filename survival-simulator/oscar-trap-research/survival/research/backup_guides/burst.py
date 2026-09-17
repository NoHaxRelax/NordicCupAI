"""Bounded native reserve births during an observed emergency.

At most eight extra guides; native parent-nearby births and mutations. Uses
ordinary observed distance to choose the farthest currently informed parent.
This explicitly spends more guides under the infinite-agent-energy fixture.
"""
from backup_guides.policy import BackupGuides
from backup_guides.mutation_aware import MutationAwareGuide
from release_validation.release_mixin_v2 import ReleaseMixin


class ReleasableGuide(ReleaseMixin, MutationAwareGuide):
    pass


class BurstGuides(BackupGuides):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.births_requested = 3  # disable the predecessor's spaced birth rule
        self.burst_used = 0
        self.parent_births = {}

    def _controller(self, aid):
        if aid not in self.controllers:
            self.controllers[aid] = ReleasableGuide(self.static_map,
                bait_id=self.bait_id, guide_id=aid)
        return self.controllers[aid]

    def act(self, observations, sim_time):
        actions = super().act(observations, sim_time)
        if self.burst_used >= 8:
            return actions
        candidates = []
        for state in observations:
            aid = state['agent_id']
            if aid == self.bait_id or self.controllers[aid].released:
                continue
            distance = self._nearest_predator(state)
            if distance is not None and distance < 75.:
                candidates.append((distance, -self.parent_births.get(aid, 0), aid))
        if candidates:
            distance, _, parent = max(candidates)
            for action in actions:
                if action['agent_id'] == parent:
                    action['spawn_agent'] = True
            self.parent_births[parent] = self.parent_births.get(parent, 0) + 1
            self.burst_used += 1
            self.events.append(dict(time=round(sim_time, 1), kind='bounded_emergency_reserve_birth',
                parent_id=parent, request_number=self.burst_used, observed_distance=distance))
        return actions
