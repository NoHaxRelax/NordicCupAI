"""Development: at most16 native reserve births, permitting several per tick.

The eight-birth wrapper can lose every guide before it spends its budget.
Here each currently informed guide may request a native child in the same
tick. This changes only public action requests; native birth positions, trait
mutations and deaths remain untouched. The explicit cost is up to17 guides
plus one predeployed bait per encounter, under infinite agent energy.
"""
from backup_guides.policy import BackupGuides
from integrated_guide.policy_v30_wide_route import Guide


class ParallelGuides(BackupGuides):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.births_requested = 3  # disable the predecessor's spaced birth rule
        self.reserve_requests = 0

    def _controller(self, aid):
        if aid not in self.controllers:
            self.controllers[aid] = Guide(
                self.static_map, bait_id=self.bait_id, guide_id=aid)
        return self.controllers[aid]

    def act(self, observations, sim_time):
        actions = super().act(observations, sim_time)
        candidates = []
        for state in observations:
            aid = state['agent_id']
            if aid == self.bait_id or self.controllers[aid].released:
                continue
            distance = self._nearest_predator(state)
            if distance is not None and distance < 75.:
                candidates.append((distance, aid))
        by_id = {action['agent_id']: action for action in actions}
        for distance, aid in sorted(candidates, reverse=True):
            if self.reserve_requests >= 16:
                break
            by_id[aid]['spawn_agent'] = True
            self.reserve_requests += 1
            self.events.append(dict(time=round(sim_time, 1),
                kind='parallel_native_reserve_request', parent_id=aid,
                request_number=self.reserve_requests, observed_distance=distance))
        return actions
