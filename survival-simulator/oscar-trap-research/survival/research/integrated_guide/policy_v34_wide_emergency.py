"""Development combination: wide terrain routes with native emergency reserves.

Uses the unchanged v30 single-guide controller and the unchanged eight-birth
emergency wrapper. Every child receives ordinary DTOs and static geometry.
No fixture position, native predator target, or hidden rest state is read.
This is a distinct controller and its results must not be pooled with v27/v30.
"""
from backup_guides.emergency import EmergencyGuides as EmergencyBase
from integrated_guide.policy_v30_wide_route import Guide


class EmergencyGuides(EmergencyBase):
    def _controller(self, aid):
        if aid not in self.controllers:
            self.controllers[aid] = Guide(
                self.static_map, bait_id=self.bait_id, guide_id=aid)
        return self.controllers[aid]
