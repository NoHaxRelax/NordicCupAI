"""Behavior-preserving obstacle broad phase for the frozen eight-reserve rule."""
from integrated_guide.spatial_geometry import SpatialGeometry
from backup_guides.burst import BurstGuides as Base, ReleasableGuide

class Guide(SpatialGeometry,ReleasableGuide):
    pass

class BurstGuides(Base):
    def _controller(self,aid):
        if aid not in self.controllers:
            self.controllers[aid]=Guide(self.static_map,bait_id=self.bait_id,guide_id=aid)
        return self.controllers[aid]
