from terrain_guide.policy import TerrainGuide as Base
from simple_chase.policy_v21_boundary import BoundaryGuide
class TerrainBoundaryGuide(Base):
    _select_site=BoundaryGuide._select_site
