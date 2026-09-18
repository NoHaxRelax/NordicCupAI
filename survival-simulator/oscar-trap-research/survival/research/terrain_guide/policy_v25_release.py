"""Compose frozen terrain routing, replaceable gap fallbacks and guide retirement."""
from terrain_guide.policy import TerrainGuide
from simple_chase.policy_v24_short_fallback import SimpleChase as ExpandedSites
from release_validation.release_mixin_v2 import ReleaseMixin
class ReleaseGuide(ReleaseMixin,TerrainGuide,ExpandedSites):pass
