"""Wide terrain routes, viable terrain moves, moving final intake, compact fallback.

Runtime creature input is native observations only. Geometry is static and
permitted. No reserve births. This is a new frozen candidate, not a correction
to the earlier v30 fresh batch.
"""
from short_overlap_sol.policy_v30_viable_terrain_final import Guide as ViableGuide
from integrated_guide.spatial_geometry import SpatialGeometry
from integrated_guide.wide_route_mixin import WideTerrainRoute
from replaceable_sites.selector_compact import enumerate_sites


class Guide(WideTerrainRoute, SpatialGeometry, ViableGuide):
    def _select_site(self):
        try:
            return super()._select_site()
        except ValueError as error:
            if 'no eligible' not in str(error):
                raise
        sites=enumerate_sites(dict(width=self.width,height=self.height,obstacles=self.rects))
        if not sites:
            raise ValueError('no eligible site including experimental compact approach')
        return min(enumerate(sites),key=lambda row:(bool(row[1]['boundary_indices']),
            bool(row[1]['approach_lane_offset']),row[0]))[1]
