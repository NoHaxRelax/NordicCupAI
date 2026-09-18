"""Wide terrain routing with an experimental compact-runup site fallback."""
from integrated_guide.policy_v30_wide_route import Guide as WideGuide
from replaceable_sites.selector_compact import enumerate_sites


class Guide(WideGuide):
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
