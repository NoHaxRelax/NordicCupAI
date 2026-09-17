"""V21 with narrower agent-passable gaps only when ordinary sites are absent."""
from simple_chase.policy_v21_replaceable import SimpleChase as Base
class SimpleChase(Base):
    def _select_site(self):
        try:return super()._select_site()
        except ValueError as error:
            if 'no eligible' not in str(error):raise
        from replaceable_sites.selector import enumerate_sites
        sites=enumerate_sites(dict(width=self.width,height=self.height,obstacles=self.rects),min_gap=10.1)
        if not sites:raise ValueError('no eligible replaceable site including narrow-gap fallback')
        return min(enumerate(sites),key=lambda row:(bool(row[1]['boundary_indices']),bool(row[1]['approach_lane_offset']),row[0]))[1]
