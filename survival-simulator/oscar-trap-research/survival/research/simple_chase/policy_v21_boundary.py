"""Development diagnostic: explicitly select an eligible boundary gap."""
from simple_chase.policy_v21_replaceable import SimpleChase as Base
class BoundaryGuide(Base):
    def _select_site(self):
        from replaceable_sites.selector import enumerate_sites
        sites=enumerate_sites(dict(width=self.width,height=self.height,obstacles=self.rects))
        sites=[s for s in sites if s['boundary_indices']]
        if not sites:raise ValueError('no eligible boundary site for diagnostic')
        return sites[0]
