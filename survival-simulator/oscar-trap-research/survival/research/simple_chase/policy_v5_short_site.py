"""V4 contact controller, selecting among short or long static refuges."""
from simple_chase.policy_v4_final_contact import SimpleChase as Base
from real_map_guide_sol.policy_v15_hearing_reacquisition_frozen import add, mul


class SimpleChase(Base):
    def _select_site(self):
        sites=[]
        for i,a in enumerate(self.rects):
            for j,b in enumerate(self.rects):
                if i==j:continue
                for axis in (0,1):
                    ax,ay,aw,ah=(a if axis==0 else (a[1],a[0],a[3],a[2]))
                    bx,by,bw,bh=(b if axis==0 else (b[1],b[0],b[3],b[2]))
                    gap=bx-(ax+aw);low=max(ay,by);high=min(ay+ah,by+bh);overlap=high-low
                    if not (10.9<=gap<=19.1 and overlap>=20.):continue
                    cross=ax+aw+gap/2
                    xy=lambda u,v:(u,v) if axis==0 else (v,u)
                    for sign in (1,-1):
                        mouth=xy(cross,low if sign==1 else high);inward=xy(0.,float(sign))
                        # The validated refuge frontier uses a five-unit bait
                        # depth from this mouth. Keep it fixed for all real-map
                        # runs so the delivery problem is evaluated directly.
                        goal=add(mouth,mul(inward,5.));far=add(mouth,mul(inward,-125.));hold=add(mouth,mul(inward,-75.))
                        if not (self._free(goal,5.01) and self._free(far,11.01) and self._free(hold,11.01)):continue
                        if not self._clear(far,hold,11.01):continue
                        # Select a site with a usable final runup rather than
                        # choosing a site first and rejecting the whole map.
                        runups=[add(mouth,mul(inward,-d)) for d in (250.,225.,200.,180.,160.,145.)]
                        if not any(self._clear(far,r,11.) for r in runups):continue
                        if not self._clear(hold,mouth,11.,ignore=(i,j)):continue
                        # Prefer long overlap, central mouths and uncluttered staging.
                        margin=min(far[0],far[1],self.width-far[0],self.height-far[1])
                        score=overlap+.05*margin
                        sites.append((score,dict(mouth=mouth,inward=inward,cross=(-inward[1],inward[0]),
                            goal=goal,far=far,hold=hold,gap=gap,overlap=overlap,obstacle_indices=[i,j],axis=axis)))
        if not sites:raise ValueError("static map has no eligible 11-19 gap with overlap20 and clear axial runup")
        site=max(sites,key=lambda row:row[0])[1]
        return {k:(list(v) if isinstance(v,tuple) else v) for k,v in site.items()}

