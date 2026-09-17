"""Exact static-obstacle broad phase; same geometry predicates, fewer rectangles."""
import math


class SpatialGeometry:
    CELL=80.

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._build_geometry_index()

    def _build_geometry_index(self):
        index={}
        for i,(x,y,w,h) in enumerate(self.rects):
            for cx in range(math.floor(x/self.CELL),math.floor((x+w)/self.CELL)+1):
                for cy in range(math.floor(y/self.CELL),math.floor((y+h)/self.CELL)+1):
                    index.setdefault((cx,cy),[]).append(i)
        self._geometry_index=index

    def _nearby_rectangles(self,a,b,radius):
        # Include a tiny broad-phase tolerance at cell boundaries; the original
        # exact predicates still decide contact, so extra candidates are safe.
        lo=(math.floor((min(a[0],b[0])-radius-1e-9)/self.CELL),math.floor((min(a[1],b[1])-radius-1e-9)/self.CELL))
        hi=(math.floor((max(a[0],b[0])+radius+1e-9)/self.CELL),math.floor((max(a[1],b[1])+radius+1e-9)/self.CELL))
        found=set()
        for cx in range(lo[0],hi[0]+1):
            for cy in range(lo[1],hi[1]+1):
                found.update(self._geometry_index.get((cx,cy),()))
        return found

    def _free(self,p,radius,ignore=()):
        if not hasattr(self,'_geometry_index'):
            return super()._free(p,radius,ignore)
        if not(radius<=p[0]<=self.width-radius and radius<=p[1]<=self.height-radius):return False
        for i in self._nearby_rectangles(p,p,radius):
            if i in ignore:continue
            x,y,w,h=self.rects[i]
            if x-radius<p[0]<x+w+radius and y-radius<p[1]<y+h+radius:return False
        return True

    def _clear(self,a,b,radius,ignore=()):
        if not hasattr(self,'_geometry_index'):
            return super()._clear(a,b,radius,ignore)
        if not self._free(a,radius,ignore) or not self._free(b,radius,ignore):return False
        for i in self._nearby_rectangles(a,b,radius):
            if i in ignore:continue
            x,y,w,h=self.rects[i]
            if self._segment_rect(a,b,(x-radius,y-radius,w+2*radius,h+2*radius)):return False
        return True
