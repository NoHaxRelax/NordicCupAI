"""Broad-phase equivalence on endpoints, touching walls and arbitrary segments."""
import json
import random
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'))
from real_map_guide_sol.policy_v15_hearing_reacquisition_frozen import RealMapGuidePolicy
from integrated_guide.spatial_geometry import SpatialGeometry

class Indexed(SpatialGeometry,RealMapGuidePolicy):pass

rng=random.Random(17092026)
rects=[(0.,0.,1600.,30.),(0.,1170.,1600.,30.),(0.,0.,30.,1200.),(1570.,0.,30.,1200.)]
rects += [(rng.uniform(30,1450),rng.uniform(30,1050),rng.uniform(20,100),rng.uniform(20,100)) for _ in range(80)]
base=object.__new__(RealMapGuidePolicy);fast=object.__new__(Indexed)
for obj in (base,fast):obj.width=1600.;obj.height=1200.;obj.rects=rects
fast._build_geometry_index()
cases=[]
for _ in range(6000):
 a=(rng.uniform(-50,1650),rng.uniform(-50,1250))
 b=(a[0]+rng.uniform(-30,30),a[1]+rng.uniform(-30,30)) if rng.random()<.8 else (rng.uniform(0,1600),rng.uniform(0,1200))
 r=rng.choice((0.,5.,5.01,10.,11.,20.,30.))
 cases.append((a,b,r,() if rng.random()<.9 else (rng.randrange(len(rects)),)))
for i,(x,y,w,h) in enumerate(rects):
 for r in (0.,5.01,10.,20.):
  for a in ((x-r,y-r),(x+w+r,y-r),(x+w+r,y+h+r),(x-r,y+h+r)):
   cases.extend((a,(a[0]+dx,a[1]+dy),r,()) for dx,dy in ((0,0),(1e-8,0),(-1e-8,0),(0,1e-8),(0,-1e-8),(40,40)))
for a,b,r,ignore in cases:
 assert base._free(a,r,ignore)==fast._free(a,r,ignore),(a,r,ignore)
 assert base._clear(a,b,r,ignore)==fast._clear(a,b,r,ignore),(a,b,r,ignore)
times={}
for name,obj in [('baseline',base),('indexed',fast)]:
 start=time.perf_counter()
 for _ in range(2):
  for a,b,r,ignore in cases:obj._free(a,r,ignore);obj._clear(a,b,r,ignore)
 times[name]=time.perf_counter()-start
result=dict(schema='static-obstacle-broad-phase-equivalence-v1',cases=len(cases),mismatches=0,seconds=times,
            limitations='Deterministic geometric predicate comparison, no native simulation steps or reliability evidence.')
out=ROOT/'results/integrated_guide';out.mkdir(parents=True,exist_ok=True)
(out/'spatial_geometry_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
