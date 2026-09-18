"""Oracle census of enclosed predator-clear pockets and short jump entrances.

One-unit raster is a candidate screen, not an exact reachability proof. The
mechanism sought is sprint-15 entry into a pocket whose only exit exceeds the
predator's untargeted walking request of 11. No engine changes or controller.
"""
from common import *
import numpy as np
from scipy.ndimage import label,distance_transform_edt
from src.core import SimulationCore

def census(env):
    free=np.ones((1600,1200),bool)
    for o in env.obstacles:
        x0=max(0,math.floor(o.x-10)+1);x1=min(1600,math.ceil(o.x+o.width+10))
        y0=max(0,math.floor(o.y-10)+1);y1=min(1200,math.ceil(o.y+o.height+10))
        free[x0:x1,y0:y1]=False
    labels,n=label(free,np.ones((3,3)));counts=np.bincount(labels.flat);counts[0]=0
    main=int(np.argmax(counts));dist,indices=distance_transform_edt(labels!=main,return_indices=True)
    pockets=[]
    for k in range(1,n+1):
        if k==main or counts[k]<4:continue
        pts=np.argwhere(labels==k);ds=dist[pts[:,0],pts[:,1]];q=pts[np.argmin(ds)];v=indices[:,q[0],q[1]]
        pockets.append(dict(area=int(counts[k]),minimum_integer_point_gap=float(ds.min()),inside=q.tolist(),outside=v.tolist(),
            potentially_sprint_only=bool(11<ds.min()<=15),
            terrain_inside=env.biome_map[q[0],q[1]].type,terrain_outside=env.biome_map[v[0],v[1]].type))
    return pockets

def refine():
    from scipy.spatial import cKDTree
    rows=json.loads((OUT/'alternative-pocket-survey.json').read_text())['data']
    m=next(m for m in rows if m['seed']==16)
    pocket=min(m['pockets'],key=lambda p:p['minimum_integer_point_gap'])
    env=SimulationCore(seed=16).env;clouds=[]
    for center in [pocket['inside'],pocket['outside']]:
        a=np.arange(-3,3.0001,.025)
        points=np.stack(np.meshgrid(a,a,indexing='ij'),axis=-1).reshape(-1,2)+center
        valid=np.ones(len(points),bool)
        for o in env.obstacles:
            valid &= ~((points[:,0]>o.x-10)&(points[:,0]<o.x+o.width+10)&
                       (points[:,1]>o.y-10)&(points[:,1]<o.y+o.height+10))
        clouds.append(points[valid])
    distance,ids=cKDTree(clouds[1]).query(clouds[0]);k=np.argmin(distance)
    save('alternative-pocket-refinement',dict(seed=16,inside=clouds[0][k].tolist(),
        outside=clouds[1][ids[k]].tolist(),fine_grid_gap=float(distance[k]),resolution=.025))

if __name__=='__main__' and '--refine' in sys.argv:
    refine()
elif __name__=='__main__':
    rows=[]
    for seed in range(1,21):
        env=SimulationCore(seed=seed).env;pockets=census(env)
        rows.append(dict(seed=seed,pockets=pockets))
        print(seed,len(pockets),sum(p['potentially_sprint_only'] for p in pockets),flush=True)
        save('alternative-pocket-survey',rows)
