"""Privileged site availability survey, including arbitrary river angles.

This reads native biome pixels and obstacle geometry. It is deliberately not
claimed to discover a map from the competition observation DTO.
"""
from common import *
import argparse
import random
import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter
from src.elements.biome import Map_generator

def frame(site):
    th = site['angle']
    return np.array([[math.cos(th), -math.sin(th)],
                     [math.sin(th), math.cos(th)]])

def world(site, xy):
    return np.asarray(xy) @ frame(site).T + np.array(site['center'])

def local(site, xy):
    return (np.asarray(xy) - np.array(site['center'])) @ frame(site)

def find_sites(biomes):
    water = np.fromiter((b.type == 'river' for b in biomes.flat), bool,
                        count=biomes.size).reshape(biomes.shape)
    dry = np.fromiter((b.move_penalty == 1 for b in biomes.flat), bool,
                      count=biomes.size).reshape(biomes.shape)
    edt = distance_transform_edt(water)
    # Sample ridge points, avoiding enumeration of all river pixels.
    ridge = (edt >= 18) & (edt <= 32) & (edt >= maximum_filter(edt, size=11)-.1)
    points = np.argwhere(ridge)
    chosen = []
    for pt in points:
        if min(pt) < 105 or pt[0] > 1495 or pt[1] > 1095: continue
        if any(np.linalg.norm(pt - p) < 28 for p in chosen): continue
        chosen.append(pt)
    sites = []
    across = np.arange(-90, 91, 1)
    along = np.arange(-40, 41, 5)
    grid = np.stack(np.meshgrid(across, along, indexing='ij'), axis=-1)
    for pt in chosen:
        best = None
        for deg in range(0, 180, 10):
            site = dict(center=pt.tolist(), angle=math.radians(deg))
            pts = world(site, grid).astype(int)
            if pts[:,:,0].min()<0 or pts[:,:,0].max()>=1600 or pts[:,:,1].min()<0 or pts[:,:,1].max()>=1200: continue
            w = water[pts[:,:,0], pts[:,:,1]]
            if not w[90,:].all(): continue
            bounds = []
            for j in range(len(along)):
                lo = hi = 90
                while lo>0 and w[lo-1,j]: lo-=1
                while hi<180 and w[hi+1,j]: hi+=1
                bounds.append((across[lo]-.5, across[hi]+.5))
            left, right = min(b[0] for b in bounds), max(b[1] for b in bounds)
            interior = min(b[1] for b in bounds)-max(b[0] for b in bounds)
            envelope = right-left
            if envelope>62 or interior<32: continue
            mid=(left+right)/2
            site['center']=world(site,[mid,0]).tolist()
            site.update(width=envelope, common_water_width=interior)
            # A whole bait retreat lane, not only two dry initial points.
            bank = np.array([(side*x,y) for side in [-1,1]
                for x in np.arange(envelope/2+2,envelope/2+65,5)
                for y in range(-35,36,5)])
            pts=world(site,bank).astype(int)
            if pts[:,0].min()<35 or pts[:,0].max()>1565 or pts[:,1].min()<35 or pts[:,1].max()>1165: continue
            if not dry[pts[:,0],pts[:,1]].all(): continue
            quality=interior-.7*envelope
            if best is None or quality>best[0]: best=(quality,site)
        if best:
            site=best[1]
            if not any(np.linalg.norm(np.array(site['center'])-s['center'])<65 for s in sites): sites.append(site)
    return sites

def qualify(env, site):
    """Clearance sampled throughout the operational corridor, conservatively."""
    B=site['width']/2
    points=np.array([(x,y) for x in np.arange(-B-65,B+66,8)
        for y in range(-40,41,8)])
    obstructed=sum(env._in_obstacle(tuple(p), 10, env.obstacles) for p in world(site,points))
    trees=[[],[]]
    fruits=[[],[]]
    for i,side in enumerate([-1,1]):
        home=world(site,[side*(B+95),0])
        for t in env.trees:
            uv=local(site,[t.x,t.y])
            if side*uv[0]>B+10 and np.linalg.norm(np.array([t.x,t.y])-home)<120:
                trees[i].append(dict(position=[t.x,t.y],age=t.age))
        for f in env.fruits:
            uv=local(site,[f.x,f.y])
            if side*uv[0]>B+10 and np.linalg.norm(np.array([f.x,f.y])-home)<120:
                fruits[i].append(dict(position=[f.x,f.y],energy=f.energy))
    return dict(**site, obstructed_samples=obstructed,
                nearby_native_trees=trees, nearby_native_fruits=fruits)

def survey(start,stop):
    from src.core import SimulationCore
    rows=[]
    for seed in range(start,stop):
        biome=Map_generator(1600,1200,random.Random(seed),num_biomes=10).generate()
        sites=find_sites(biome)
        if sites:
            env=SimulationCore(seed=seed).env
            sites=[qualify(env,s) for s in sites]
        rows.append(dict(seed=seed,sites=sites))
        print(seed,len(sites),sum(s['obstructed_samples']==0 for s in sites),flush=True)
        save(f'sites-{start}-{stop-1}',rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',type=int,default=1);p.add_argument('--stop',type=int,default=41)
    a=p.parse_args();survey(a.start,a.stop)
