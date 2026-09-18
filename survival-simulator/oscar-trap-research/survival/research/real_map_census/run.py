"""Static native-map census. No actors, simulation steps, or hidden policy inputs."""
from pathlib import Path
import sys, json, math, os
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
sys.path[:0]=[str(ROOT/'vendor/survival-simulator'),str(ROOT/'research')]
from src.core import SimulationCore
from real_map_guide_sol.policy import RealMapGuidePolicy
OUT=ROOT/'results/real_map_census'

def examine(seed):
    if Path('/tmp/predator-intake-stop').exists():return None
    saved=OUT/f'map-{seed}.json'
    if saved.exists():
        rects=json.loads(saved.read_text())['obstacles'];width,height=1600,1200
    else:
        env=SimulationCore(starting_agents=0,starting_predators=0,seed=seed).env
        rects=[(float(o.x),float(o.y),float(o.width),float(o.height)) for o in env.obstacles]
        width,height=env.width,env.height
    # Reuse only pure geometric predicates, never execute a controller.
    g=object.__new__(RealMapGuidePolicy);g.width=width;g.height=height;g.rects=rects
    candidates=[]
    for i,a in enumerate(rects):
        for j,b in enumerate(rects):
            if i==j:continue
            for axis in (0,1):
                ax,ay,aw,ah=a if axis==0 else (a[1],a[0],a[3],a[2])
                bx,by,bw,bh=b if axis==0 else (b[1],b[0],b[3],b[2])
                gap=bx-ax-aw;lo=max(ay,by);hi=min(ay+ah,by+bh);overlap=hi-lo
                if not(10.9<=gap<=19.1 and overlap>=10.1):continue
                cross=ax+aw+gap/2
                xy=lambda u,v:(u,v) if axis==0 else(v,u)
                for sign in (1,-1):
                    mouth=xy(cross,lo if sign==1 else hi);direction=xy(0,sign)
                    point=lambda depth:(mouth[0]+direction[0]*depth,mouth[1]+direction[1]*depth)
                    # Stop corridor check just before the first expanded wall
                    # face, not inside its intended blocking region. The two
                    # faces can be offset; rectangle intersection is closed.
                    face_offset=(lo-min(ay,by)) if sign==1 else (max(ay+ah,by+bh)-hi)
                    goal=point(5);far=point(-125);hold=point(-75);front=point(-face_offset-11.02)
                    if not g._free(goal,5.01):continue
                    staging=g._free(far,11.01) and g._free(hold,11.01) and g._clear(far,hold,11.01)
                    corridor=staging and g._clear(hold,front,11.01)
                    candidates.append(dict(pair=[i,j],mouth=mouth,inward=direction,gap=gap,overlap=overlap,
                                           conservative=overlap>=54.9,staging=staging,corridor=corridor,
                                           interior_pair=i>=4 and j>=4))
    row=dict(seed=seed,obstacles=rects,candidates=candidates,
             conservative_refuge=any(c['conservative'] for c in candidates),
             conservative_policy_site=any(c['conservative'] and c['staging'] for c in candidates),
             conservative_clear_approach=any(c['conservative'] and c['corridor'] for c in candidates),
             conservative_interior_refuge=any(c['conservative'] and c['interior_pair'] for c in candidates),
             expanded_refuge=bool(candidates),expanded_clear_approach=any(c['corridor'] for c in candidates))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'map-{seed}.json').write_text(json.dumps(row)+'\n')
    return row

def interval(k,n):
    z=1.96;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [round(100*(c-r),1),round(100*(c+r),1)]

if __name__=='__main__':
    seeds=list(range(10000,10128));rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(examine,seeds):
            if row:rows.append(row)
            if len(rows)%16==0:print({'maps_complete':len(rows)},flush=True)
    metrics={}
    for key in ['conservative_refuge','conservative_interior_refuge','conservative_policy_site','conservative_clear_approach','expanded_refuge','expanded_clear_approach']:
        k=sum(r[key] for r in rows);metrics[key]=dict(maps=k,total=len(rows),percent=round(100*k/len(rows),1),wilson95_percent=interval(k,len(rows)))
    summary=dict(seed_range=[seeds[0],seeds[-1]],depth=5,gap=[10.9,19.1],conservative_overlap=54.9,
                 expanded_overlap=10.1,notes='Static geometric availability only; short overlaps unvalidated; no delivery/retention success implied. No simulation was stepped.',metrics=metrics)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
