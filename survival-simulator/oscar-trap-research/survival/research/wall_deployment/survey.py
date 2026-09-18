"""Native map census and analytical predator-count distribution, not policy input.

The count calculation propagates the exact per-tick birth probabilities with
one-shot placement rejection integrated over each map's free spawn area. It
does not simulate creature movement, food, player survival, or a game score.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
from experiments import SimulationCore, sites, OUT, verify_source


def acceptance(env):
    # Native spawn x/y are uniform in [20,width-20] and [20,height-20].
    # _is_position_free(x,y,10,10) treats this as the top-left of a box.
    domain=box(20,20,env.width-20,env.height-20)
    blocked=unary_union([box(w.x-10,w.y-10,w.x+w.width,w.y+w.height)
                         for w in env.obstacles]).intersection(domain)
    return 1-blocked.area/domain.area


def distribution(probability, times):
    mass=np.zeros(121);mass[0]=1
    denominators=np.maximum(1,np.arange(len(mass)))
    output=[]
    for tick in range(1,round(max(times)*10)+1):
        transition=mass*(probability*.1*(tick*.1)*.0001/denominators)
        mass-=transition
        mass[1:]+=transition[:-1]
        if tick in {round(t*10) for t in times}:
            assert abs(mass.sum()-1)<1e-10
            output.append(mass.copy())
    return np.array(output)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--maps',type=int,default=50)
    ap.add_argument('--start',type=int,default=101);args=ap.parse_args()
    verify_source();rows=[];times=[30,60,120,180,300,600,900,1800,3000]
    masses=[]
    for seed in range(args.start,args.start+args.maps):
        env=SimulationCore(seed=seed).env
        candidate=sites(env);q=acceptance(env)
        all_thin=[i for i,w in enumerate(env.obstacles[4:],4)
            if (w.width<=35 and w.height>=70) or (w.height<=35 and w.width>=70)]
        row=dict(seed=seed,geometric_walls=len(all_thin),
            usable_walls=len({s['wall_index'] for s in candidate}),usable_faces=len(candidate),
            food_faces=sum(s['trees']>0 for s in candidate),spawn_acceptance=q,sites=candidate)
        rows.append(row);masses.append(distribution(q,times))
        if len(rows)%5==0: print('survey',len(rows),'/',args.maps,flush=True)
        # Retain partial progress without pretending these are game replays.
        (OUT/'scout-wall-survey.json').write_text(json.dumps(dict(scope=__doc__,maps=rows),indent=2))
    mass=np.mean(masses,axis=0);counts=np.arange(mass.shape[1])
    count_rows=[]
    for t,p in zip(times,mass):
        cumulative=np.cumsum(p)
        count_rows.append(dict(seconds=t,mean=float(p@counts),
            p10=int(np.searchsorted(cumulative,.10)),median=int(np.searchsorted(cumulative,.5)),
            p90=int(np.searchsorted(cumulative,.90)),zero=float(p[0]),
            probabilities=p.tolist()))
    result=dict(scope=__doc__,maps=rows,predator_counts=count_rows,
        assumptions=['Native default starts with zero predators.',
            'Predators persist; rest does not remove them.',
            'Counts assume the game continues to the stated time; they are not survival predictions.',
            'Probability calculation integrates uniform placement over actual obstacles, not a no-obstacle approximation.'])
    (OUT/'scout-wall-survey.json').write_text(json.dumps(result,indent=2)+'\n')
    print({key:dict(min=min(r[key] for r in rows),median=float(np.median([r[key] for r in rows])),
        max=max(r[key] for r in rows),zero=sum(r[key]==0 for r in rows))
        for key in ['geometric_walls','usable_walls','usable_faces','food_faces']},flush=True)
    print([{k:v for k,v in r.items() if k!='probabilities'} for r in count_rows],flush=True)


if __name__=='__main__':main()
