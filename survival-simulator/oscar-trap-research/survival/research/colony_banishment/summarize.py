"""Paired summaries. Full-run, intention-to-treat comparisons keep bait costs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics as st


def summarize(paths):
    out=[]
    for path in paths:
        data=json.loads(Path(path).read_text());groups=defaultdict(list)
        for row in data['runs']:groups[row['scenario']].append(row)
        for scenario,rows in groups.items():
            by={(r['seed'],r['mode']):r for r in rows}
            for control,treatment in [('control','banish'),('oracle-control','oracle'),('nursery','control')]:
                seeds=sorted(seed for seed,mode in by if mode==control and (seed,treatment) in by)
                if not seeds:continue
                pairs=[]
                for seed in seeds:
                    c,b=by[seed,control],by[seed,treatment]
                    cm,bm=c['metrics'],b['metrics']
                    pairs.append(dict(seed=seed,score_delta=b['score']-c['score'],duration_delta=b['duration']-c['duration'],
                        harvest_delta=bm['harvest_energy']-cm['harvest_energy'],capture_delta=bm['captures']-cm['captures'],
                        worker_exposure_delta=bm['worker_exposure_100']-cm['worker_exposure_100'],
                        worker_exposure_rate_delta=100*(bm['worker_exposure_100']/max(1,bm['worker_seconds'])-cm['worker_exposure_100']/max(1,cm['worker_seconds'])),
                        worker_seconds_delta=bm['worker_seconds']-cm['worker_seconds'],
                        identical_initial_rng=c['initial_rng_sha256']==b['initial_rng_sha256']))
                rng=random.Random(917)
                boot=sorted(st.mean(rng.choices([r['score_delta'] for r in pairs],k=len(pairs))) for _ in range(4000))
                means={k:round(st.mean(p[k] for p in pairs),4) for k in pairs[0] if k not in ('seed','identical_initial_rng')}
                group=dict(file=str(path),scenario=scenario,control=control,treatment=treatment,n=len(seeds),
                    score_wins=sum(p['score_delta']>1e-6 for p in pairs),score_ties=sum(abs(p['score_delta'])<=1e-6 for p in pairs),
                    mean_deltas=means,score_mean_bootstrap95=[round(boot[100],3),round(boot[3899],3)],
                    bootstrap_note='Descriptive paired seed bootstrap; small exploratory samples, not confirmatory inference.',pairs=pairs)
                arm={}
                for mode in (control,treatment):
                    rr=[by[seed,mode] for seed in seeds]
                    returns=[rel['return_t']-rel['t'] for r in rr for rel in r['releases'] if rel['return_t'] is not None]
                    arm[mode]=dict(score=st.mean(r['score'] for r in rr),duration=st.mean(r['duration'] for r in rr),
                        reached_horizon=sum(r['duration']>=r['horizon']-.05 for r in rr),
                        harvest=st.mean(r['metrics']['harvest_energy'] for r in rr),
                        captures=st.mean(r['metrics']['captures'] for r in rr),
                        worker_captures=st.mean(r['metrics']['worker_captures'] for r in rr),
                        guide_captures=st.mean(r['metrics']['guide_captures'] for r in rr),
                        worker_seconds=st.mean(r['metrics']['worker_seconds'] for r in rr),
                        worker_exposure=st.mean(r['metrics']['worker_exposure_100'] for r in rr),
                        guide_seconds=st.mean(r['metrics']['guide_seconds'] for r in rr),
                        guide_movement_energy=st.mean(r['metrics']['guide_move_energy'] for r in rr),
                        guide_target_seconds=st.mean(r['metrics']['guide_target_seconds'] for r in rr),
                        guides=sum(e['kind']=='guide_selected' for r in rr for e in r['events']),
                        releases=sum(len(r['releases']) for r in rr),returned=len(returns),
                        observed_return_delay_median=st.median(returns) if returns else None,
                        boundary_violations=sum(r['metrics']['physical_boundary_violations'] for r in rr))
                group['arms']=arm;out.append(group)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');p.add_argument('--out');a=p.parse_args()
    result=summarize(a.paths)
    if a.out:Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    for r in result:print(json.dumps({k:r[k] for k in ('file','scenario','control','treatment','n','score_wins','mean_deltas','score_mean_bootstrap95')}))
