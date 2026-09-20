"""Aggregate saved local comparisons; never calls the competition API."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def load(directory):
    return [json.loads(p.read_text()) for p in sorted(directory.glob('*-s*.json'))]


def metrics(runs):
    ss=[r['summary'] for r in runs]
    outcomes=Counter()
    for s in ss: outcomes.update(s['outcomes'])
    attempts=sum(s['bursts'] for s in ss)
    return dict(games=len(ss),seeds=sorted(s['seed'] for s in ss),bursts=attempts,
                transfers=outcomes['predator'],failed=attempts-outcomes['predator'],
                transfer_rate=outcomes['predator']/attempts if attempts else None,
                mean_score=statistics.mean(s['score'] for s in ss),
                mean_survival=statistics.mean(s['survival'] for s in ss),
                public_confirmed=sum(s.get('public_confirmed') or 0 for s in ss),
                max_game_p99_admission_ms=max((s.get('admission_ms',{}).get('p99',0) for s in ss),default=0),
                max_admission_ms=max((s.get('admission_ms',{}).get('maximum',0) for s in ss),default=0))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); ap.add_argument('--out',type=Path)
    args=ap.parse_args(); cohorts={}
    for directory in sorted(args.root.glob('contact-*')):
        if not directory.is_dir(): continue
        runs=load(directory)
        if not runs: continue
        cohorts[directory.name]={mode:metrics([r for r in runs if r['summary']['selector']==mode])
                                 for mode in sorted({r['summary']['selector'] for r in runs})}
    old=load(args.root.parent/'debug-20260920-3Q8ISQ/results')
    cohorts['prior-diagnostic-baseline']={'contact_nearest':metrics(old)}
    result=dict(scope='local mypc C++ games; no API transport',cohorts=cohorts)
    if args.out: args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
