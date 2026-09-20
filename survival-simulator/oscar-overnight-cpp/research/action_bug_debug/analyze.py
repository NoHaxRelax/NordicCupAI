"""Summarize C++ traces without guessing causes from score deltas."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def analyze(b):
    t=b['decision_t']; farm=b['farm']
    rows=[r for r in b['trace'] if abs(r['t']-t)<1e-7]
    agents={r['agent']:r for r in rows if r['kind']=='after_actions'}
    f=agents[farm]
    dist=lambda p: math.hypot(p['x']-f['x'],p['y']-f['y'])
    preds=[]
    for p in [r for r in rows if r['kind']=='pred_before']:
        q=dict(p,d0=dist(p))
        rr=[r for r in rows if r.get('pred')==p['pred']]
        q['slept']=any(r['kind']=='rest_skip' for r in rr)
        q['target']=next((r['agent'] for r in rr if r['kind']=='target'),None)
        q['target_angle']=next((r['angle'] for r in rr if r['kind']=='target'),None)
        q['after']=next((r for r in rr if r['kind']=='pred_after'),None)
        q['d1']=dist(q['after']) if q['after'] else q['d0']
        q['move']=next((r for r in rr if r['kind']=='move'),None)
        if q['move']:
            m=q['move']
            q['desired_d1']=math.hypot(m['desired_x']-f['x'],m['desired_y']-f['y'])
        preds.append(q)
    nearest=min(preds,key=lambda p:p['d0']) if preds else None
    pursuers=[p for p in preds if p['target']==farm]
    closest_pursuer=min(pursuers,key=lambda p:p['d1']) if pursuers else None
    skipped=any(r['kind']=='skip' and r['skipped']==farm for r in rows)
    if b['outcome']=='predator': reason='transferred'
    elif any(r['kind']=='death' and r['agent']==farm for r in rows): reason='skip_failed'
    elif nearest and nearest['slept'] and not pursuers: reason='sleeping_predator'
    elif not pursuers: reason='different_target_or_no_target'
    elif closest_pursuer['d0'] - closest_pursuer['move']['distance'] >= 15:
        reason='insufficient_reach'
    else: reason='steering_geometry'
    return dict(seed=b['seed'],t=t,farm=farm,doomed=b['doomed'],observed_d=b['pred_d'],
        skipped=skipped,outcome=b['outcome'],reason=reason,
        death_delay=round(b.get('death_t',t)-t,3),nearest=nearest,pursuer=closest_pursuer,
        predators=preds)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('directory',type=Path); args=ap.parse_args()
    rows=[]
    for file in sorted(args.directory.glob('*-s*.json')):
        data=json.loads(file.read_text())
        if not data['summary']['trace_enabled']: continue
        rows.extend(analyze(b) for b in data['bursts'])
    print('Counts:',json.dumps(Counter(r['reason'] for r in rows)))
    print('Skip/death delay:',json.dumps(Counter(str((r['skipped'],r['death_delay'],r['outcome'])) for r in rows)))
    for r in rows:
        if r['reason']=='transferred': continue
        p=r['pursuer'] or r['nearest']
        print(json.dumps(dict(seed=r['seed'],t=round(r['t'],1),farm=r['farm'],reason=r['reason'],observed_d=r['observed_d'],
            d0=round(p['d0'],2),d1=round(p['d1'],2),target=p['target'],angle=p['target_angle'],rest=p['slept'],
            move=p.get('move'),desired_d1=p.get('desired_d1'))))
