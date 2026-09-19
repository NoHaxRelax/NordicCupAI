"""Aggregate 30+1 native runs, including rear-side and preload failures."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from guide_batch_report import rate


def report(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    cases=[json.loads(p.read_text()) for p in sorted(folder.glob('case-*/result.json'))]
    expected={r['index']:(r['seed'],r['encounter_seed']) for r in manifest['jobs']}
    actual={r['index']:(r['seed'],r['encounter_seed']) for r in cases}
    if len(cases)!=len(expected) or actual!=expected:
        raise ValueError('Batch incomplete or seed/index mismatch')
    for name,digest in manifest['source_hashes'].items():
        if hashlib.sha256((folder/'source'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Source snapshot hash mismatch: {name}')
    counts=Counter(r['outcome'] for r in cases)
    eligible=[r for r in cases if r.get('eligible_sites',0)>0]
    settled=[r for r in cases if r.get('baseline',{}).get('passed')]
    passed=[r for r in cases if r['outcome']=='delivery_pass']
    wrong_final=[r for r in cases if r.get('replacement_side_final_period') or r.get('final',{}).get('replacement_side_ids')]
    proxy=[r for r in cases if r.get('final_hold_min')==31 and r.get('final',{}).get('bait_alive')]
    end_only=Counter()
    for r in cases:
        if not r.get('baseline',{}).get('passed'):
            label=r['outcome']
        elif not r.get('final',{}).get('bait_alive'):
            label='bait_dead'
        elif r['final'].get('replacement_side_ids'):
            label='wrong_side_at_end'
        elif r.get('initial_min_held',30)<30:
            label='initial_predators_escaped'
        elif (r.get('final_hold_min')==31 and r['final']['phase']=='final_hold'
              and r['seconds']-r['deliveries'][-1]['end_time']>=30.-1e-8):
            label='delivery_pass'
        else:
            label='delivery_or_retention_failed'
        r['outcome_rear_at_end_only']=label
        end_only[label]+=1
    summary=dict(maps=len(cases),outcomes=dict(counts),
                 outcomes_rear_at_end_only=dict(end_only),
                 success_all_maps=rate(len(passed),len(cases)),
                 success_known_eligible_maps=rate(len(passed),len(eligible)),
                 success_valid_preloads=rate(len(passed),len(settled)),
                 known_eligible_sites=len(eligible),valid_preloads=len(settled),
                 wrong_side_final_period=len(wrong_final),
                 wrong_side_at_final_frame=sum(bool(r.get('final',{}).get('replacement_side_ids')) for r in cases),
                 original_retention_failures_any_outcome=sum(r.get('initial_min_held',30)<30 for r in cases),
                 all_31_held_final_period_before_side_and_original_retention_checks=len(proxy),
                 source_hashes=manifest['source_hashes'],
                 limitations=['30 predators are preloaded; delivering the initial 30 is not tested.',
                    'Bait full energy/no aging; guide energy/aging native; static edges known.',
                    'A 30-second final hold is not whole-game retention.',
                    'Rear occupancy is a geometric/hearing-distance proxy, not a live bait-replacement test.',
                    'No-site, setup errors, and computational timeouts remain in all-map denominator.',
                    'One random visible encounter per map; guide death alone is not a failure.'])
    (folder/'aggregate.json').write_text(json.dumps(summary,indent=2)+'\n')
    (folder/'cases.json').write_text(json.dumps(cases)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    summary=report(parser.parse_args().folder)
    print(json.dumps({k:v for k,v in summary.items() if k!='source_hashes'},indent=2))
