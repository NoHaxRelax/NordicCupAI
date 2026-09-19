"""Summarize a completed guide_batch run without excluding failed maps."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics


def rate(successes, total):
    if not total:
        return dict(successes=successes, total=total, fraction=None, wilson_95=None)
    p, z = successes/total, 1.959963984540054
    center = (p+z*z/(2*total))/(1+z*z/total)
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return dict(successes=successes, total=total, fraction=p,
                wilson_95=[center-half, center+half])


def report(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    results = [json.loads(p.read_text()) for p in sorted(folder.glob('case-*/result.json'))]
    by_index = {r['index']:r for r in results}
    expected = manifest['jobs']
    if len(by_index)!=len(results) or len(results)!=len(expected):
        raise ValueError(f'Expected {len(expected)} unique results, found {len(results)}')
    for job in expected:
        r = by_index[job['index']]
        if any(r[k]!=v for k,v in job.items()):
            raise ValueError(f'Seed/index mismatch at {job["index"]}')
    audit_path = folder/'setup_audit.json'
    audits = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    for r in results:
        if str(r['index']) in audits:
            audit = audits[str(r['index'])]
            if any(audit[k]!=r[k] for k in ('seed','encounter_seed')):
                raise ValueError('Setup audit seed mismatch')
            r['eligible_sites'] = audit['eligible_sites']
            r['setup_audited_separately'] = True
        retry = list((folder/'retries'/f"case-{r['index']:04}").glob('map-*/summary.json'))
        if retry:
            completed = json.loads(retry[0].read_text())
            if any(completed[k]!=r[k] for k in ('seed','encounter_seed')):
                raise ValueError('Retry seed mismatch')
            r['original_attempt_outcome'] = r['outcome']
            r.update(completed)
    outcomes = Counter(r['outcome'] for r in results)
    eligible = [r for r in results if r.get('eligible_sites',0)>0]
    delivered = [r for r in results if r['outcome']=='delivery_proxy_pass']
    delivered_alive = [r for r in delivered if r['final']['guide_alive']]
    alive = [r for r in eligible if r.get('final',{}).get('guide_alive')]
    measured = [r['contact_metrics'] for r in results if 'contact_metrics' in r]
    alive_ticks = sum(r['alive_ticks'] for r in measured)
    detected_ticks = sum(r['detectable_ticks'] for r in measured)
    summary = dict(
        maps=len(results), outcomes=dict(outcomes),
        bait_site_available=rate(len(eligible),len(results)),
        delivery_all_maps=rate(len(delivered),len(results)),
        delivery_eligible_maps=rate(len(delivered),len(eligible)),
        delivery_with_guide_alive_all_maps=rate(len(delivered_alive),len(results)),
        delivery_with_guide_alive_eligible_maps=rate(len(delivered_alive),len(eligible)),
        guide_alive_at_run_end=len(alive),
        delivered_guide_alive=len(delivered_alive), delivered_guide_dead=len(delivered)-len(delivered_alive),
        guide_caught=sum(bool(r.get('guide_caught')) for r in eligible),
        alive_ticks=alive_ticks, predator_sensed_guide_ticks=detected_ticks,
        predator_sensed_guide_fraction=detected_ticks/alive_ticks if alive_ticks else None,
        runs_always_detectable=sum(r['detectable_ticks']==r['alive_ticks'] for r in measured),
        contact_metrics_completed_runs=len(measured),
        median_delivery_sim_seconds=statistics.median(r['seconds'] for r in delivered) if delivered else None,
        source_hashes=manifest['source_hashes'],
        limitations=[
            'One arranged, initially visible encounter per randomly seeded native map; one chosen bait site.',
            '60-second simulation horizon, stopping early after a 10-second continuous delivery proxy.',
            'Delivery proxy: tracked predator within 40 units of bait and sensing it; not full-game retention or causal proof.',
            'Guide has native energy/aging; bait has full energy and disabled aging. Static edges/localization are known.',
            'Guide survival means alive when the run ends; successful runs can end before 60 seconds.',
            'Contact fraction weights alive ticks, which are correlated; no binomial interval is claimed for ticks.',
            'Maps lacking a usable bait site remain in the all-map denominator. Errors/timeouts remain failures.',
        ])
    (folder/'aggregate.json').write_text(json.dumps(summary,indent=2)+'\n')
    (folder/'cases.json').write_text(json.dumps(results)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    args=parser.parse_args()
    summary=report(args.folder)
    print(json.dumps({k:v for k,v in summary.items() if k!='source_hashes'},indent=2))
