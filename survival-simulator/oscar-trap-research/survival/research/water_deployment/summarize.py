"""Audit saved evidence without invoking the simulator."""
from common import *

def read(name):
    return json.loads((OUT/(name+'.json')).read_text())['data']

def summary():
    surveys=read('sites-1-40')+read('sites-41-80')
    result=dict(maps_surveyed=len(surveys),maps_with_candidates=sum(bool(m['sites']) for m in surveys),
        candidate_sites=sum(len(m['sites']) for m in surveys),
        clear_sampled_corridors=sum(s['obstructed_samples']==0 for m in surveys for s in m['sites']))
    suites=['native','relay','orchard','heldout','acquisition','native-renewal',
        'native-renewal-v3','renewal','renewal-v3','dogleg','multi','twins','association','corridor','approach-relay']
    result['suites']={}
    for name in suites:
        rows=read(name);runs=[r for r in rows if not r.get('skipped')]
        for r in runs:
            assert 0<=r['joint_fraction']<=1
            assert r['joint_fraction']<=min(r['water_fraction'],r['bait_attention_fraction'])+1e-9
            assert len(r['births'])*100==r['birth_energy_cost']
            assert all(abs(b['energy']-75)<1e-9 for b in r['births'])
        result['suites'][name]=dict(cases=len(rows),skipped=len(rows)-len(runs),
            captures=sum(r['first_capture_s'] is not None for r in runs),
            joint_at_least_99pct=sum(r['joint_fraction']>=.99 for r in runs),
            maximum_elapsed_s=max((r['elapsed_s'] for r in runs),default=0),
            handoffs=sum(len(r['handoffs']) for r in runs),births=sum(len(r['births']) for r in runs))
    result['matched_score_comparison']=[dict(site=r['case']['site_index'],strategy=r['case']['comparison'],
        score=r['score'],fallback_s=r.get('fallback_s'),total_captures=sum(d['cause']=='capture' for d in r['deaths']),
        initial_shore_workers=[dict(id=d['id'],t=d['t'],cause=d['cause']) for d in r['deaths'] if d['id'] in [2,3]])
        for r in read('protection-comparison')]
    save('summary',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':summary()
