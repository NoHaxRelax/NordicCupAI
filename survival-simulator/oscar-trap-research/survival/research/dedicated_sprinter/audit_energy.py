"""Audit historical final metrics; no simulation/controller changes.
The engine skips the age debit when the base living debit already kills the
agent. Early diagnostics counted that skipped debit as received nutrition.
Correct only provable non-contact starvation records, and annotate correction.
New experiment.py computes this branch correctly before the engine step.
"""
from pathlib import Path
import json
OUT=Path(__file__).resolve().parents[2]/'results/dedicated_sprinter'
def audit():
    report=[]
    for p in sorted(OUT.glob('*-v5.json')):
        data=json.loads(p.read_text());changed=False
        for r in data.get('runs',[]):
            if 'food_received' not in r:continue
            if r.get('accounting_version')!='source-order-v2' and 'diagnostic_correction' not in r and r.get('death')=='starvation' and r['duration']>r['scenario'].get('max_age',90):
                assert r['min_gap']>=15,'Ambiguous cause; rerun rather than infer.'
                skipped=.01*r['duration']
                r['passive_energy']=round(r['passive_energy']-skipped,4)
                r['food_received']=round(r['food_received']-skipped,4)
                r['diagnostic_correction']='Removed skipped final aging debit from passive energy and inferred food receipt; simulation outcome unchanged.'
                changed=True
            if r.get('accounting_version')!='source-order-v2':r['accounting_version']='source-order-v2';changed=True
            residual=r['scenario'].get('energy',500)+r['food_received']-r['movement_energy']-r['turn_energy']-r['passive_energy']-r['remaining_energy']
            assert abs(residual)<.002,(p.name,residual)
            assert r['food_received']<=r['nutrition']+.002,(p.name,r['food_received'],r['nutrition'])
            report.append(dict(file=p.name,policy=r['policy'],residual=round(residual,6)))
        if changed:p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    (OUT/'energy-audit.json').write_text(json.dumps(dict(records=len(report),checks=report),indent=2)+'\n')
    print('Energy balance and nutrition caps verified:',len(report),'records')
if __name__=='__main__':audit()
