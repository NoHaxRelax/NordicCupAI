"""Build compact, auditable summaries from retained final-controller results."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/wall_deployment'
FILES=['natural-renewal-v7.json','natural-native-v7.json','fullgame-v7.json',
       'fullgame-fresh-heldout-v7.json','protected-side-stress-v7.json','rest-acquisition.json']


def main():
    summary=[]
    for name in FILES:
        data=json.loads((OUT/name).read_text())
        rows=[]
        for r in data['runs']:
            if r.get('skipped'):
                rows.append({k:r[k] for k in ('seed','mode','skipped')});continue
            keys=('seed','mode','wall','width','height','heading','initial_predator_energy','seconds','alive',
                  'score','captures','food_count','food_energy','max_generation','held_fraction',
                  'first_hold_loss','success','tail_hold','pose_error_mean','pose_error_max','legal_actions')
            row={k:r[k] for k in keys if k in r}
            row['policy_metrics']=r['policy_metrics']
            if 'births' in r:
                row['births']=len(r['births'])
                row['late_trace_predators']=r['trace'][-1]['predators']
                predicted=r['seconds']+r['food_energy']/1000-sum(d['energy']/100 for d in r['deaths'] if d['cause']=='capture')
                row['score_accounting_error']=round(r['score']-predicted,5)
                assert abs(row['score_accounting_error'])<.002
            rows.append(row)
        summary.append(dict(file=name,policy_sha256=data.get('policy_sha256'),runs=rows))
    value=dict(source_commit='acfc31a4003a5f91bf11032a02cd98c178ddbd7e',
        controller_sha256=hashlib.sha256((ROOT/'research/wall_deployment/controller.py').read_bytes()).hexdigest(),
        retained_final_runs=sum(len(r['runs']) for r in summary),suites=summary,
        limitations=['Arranged sites are not full-game acquisition.',
                     'Native unordered object sets and divergent RNG histories limit exact seed pairing.',
                     'Final 180-second generated runs are prefixes, not completed 3000-second games.',
                     'True coordinates are used only for fixtures, rendering and evaluation.'])
    (OUT/'summary.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(retained_final_runs=value['retained_final_runs'],
        suites=[dict(file=s['file'],runs=len(s['runs'])) for s in summary]),indent=2))


if __name__=='__main__':main()
