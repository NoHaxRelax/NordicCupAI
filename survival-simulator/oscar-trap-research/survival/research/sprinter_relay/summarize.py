"""Aggregate only final, corrected evaluation suites; keep pilot failures separate."""
import json
import statistics as st
from pathlib import Path
OUT=Path(__file__).resolve().parents[2]/'results/sprinter_relay'
summary={}
for phase in ('heldout','workers','multiple','renewal-screen','replays'):
    path=OUT/(phase+'.json')
    if not path.exists():continue
    runs=json.loads(path.read_text())['runs']
    groups=[]
    for energy in (150,500):
        for policy in ('none','orbit_single','orbit'):
            rows=[r for r in runs if r['parameters']['energy']==energy and r['parameters']['policy']==policy]
            if not rows:continue
            mean=lambda key:round(st.mean(r[key] for r in rows),3)
            groups.append(dict(energy=energy,policy=policy,n=len(rows),retention=mean('active_retention'),
                worker_seconds=mean('worker_seconds'),worker_captures=sum(r['worker_captures'] for r in rows),
                bait_captures=sum(r['bait_captures'] for r in rows),bait_food=round(st.mean(r['food_energy']['bait'] for r in rows),1),
                worker_food=round(st.mean(r['food_energy']['worker'] for r in rows),1),
                movement_turn_bait=round(st.mean(r['movement_turn_energy']['bait'] for r in rows),1),
                longest_gap=round(max(r['longest_active_gap_seconds'] for r in rows),1),
                births=sum(len(r['births']) for r in rows),prepared_newborns=sum(len(r['newborn_prepared']) for r in rows),
                actual_switches=sum(r['actual_target_switches'] for r in rows)))
    summary[phase]=dict(runs=len(runs),groups=groups)
    print(phase)
    for row in groups:print(row)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
