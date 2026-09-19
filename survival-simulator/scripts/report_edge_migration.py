"""Summarize the small paired pilot; keep selection uncertainty explicit."""
import json,pathlib,math
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/edge-migration'
rows=[json.loads(line) for name in ('pilot16','narrow16') for line in (OUT/name/'games.jsonl').read_text().splitlines()]
baseline={r['seed']:r for r in rows if r['arm']=='baseline'}
summary=[]
for arm in dict.fromkeys(r['arm'] for r in rows):
    rr=[r for r in rows if r['arm']==arm];assert len(rr)==16
    delta=np.array([r['score']-baseline[r['seed']]['score'] for r in rr])
    confined=np.array([r['confinement_fraction']-baseline[r['seed']]['confinement_fraction'] for r in rr])
    half=2.131449545559323*delta.std(ddof=1)/math.sqrt(len(rr))
    summary.append(dict(arm=arm,n=len(rr),mean_score=np.mean([r['score'] for r in rr]),paired_gain=delta.mean(),
                        paired_ci95=[delta.mean()-half,delta.mean()+half],
                        confined_fraction=np.mean([r['confinement_fraction'] for r in rr]),
                        paired_confinement_gain=confined.mean(),score_wins=int(sum(delta>0))))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for s in summary:print(f"| {s['arm']} | {s['mean_score']:.1f} | {s['paired_gain']:+.1f} | {s['paired_ci95'][0]:+.1f} to {s['paired_ci95'][1]:+.1f} | {s['confined_fraction']*100:.2f}% |")
