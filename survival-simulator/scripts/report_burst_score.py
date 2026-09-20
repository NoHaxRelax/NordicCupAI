"""Summarize paired pilot; intervals are descriptive, not selection-adjusted."""
import json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]/'docs/burst-score'
rows=[json.loads(l) for p in (ROOT/'pilot').glob('*/games.jsonl') for l in p.read_text().splitlines()]
names=sorted({r['label'] for r in rows}); seeds=sorted({r['seed'] for r in rows}); by={(r['label'],r['seed']):r for r in rows}
assert len(rows)==len(by)==len(names)*len(seeds)
rng=np.random.default_rng(20260920); ix=rng.integers(0,len(seeds),(10000,len(seeds)))
scores={n:np.array([by[n,s]['score'] for s in seeds]) for n in names}
def ci(x):return list(map(float,np.quantile(x[ix].mean(axis=1),[.025,.975])))
result={}
for n in names:
 rs=[by[n,s] for s in seeds];a=sum(r['harvests'] for r in rs);t=sum(r['transfers'] for r in rs)
 result[n]=dict(n=len(rs),mean=float(scores[n].mean()),ci=ci(scores[n]),gain_vs_oscar200k=float((scores[n]-scores['oscar200k']).mean()),gain_ci=ci(scores[n]-scores['oscar200k']),gain_vs_burst400k=float((scores[n]-scores['burst400k']).mean()),gain_vs_burst400k_ci=ci(scores[n]-scores['burst400k']),transfers=t,attempts=a,transfer_rate=t/a,survival=float(np.mean([r['surv'] for r in rs])),runtime=float(np.mean([r['wall'] for r in rs])),above20k=int((scores[n]>20000).sum()),minimum=float(scores[n].min()),maximum=float(scores[n].max()))
(ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Burst score pilot','', '68 shared fresh seeds (91001–91068), eight variants, 544 complete local games. 95% intervals use 10,000 paired map bootstrap resamples. These are exploratory selection results, not an independent final test.','', '| Variant | Mean score | 95% mean CI | Successful transfers | Games above 20k | Survival seconds | Runtime seconds |','|---|---:|---|---:|---:|---:|---:|']
for n,v in sorted(result.items(),key=lambda kv:-kv[1]['mean']):
 lines.append(f"| {n} | {v['mean']:,.1f} | {v['ci'][0]:,.1f}–{v['ci'][1]:,.1f} | {v['transfers']}/{v['attempts']} | {v['above20k']}/68 | {v['survival']:.1f} | {v['runtime']:.1f} |")
lines+=['','## Equal-payload comparisons','', 'All changes below are compared with `burst400k` (400,000 turns and a 50-second cooldown).','', '| Variant | Paired score gain | 95% paired CI |','|---|---:|---|']
for n in ['fast400k','lowpop400k','birth400k','population400k','biome400k']:
 v=result[n];lo,hi=v['gain_vs_burst400k_ci'];lines.append(f"| {n} | {v['gain_vs_burst400k']:+,.1f} | {lo:+,.1f} to {hi:+,.1f} |")
lines+=['','## Interpretation','', 'The best local score is `fast1m`: 1,000,000 stationary turns, zero cooldown, otherwise Oscar’s existing observation-only contact predictor and population policy. Its 597 transfers are exactly the same count as fast400k. The extra 600,000 turns add 3,000 score per transfer; this explains the entire paired score difference. Thus the major gain is payload scaling, not better survival.', '', 'The best 400k mean is lowpop400k, which allows transfers with four live agents (min_free=2) instead of eight. birth400k lowers reproduction energy reserves. Neither guarantees 20k. The pilot has only 68 maps; confirm the chosen policy on fresh maps before promoting it.', '', 'One-million-action response size and hosted viability remain unverified. Oscar documented a clean hosted 400k no-op response, not a successful hosted 1M harvest. All experiments here were local simulations; no hosted validation/evaluation was submitted. Simulation runtime excludes HTTP serialization and organizer-side processing.', '', 'Source baseline: 43e3d52. Sweep implementation: a42d067. Raw results and exact variant manifests are in pilot/. Four pods created for this pilot were deleted after all 544 unique rows were collected and verified. SSH PC jobs completed.']
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
