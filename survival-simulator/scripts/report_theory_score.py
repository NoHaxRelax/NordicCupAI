"""Merge paired shards and report bootstrap uncertainty."""
import argparse,json,pathlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True)
a=p.parse_args();rows=[]
for path in sorted(a.input.glob('*/games.jsonl')):
    rows.extend(json.loads(x) for x in path.read_text().splitlines())
models=sorted({x['model'] for x in rows});seeds=sorted({x['seed'] for x in rows})
table={(x['model'],x['seed']):x for x in rows}
if any((m,s) not in table for m in models for s in seeds):raise SystemExit('Incomplete paired panel')
rng=np.random.default_rng(72026);idx=rng.integers(0,len(seeds),(10000,len(seeds)))
result={}
base=np.array([table['expanded',s]['score'] for s in seeds])
for m in models:
    score=np.array([table[m,s]['score'] for s in seeds]);gain=score-base
    ci=np.percentile(score[idx].mean(1),[2.5,97.5]);gci=np.percentile(gain[idx].mean(1),[2.5,97.5])
    result[m]=dict(n=len(seeds),mean_score=float(score.mean()),score_ci=list(map(float,ci)),
      paired_gain=float(gain.mean()),gain_ci=list(map(float,gci)),
      mean_survival=float(np.mean([table[m,s]['survival'] for s in seeds])),
      mean_game_seconds=float(np.mean([table[m,s]['game_seconds'] for s in seeds])),
      mean_predation_deaths=float(np.mean([table[m,s]['predation_deaths'] for s in seeds])))
a.out.mkdir(parents=True,exist_ok=True)
(a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Theoretically motivated score changes','',f'{len(seeds)} fresh paired full games. Intervals are 95% paired bootstrap intervals (10,000 map resamples).','',
       '| Model | Mean score | Score CI | Paired gain | Gain CI | Survival | Seconds/game | Predator deaths |','|---|---:|---|---:|---|---:|---:|---:|']
for m,r in sorted(result.items(),key=lambda x:-x[1]['mean_score']):
    lines.append(f"| {m} | {r['mean_score']:.1f} | {r['score_ci'][0]:.1f}–{r['score_ci'][1]:.1f} | {r['paired_gain']:+.1f} | {r['gain_ci'][0]:+.1f}–{r['gain_ci'][1]:+.1f} | {r['mean_survival']:.1f} | {r['mean_game_seconds']:.2f} | {r['mean_predation_deaths']:.1f} |")
(a.out/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
