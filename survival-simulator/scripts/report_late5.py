"""Verify every job and report paired scores and runtime eligibility."""
import json,pathlib,sys,numpy as np
from late5_config import NAMES,TRAIN,TEST,ITERATIONS
p=pathlib.Path(sys.argv[1]);assert(p/'complete.json').exists()
jobs=[json.loads(l)for l in(p/'jobs.jsonl').read_text().splitlines()];byid={j['id']:j for j in jobs}
expected={f'train/{name}/{it}/{seed}'for name in NAMES for it in range(1,ITERATIONS+1)for seed in TRAIN}|{f'final/paired/{seed}'for seed in TEST}
assert len(jobs)==len(byid)==len(expected)==10000 and set(byid)==expected
winners=json.loads((p/'frozen-winners.json').read_text());names=list(winners)+['expanded_food_baseline'];rows={n:[]for n in names}
for n in NAMES:
 trials=json.loads((p/f'{n}-trials.json').read_text());assert len(trials)==ITERATIONS
 for it,t in enumerate(trials,1):
  rr=[byid[f'train/{n}/{it}/{seed}']['rows'][0]for seed in TRAIN]
  assert all(r['seed']==s and r['model']==n for r,s in zip(rr,TRAIN))
  assert abs(np.mean([r['score']for r in rr])-t['score'])<1e-8
  assert abs(np.mean([r['game_seconds']for r in rr])-t['mean_game_seconds'])<1e-8
  assert t['eligible']==(t['mean_game_seconds']<20)
 eligible=[t for t in trials if t['eligible']]
 assert (n in winners)==bool(eligible)
 if eligible:assert winners[n]==max(eligible,key=lambda t:t['score'])
for seed in TEST:
 rr=byid[f'final/paired/{seed}']['rows'];assert len(rr)==len(names) and {r['model']for r in rr}==set(names)
 for r in rr:assert r['seed']==seed;rows[r['model']].append(r)
assert sum(len(j['rows'])for j in jobs)==8000+2000*len(names)
rng=np.random.default_rng(45001);ix=rng.integers(0,len(TEST),(10000,len(TEST)));base=np.array([r['score']for r in rows['expanded_food_baseline']]);summary={}
def interval(v):return np.quantile(v[ix].mean(1),[.025,.975]).tolist()
lines=['# Late activation: 16 BO trials ×100 full maps, then2000 fresh maps','', 'Base is expanded local food. Runtime includes initialization and result processing. Intervals are pointwise paired bootstrap intervals (10,000 resamples), not adjusted for multiple comparisons. No final-map tuning.','', '| Strategy | Training best | Final mean | Score95% CI | Paired gain95% CI | Mean seconds/game | Under20s | CPU µs/population tick |','|---|---:|---:|---|---|---:|---|---:|']
for n in sorted(names,key=lambda n:-np.mean([r['score']for r in rows[n]])):
 rr=rows[n];score=np.array([r['score']for r in rr]);runtime=np.array([r['game_seconds']for r in rr]);ci=interval(score);delta=score-base;dc=interval(delta);ticks=sum(r['steps']for r in rr)
 summary[n]=dict(mean=float(score.mean()),ci95=ci,gain=float(delta.mean()),gain_ci95=dc,mean_game_seconds=float(runtime.mean()),game_seconds_ci95=interval(runtime),runtime_eligible=bool(runtime.mean()<20),loop_cpu_us=sum(r['ns_loop_cpu']for r in rr)/ticks/1000,policy_us=sum(r['ns_policy']for r in rr)/ticks/1000,mean_survival=float(np.mean([r['survival']for r in rr])))
 t=winners[n]['score']if n in winners else None
 lines.append(f'| {n} | {t if t is not None else "—"} | {score.mean():.1f} | {ci[0]:.1f}–{ci[1]:.1f} | {delta.mean():+.1f} [{dc[0]:+.1f}, {dc[1]:+.1f}] | {runtime.mean():.2f} | {"yes"if runtime.mean()<20 else "NO: excluded"} | {summary[n]["loop_cpu_us"]:.1f} |')
lines+=['','## Selected activation rules','', '| Strategy | Time threshold (ticks / seconds) | Population below | Logic | Persistence |','|---|---|---:|---|---|']
for n,w in winners.items():
 c=w['config'];logic=['time','population','AND','OR'][int(c['gate_logic'])];persist=['reversible','latched','one episode'][int(c['gate_persistence'])]
 lines.append(f'| {n} | {c["gate_ticks"]:.0f} / {c["gate_ticks"]*.1:.1f} | {c["gate_population"]:.0f} | {logic} | {persist} |')
lines+=['','Strategies failing the final runtime limit are not recommended. A reversible switch restores settings but retains learned map information. Training/test gaps include selection optimism and map differences.','', '![BO evolution](training-evolution.png)']
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(p/'RESULTS.md').write_text('\n'.join(lines)+'\n')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(5,1,figsize=(10,13),sharex=True)
for ax,n in zip(axs,NAMES):
 t=json.loads((p/f'{n}-trials.json').read_text());ax.scatter(range(1,17),[r['score']for r in t],c=['tab:blue'if r['eligible']else 'red'for r in t],s=18,label='Training (red: too slow)');ax.plot(range(1,17),[r['best']if r['best']is not None else np.nan for r in t],label='Best eligible training')
 if n in summary:ax.axhline(summary[n]['mean'],color='orange',label='Final mean')
 ax.axhline(base.mean(),color='green',ls='--',label='Final baseline');ax.set_title(n);ax.legend(fontsize=7)
axs[-1].set_xlabel('BO iteration');fig.tight_layout();fig.savefig(p/'training-evolution.png',dpi=150)
print(json.dumps(summary,indent=2))
