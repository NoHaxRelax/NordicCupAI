"""Verify and summarize all frozen full-game results, paired CIs and BO history."""
import sys,json,pathlib
import numpy as np
from rock320_config import *
def main():
 out=pathlib.Path(sys.argv[1]);assert(out/'complete.json').exists()
 jobs=[json.loads(l)for l in(out/'jobs.jsonl').read_text().splitlines()]
 assert len(jobs)==len({j['id']for j in jobs})==18000 # 16000 training games + 2000 three-model bundles
 expected={f'train/rock_face_steer/{it}/{seed}'for it in range(1,51)for seed in TRAIN}|{f'final/paired/{seed}'for seed in TEST}
 assert {j['id']for j in jobs}==expected
 train={n:[]for n in NAMES};final={n:[]for n in NAMES+['expanded_food_baseline','previous_rock_face']}
 for j in jobs:
  if j['id'].startswith('final/'):
   assert len(j['rows'])==3
   for r in j['rows']:final[r['model']].append(r)
  else:
   assert len(j['rows'])==1
   train[j['rows'][0]['model']].append(j['rows'][0])
 for n,rows in train.items():
  assert len(rows)==16000
  for seed in TRAIN:assert sum(r['seed']==seed for r in rows)==50
 frozen=json.loads((out/'frozen-winners.json').read_text())
 for n in NAMES:
  t=json.loads((out/f'{n}-trials.json').read_text());assert len(t)==50
  assert max(t,key=lambda r:r['score'])==frozen[n]
 for rows in final.values():
  rows.sort(key=lambda r:r['seed']);assert [r['seed']for r in rows]==TEST
 rng=np.random.default_rng(27000);ix=rng.integers(0,2000,(10000,2000));summary={}
 b=np.array([r['score']for r in final['expanded_food_baseline']])
 lines=['# Self-stuck: 50 BO iterations × 320 full maps, then 2,000 fresh maps','',
 'New rock-face winner, previous rock-face winner and unchanged expanded-local-food baseline. All three policies for each final seed ran on the same worker. The final seeds were reused from the prior campaign: this is a repeat benchmark, not a fresh holdout. Intervals are pointwise paired percentile bootstrap intervals (10,000 resamples), not multiplicity-adjusted. No tuning on final maps.','',
 '| Model | Training best | Test mean | 95% CI | Paired gain | Paired 95% CI | Policy µs/tick | Loop CPU µs/tick | Seconds/game |',
 '|---|---:|---:|---|---:|---|---:|---:|---:|']
 for n,rows in sorted(final.items(),key=lambda z:-np.mean([r['score']for r in z[1]])):
  s=np.array([r['score']for r in rows]);ci=np.quantile(s[ix].mean(1),[.025,.975]);diff=s-b;dc=np.quantile(diff[ix].mean(1),[.025,.975]);ticks=sum(r['steps']for r in rows)
  stats=dict(mean=float(s.mean()),ci95=ci.tolist(),gain=float(diff.mean()),gain_ci95=dc.tolist(),policy_us=sum(r['ns_policy']for r in rows)/ticks/1000,loop_cpu_us=sum(r['ns_loop_cpu']for r in rows)/ticks/1000,seconds_per_game=sum(r['ns_loop_wall']for r in rows)/len(rows)/1e9,mean_survival=float(np.mean([r['survival']for r in rows])))
  summary[n]=stats;t=frozen[n]['score']if n in frozen else None
  lines.append(f'| {n} | {t if t is not None else "—"} | {stats["mean"]:.1f} | {ci[0]:.1f}–{ci[1]:.1f} | {diff.mean():+.1f} | {dc[0]:+.1f}–{dc[1]:+.1f} | {stats["policy_us"]:.1f} | {stats["loop_cpu_us"]:.1f} | {stats["seconds_per_game"]:.1f} |')
 lines+=['','Training/test differences include selection optimism and map sampling. They use the same full-game initialization, unlike the earlier late-game checkpoint campaign. Policy timing includes scheduling delays; loop CPU time is per-process CPU. A tick covers the whole population.','', '![BO evolution](training-evolution.png)']
 prev=np.array([r['score']for r in final['previous_rock_face']]);new=np.array([r['score']for r in final['rock_face_steer']]);delta=new-prev;ci=np.quantile(delta[ix].mean(1),[.025,.975])
 summary['rock_face_steer']['gain_vs_previous']=float(delta.mean());summary['rock_face_steer']['gain_vs_previous_ci95']=ci.tolist()
 lines += ['',f'New versus previous rock-face winner: {delta.mean():+.1f} points, paired 95% CI [{ci[0]:+.1f}, {ci[1]:+.1f}].']
 (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,1,figsize=(10,5),sharex=True, squeeze=False)
 axs=axs.ravel()
 for ax,n in zip(axs,NAMES):
  t=json.loads((out/f'{n}-trials.json').read_text());y=[r['score']for r in t];ax.scatter(range(1,51),y,s=10,alpha=.5);ax.plot(range(1,51),np.maximum.accumulate(y),label='Best training');ax.axhline(summary[n]['mean'],color='orange',label='Frozen 2000-map mean');ax.axhline(b.mean(),color='green',linestyle='--',label='Baseline 2000-map mean');ax.set_title(n);ax.set_ylabel('Mean score');ax.legend(fontsize=8)
 axs[-1].set_xlabel('BO iteration');fig.tight_layout();fig.savefig(out/'training-evolution.png',dpi=160)
if __name__=='__main__':main()
