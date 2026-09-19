import argparse,json,pathlib,itertools
import numpy as np
def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=pathlib.Path);ap.add_argument('--seed-start',type=int,default=10001);ap.add_argument('--reference',default='previous_winner');a=ap.parse_args();root=a.directory
    rows=[];seconds=0.;reference=None
    for i in range(10):
        d=root/f'shard-{i}';done=json.loads((d/'complete.json').read_text())
        seconds+=done['elapsed_seconds'];m=json.loads((d/'manifest.json').read_text())
        assert done['games']==100*len(m['configs'])
        if reference is None:reference=m
        assert m['configs']==reference['configs'] and m['sources']==reference['sources']
        rows.extend(json.loads(s)for s in (d/'games.jsonl').read_text().splitlines())
    assert len(rows)==1000*len(reference['configs'])
    idx=np.random.default_rng(12000).integers(0,1000,size=(10000,1000));scores={};boot={};result=[]
    for name in reference['configs']:
        r=sorted((r for r in rows if r['model']==name),key=lambda x:x['seed'])
        assert [x['seed']for x in r]==list(range(a.seed_start,a.seed_start+1000))
        score=np.array([x['score']for x in r]);scores[name]=score;boot[name]=score[idx].mean(axis=1)
        steps=sum(x['steps']for x in r);assert steps>0
        entry=dict(model=name,n=1000,mean=float(score.mean()),ci95=np.quantile(boot[name],[.025,.975]).tolist(),steps=steps)
        for k in ('interface','policy','engine','loop_wall','loop_cpu'):
            entry[k+'_us_per_tick']=sum(x['ns_'+k]for x in r)/steps/1000
        result.append(entry)
    result.sort(key=lambda x:x['mean'],reverse=True);paired=[]
    for x,y in itertools.combinations(scores,2):
        delta=scores[x]-scores[y]
        paired.append(dict(a=x,b=y,mean_difference=float(delta.mean()),ci95=np.quantile(boot[x]-boot[y],[.025,.975]).tolist(),wins=int((delta>0).sum())))
    lines=['# Frozen 1000-map evaluation','',
      f"{len(reference['configs'])} frozen configurations, identical fresh world seeds {a.seed_start}–{a.seed_start+999}. 95% percentile bootstrap intervals use 10,000 shared map resamples. No tuning on this panel.",'',
      '| Model | Mean score | 95% CI | Policy µs/tick | Whole loop CPU µs/tick | Whole loop wall µs/tick |',
      '|---|---:|---|---:|---:|---:|']
    for r in result:
        lines.append(f"| {r['model']} | {r['mean']:.1f} | {r['ci95'][0]:.1f}–{r['ci95'][1]:.1f} | {r['policy_us_per_tick']:.1f} | {r['loop_cpu_us_per_tick']:.1f} | {r['loop_wall_us_per_tick']:.1f} |")
    for baseline in (a.reference,'original_baseline'):
        lines+=['',f'## Paired differences versus {baseline}','', '| Model | Mean difference | 95% paired CI |','|---|---:|---|']
        for r in result:
            n=r['model']
            if n==baseline:continue
            delta=float((scores[n]-scores[baseline]).mean());ci=np.quantile(boot[n]-boot[baseline],[.025,.975])
            lines.append(f'| {n} | {delta:+.1f} | {ci[0]:+.1f}–{ci[1]:+.1f} |')
    lines+=['','A tick means one simulation update for the whole population, not one agent action. Timing is total measured time divided by total ticks across 1000 games; initialization is excluded. Policy timing uses native elapsed clocks and includes scheduling delays; loop CPU uses per-process CPU time. All models share each pod and its 100-map shard, in randomized job order. Measurements reflect 32 concurrent workers, hardware differences and profiling overhead, not isolated production latency. Interface and engine breakdowns are in summary.json.','',
      f'Intervals are pointwise, not corrected for multiple comparisons. A top test rank alone does not establish superiority. All {len(paired)} pairwise comparisons are in paired.json.','',f'Attributed active compute: ${seconds*.96/3600:.2f}, excluding setup/storage/idle pod uptime.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n');(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'paired.json').write_text(json.dumps(paired,indent=2)+'\n')
if __name__=='__main__':main()
