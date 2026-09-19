"""Render a completed avoidance16 experiment, paired intervals, and score plots."""
import argparse, collections, gzip, json, pathlib, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('results');ap.add_argument('--out',required=True);a=ap.parse_args()
    root=pathlib.Path(a.results);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    m=json.loads((root/'manifest.json').read_text());done=json.loads((root/'completion.json').read_text())
    rows=[json.loads(line) for line in (root/'games.jsonl').read_text().splitlines()]
    assert len(rows)==done['games']==16*m['rounds']*8+17*32+3*64
    assert len({(r['stage'],r['label'],r['seed']) for r in rows})==len(rows)
    ideas={x['id']:x['idea'] for x in m['families']};ideas['baseline']='Unchanged Oscar Orchard avoidance'
    groups=collections.defaultdict(dict)
    for r in rows:groups[(r['stage'],r['label'])][r['seed']]=r
    summary={'games':len(rows),'elapsed_seconds':done['elapsed_seconds'],'validation':{},'confirmation':{}}
    rng=np.random.default_rng(98017)
    for stage,n,alpha in [('validation',32,.05),('confirmation',64,.025)]:
        seeds=m[stage+'_seeds'];base=groups[(stage,'baseline')];indices=rng.integers(0,n,size=(20000,n))
        for (st,label),data in groups.items():
            if st!=stage:continue
            assert sorted(data)==seeds
            scores=np.array([data[s]['score'] for s in seeds]);d=scores-np.array([base[s]['score'] for s in seeds]);boot=d[indices].mean(axis=1)
            summary[stage][label]={'idea':ideas[label],'n':n,'mean':float(scores.mean()),'median':float(np.median(scores)),'std':float(scores.std(ddof=1)),'delta':float(d.mean()),'paired_interval':np.quantile(boot,[alpha/2,1-alpha/2]).tolist(),'interval_level':1-alpha,'wins':int(sum(d>0)),'ties':int(sum(d==0)),'mean_predator_deaths':float(np.mean([v['pdeaths'] for v in data.values()])),'mean_runtime_seconds':float(np.mean([v['wall'] for v in data.values()]))}
    confirm=summary['confirmation'];winner=max(confirm,key=lambda k:confirm[k]['mean'])
    summary['highest_confirmation_mean']=winner
    summary['evidence_of_improvement']=[k for k,v in confirm.items() if k!='baseline' and v['paired_interval'][0]>0]
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name in ['manifest.json','completion.json','trials.json','tuned-configs.json','confirmation-configs.json','validation-ranking.json']:
        shutil.copy2(root/name,out/name)
    with gzip.GzipFile(filename=str(out/'games.jsonl.gz'),mode='wb',mtime=0) as f:f.write((root/'games.jsonl').read_bytes())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})
    valid=summary['validation'];order=sorted((k for k in valid if k!='baseline'),key=lambda k:valid[k]['mean'])
    fig,ax=plt.subplots(figsize=(10,8))
    for i,k in enumerate(order):
        v=valid[k];lo,hi=v['paired_interval'];ax.plot([lo,hi],[i,i],color='#91a6b8',lw=2);ax.scatter(v['delta'],i,color='#137f70' if v['delta']>0 else '#b25048',s=35,zorder=3)
        ax.text(max(hi,0)+12,i,f"{v['mean']:.0f}",va='center',fontsize=9)
    ax.axvline(0,color='#34445a',lw=1);ax.set_yticks(range(16),order);ax.set_xlabel('Mean score difference versus baseline (paired bootstrap 95% interval)');ax.set_title('All 16 tuned ideas · 32 unseen comparison maps\nLabels show mean score; these intervals are exploratory',loc='left',pad=14);ax.grid(axis='x',alpha=.15);fig.tight_layout();fig.savefig(out/'validation.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5));colors=['#60758b','#138878','#d38a28']
    labels=sorted(confirm,key=lambda k:confirm[k]['mean'],reverse=True)
    allscores=[r['score'] for r in rows if r['stage']=='confirmation'];bins=np.linspace(0,max(allscores)*1.025,16)
    for color,k in zip(colors,labels):
        scores=[r['score'] for r in groups[('confirmation',k)].values()]
        axs[0].hist(scores,bins=bins,histtype='step',lw=2,label=f"{k}: mean {confirm[k]['mean']:.0f}",color=color)
    axs[0].set_xlabel('Final game score');axs[0].set_ylabel('Games');axs[0].legend(fontsize=8);axs[0].set_title('Score distribution · 64 fresh maps')
    finalists=[k for k in labels if k!='baseline']
    for i,k in enumerate(finalists):
        v=confirm[k];lo,hi=v['paired_interval'];axs[1].plot([lo,hi],[i,i],lw=3,color='#91a6b8');axs[1].scatter(v['delta'],i,color='#138878',s=50)
    axs[1].axvline(0,color='#34445a');axs[1].set_yticks(range(len(finalists)),finalists);axs[1].set_ylim(-.5,1.5);axs[1].set_xlabel('Paired score difference versus baseline');axs[1].set_title('Confirmation · 97.5% intervals\nBonferroni coverage for two comparisons');fig.tight_layout();fig.savefig(out/'confirmation.png',dpi=170);plt.close(fig)
    text=['# Avoidance experiment results','',f"Completed **{len(rows):,} native full games** in **{done['elapsed_seconds']/60:.1f} minutes** on 32 Runpod vCPUs.",'','The unchanged baseline and each candidate ran on identical seeds within each evaluation split. Training maps were never used for the following comparisons.','', '## Final confirmation','', '| Approach | Mean score | Difference | Paired 97.5% interval | Wins / 64 |','|---|---:|---:|---:|---:|']
    for k in labels:
        v=confirm[k];lo,hi=v['paired_interval'];text.append(f"| {k} | {v['mean']:.1f} | {v['delta']:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {v['wins']} |")
    text+=['','The two finalists were selected using the separate comparison maps. Their intervals use 97.5% paired bootstrap coverage each (Bonferroni adjustment for two comparisons). Intervals crossing zero do not establish an improvement.','', '![Confirmation distributions](confirmation.png)','', '## All sixteen approaches','', 'These are selection-stage results on 32 maps, not the final confirmation.','', '| Approach | Idea | Mean score | Difference |','|---|---|---:|---:|']
    for k in sorted(valid,key=lambda k:valid[k]['mean'],reverse=True):
        v=valid[k];text.append(f"| {k} | {v['idea']} | {v['mean']:.1f} | {v['delta']:+.1f} |")
    text+=['','![Comparison intervals](validation.png)','', '## Limits','', '- Each family received only 16 trials on eight training maps: one baseline-like initial setting, five random starts, and ten Gaussian-process expected-improvement proposals. More tuning could change the ranking.','- Each approach also tunes reaction and sprint distances. Results compare tuned policy packages; they do not isolate the causal benefit of the named mechanism.', '- Raw scores measure complete ordinary games. Approximate pursuit search does not use hidden predator state, and is not an exact physics rollout.','- Corner settings target ±10°, but these score tests do not independently verify that angular behavior.','- Comparison-stage intervals are exploratory and unadjusted across the 16 approaches. Confirmation is kept separate to reduce selection bias.','- Runtime is elapsed experiment time, excluding pod startup, build, transfer and teardown; billed cost is recorded separately.','', '## Reproduction','', 'Exact configurations: [tuned-configs.json](tuned-configs.json). Source and environment hashes: [manifest.json](manifest.json). Trial history: [trials.json](trials.json). Every game: [games.jsonl.gz](games.jsonl.gz).','']
    (out/'RESULTS.md').write_text('\n'.join(text))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
