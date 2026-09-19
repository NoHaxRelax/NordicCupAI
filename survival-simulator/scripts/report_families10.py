"""Render all ten tuning trajectories and the untouched test-set comparison."""
import argparse,json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tune_families10 import FAMILIES,TEST

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=pathlib.Path);ap.add_argument('--campaign',type=int,choices=[1,2],default=1);a=ap.parse_args();root=a.directory
    if a.campaign==2:
        import tune_families20 as campaign
        families,test=campaign.run.FAMILIES,campaign.run.TEST
        rounds,train_n=60,64
    else:families,test,rounds,train_n=FAMILIES,TEST,30,32
    allrows=[];summaries=[]
    fig,axes=plt.subplots(5,2,figsize=(12,14),constrained_layout=True)
    for ax,(fid,label,*_) in zip(axes.flat,families):
        d=root/fid
        if not (d/'complete.json').exists():raise RuntimeError(f'{fid} is incomplete; refusing final report')
        trials=json.loads((d/'trials.json').read_text())
        assert len(trials)==rounds
        rows=[json.loads(x)for x in (d/'games.jsonl').read_text().splitlines()]
        tests=sorted((r for r in rows if r['stage']=='test'),key=lambda r:r['seed'])
        assert [r['seed']for r in tests]==test
        allrows.extend(dict(family=fid,**r)for r in rows)
        summary=json.loads((d/'summary.json').read_text());summaries.append(summary)
        ax.plot(range(1,rounds+1),[r['score']for r in trials],'.',alpha=.35,label='Trial mean')
        ax.step(range(1,rounds+1),np.maximum.accumulate([r['score']for r in trials]),where='post',label='Best so far')
        ax.set(title=label,xlabel='Iteration',ylabel='Mean training score');ax.grid(alpha=.2)
    axes[0,0].legend();fig.savefig(root/'training-progress.png',dpi=150);plt.close(fig)
    summaries.sort(key=lambda r:r['mean'],reverse=True)
    fig,ax=plt.subplots(figsize=(10,6),constrained_layout=True)
    for i,r in enumerate(summaries):
        lo,hi=r['ci95'];ax.errorbar(r['mean'],i,xerr=[[r['mean']-lo],[hi-r['mean']]],fmt='o',capsize=3)
    ax.set(yticks=range(10),yticklabels=[r['family']for r in summaries],xlabel=f'Mean score on {len(test)} unseen maps (95% bootstrap interval)')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2);fig.savefig(root/'test-scores.png',dpi=150);plt.close(fig)
    lines=['# Ten-family Bayesian experiment','',
           f'{rounds} trials per family on the same {train_n} training maps; best training configuration evaluated on the same {len(test)} unseen maps. Objective: mean full-game score. Normal energy and predators, 3000-second horizon. C++ engine and policy. Policy random seed fixed at 0 independently of world seed.','',
           '| Family | Initial training | Best training | Best iteration | Test mean | 95% CI |',
           '|---|---:|---:|---:|---:|---|']
    for r in summaries:
        lines.append(f"| {r['family']} | {r['training_start']:.1f} | {r['training_best']:.1f} | {r['best_iteration']} | {r['mean']:.1f} | {r['ci95'][0]:.1f}–{r['ci95'][1]:.1f} |")
    baseline={r['seed']:r['score']for r in allrows if r['stage']=='baseline'}
    assert sorted(baseline)==test
    control_label='Previous population winner'if a.campaign==2 else'Untuned control'
    lines+=['',f"{control_label} mean: {np.mean(list(baseline.values())):.1f}."]
    original={r['seed']:r['score']for r in allrows if r['stage']=='original_baseline'}
    if a.campaign==2:
        assert sorted(original)==test
        lines+=['',f"Original baseline mean: {np.mean(list(original.values())):.1f}.",'',
                '| Family | Paired difference vs previous winner | 95% paired CI |','|---|---:|---|']
        for summary in summaries:
            score={r['seed']:r['score']for r in allrows if r['stage']=='test'and r['family']==summary['family']}
            delta=np.array([score[s]-baseline[s]for s in test])
            ci=np.quantile(np.random.default_rng(9012).choice(delta,(20000,len(test))).mean(axis=1),[.025,.975])
            summary['paired_gain']=float(delta.mean());summary['paired_ci95']=ci.tolist()
            lines.append(f"| {summary['family']} | {delta.mean():+.1f} | {ci[0]:+.1f}–{ci[1]:+.1f} |")
    lines+=['',
            f'Test ranking is descriptive: selecting the top of ten introduces selection bias. Intervals are not corrected for multiple comparisons. Training best-so-far is optimistic because it selects among {rounds} trials.','',
            '![Training progress](training-progress.png)','![Test scores](test-scores.png)']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (root/'ranking.json').write_text(json.dumps(summaries,indent=2)+'\n')
if __name__=='__main__':main()
