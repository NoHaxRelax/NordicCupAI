"""Render all ten tuning trajectories and the untouched test-set comparison."""
import argparse,json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tune_families10 import FAMILIES,TEST

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=pathlib.Path);a=ap.parse_args();root=a.directory
    allrows=[];summaries=[]
    fig,axes=plt.subplots(5,2,figsize=(12,14),constrained_layout=True)
    for ax,(fid,label,*_) in zip(axes.flat,FAMILIES):
        d=root/fid
        if not (d/'complete.json').exists():raise RuntimeError(f'{fid} is incomplete; refusing final report')
        trials=json.loads((d/'trials.json').read_text())
        assert len(trials)==30
        rows=[json.loads(x)for x in (d/'games.jsonl').read_text().splitlines()]
        tests=sorted((r for r in rows if r['stage']=='test'),key=lambda r:r['seed'])
        assert [r['seed']for r in tests]==TEST
        allrows.extend(dict(family=fid,**r)for r in rows)
        summary=json.loads((d/'summary.json').read_text());summaries.append(summary)
        ax.plot(range(1,31),[r['score']for r in trials],'.',alpha=.35,label='Trial mean')
        ax.step(range(1,31),np.maximum.accumulate([r['score']for r in trials]),where='post',label='Best so far')
        ax.set(title=label,xlabel='Iteration',ylabel='Mean training score');ax.grid(alpha=.2)
    axes[0,0].legend();fig.savefig(root/'training-progress.png',dpi=150);plt.close(fig)
    summaries.sort(key=lambda r:r['mean'],reverse=True)
    fig,ax=plt.subplots(figsize=(10,6),constrained_layout=True)
    for i,r in enumerate(summaries):
        lo,hi=r['ci95'];ax.errorbar(r['mean'],i,xerr=[[r['mean']-lo],[hi-r['mean']]],fmt='o',capsize=3)
    ax.set(yticks=range(10),yticklabels=[r['family']for r in summaries],xlabel='Mean score on 100 unseen maps (95% bootstrap interval)')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2);fig.savefig(root/'test-scores.png',dpi=150);plt.close(fig)
    lines=['# Ten-family Bayesian experiment','',
           '30 trials per family on the same 32 training maps; best training configuration evaluated on the same 100 unseen maps. Objective: mean full-game score. Normal energy and predators, 3000-second horizon. C++ engine and policy. Policy random seed fixed at 0 independently of world seed.','',
           '| Family | Initial training | Best training | Best iteration | Test mean | 95% CI |',
           '|---|---:|---:|---:|---:|---|']
    for r in summaries:
        lines.append(f"| {r['family']} | {r['training_start']:.1f} | {r['training_best']:.1f} | {r['best_iteration']} | {r['mean']:.1f} | {r['ci95'][0]:.1f}–{r['ci95'][1]:.1f} |")
    baseline={r['seed']:r['score']for r in allrows if r['stage']=='baseline'}
    assert sorted(baseline)==TEST
    lines+=['',f"Untuned control mean: {np.mean(list(baseline.values())):.1f}.",'',
            'Test ranking is descriptive: selecting the top of ten introduces selection bias. These intervals estimate individual means and do not establish superiority between families. Training best-so-far is optimistic because it selects among 30 trials.','',
            '![Training progress](training-progress.png)','![Test scores](test-scores.png)']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (root/'ranking.json').write_text(json.dumps(summaries,indent=2)+'\n')
if __name__=='__main__':main()
