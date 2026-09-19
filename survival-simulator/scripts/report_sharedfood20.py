"""Three-panel comparison: training, held-out equivalent checkpoints, full games."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]/'docs/sharedfood20'

def main():
    final={r['model']:r for r in json.loads((ROOT/'final/summary.json').read_text())}
    pairs=json.loads((ROOT/'final/paired.json').read_text())
    def delta(name):
        ref='expanded_food_baseline';r=next(r for r in pairs if {r['a'],r['b']}=={name,ref})
        return (r['mean_difference'],r['ci95'])if r['a']==name else(-r['mean_difference'],[-r['ci95'][1],-r['ci95'][0]])
    trainmeta=json.loads((ROOT/'pod-0/checkpoints.json').read_text());base_train=np.mean([r['baseline_score']-r['score'] for r in trainmeta])
    rows=[];metas=[];reference=None
    for i in range(10):
        d=ROOT/'validation'/f'shard-{i}';manifest=json.loads((d/'manifest.json').read_text());done=json.loads((d/'complete.json').read_text())
        if reference is None:reference=manifest
        assert manifest['configs']==reference['configs'] and manifest['sources']==reference['sources']
        assert done['games']==20*len(final)
        rows.extend(json.loads(l)for l in(d/'games.jsonl').read_text().splitlines());metas.extend(json.loads((d/'checkpoints.json').read_text()))
    assert sorted(r['seed']for r in metas)==list(range(18001,18201))
    gains={}
    for name in final:
        r=sorted((r for r in rows if r['model']==name),key=lambda r:r['seed'])
        assert [a['seed']for a in r]==list(range(18001,18201))
        gains[name]=np.array([a['gain']for a in r])
    idx=np.random.default_rng(18000).integers(0,200,size=(10000,200));baseline=gains['expanded_food_baseline']
    fig,axes=plt.subplots(5,4,figsize=(16,15),sharex=True,sharey=True)
    lines=['# Shared-food: training versus independent validation versus full games','',
        f'200 training checkpoints, 200 fresh validation checkpoints, 1000 fresh full games per model. Training baseline continuation {base_train:.2f}; held-out baseline continuation {baseline.mean():.2f}. All gains compare with expanded local food within their own panel.','',
        '| Family | Best trial | Train gain | Held-out checkpoint gain (95% CI) | Train − validation | Full-game gain (95% CI) | Full-game mean |',
        '|---|---:|---:|---|---:|---|---:|']
    summary=[]
    for ax,p in zip(axes.flat,sorted(ROOT.glob('pod-*/*-trials.json'))):
        name=p.name.removesuffix('-trials.json');trials=json.loads(p.read_text());best=max(trials,key=lambda r:r['score'])
        train=best['score']-base_train;vd=gains[name]-baseline;ci=np.quantile(vd[idx].mean(axis=1),[.025,.975]);test,tc=delta(name)
        lines.append(f"| {name} | {best['iteration']} | {train:+.1f} | {vd.mean():+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}] | {train-vd.mean():+.1f} | {test:+.1f} [{tc[0]:+.1f}, {tc[1]:+.1f}] | {final[name]['mean']:.1f} |")
        summary.append(dict(model=name,train_gain=float(train),validation_gain=float(vd.mean()),validation_ci95=ci.tolist(),train_minus_validation=float(train-vd.mean()),full_game_gain=test,full_game_ci95=tc))
        x=[t['iteration']for t in trials];ax.scatter(x,[t['score']-base_train for t in trials],s=12,color='#a0aabd');ax.step(x,[t['best']-base_train for t in trials],where='post',label='Training',color='#2470a0')
        ax.axhline(vd.mean(),color='#c7791d',label='Held-out tail');ax.axhline(test,color='#467c50',ls='--',label='Full game');ax.axhline(0,color='#888',lw=.6);ax.set_title(name,fontsize=10);ax.set_xlim(1,32);ax.grid(alpha=.15)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle('Best training gain over iterations, held-out continuation gain, full-game gain');fig.supxlabel('BO trial');fig.supylabel('Mean gain versus expanded local food');fig.tight_layout(rect=(.02,.02,1,.97));fig.savefig(ROOT/'training-evolution.png',dpi=150);plt.close(fig)
    lines+=['','![Evolution](training-evolution.png)','',
        'The train−validation gap measures optimism on the same continuation task (plus sampling uncertainty). The validation/full-game difference includes a change in initialization and policy-induced early-game states. The latter is not an overfitting estimate. All intervals are pointwise paired percentile bootstrap intervals, not corrected for comparing 20 candidates. All families retune breeding weights, so this is not a single-feature ablation.','',
        f"Short baseline games with checkpoints at time zero: training {sum(r['checkpoint_tick']==0 for r in trainmeta)}/200, validation {sum(r['checkpoint_tick']==0 for r in metas)}/200.",'',
        '[Full-game means, CIs and compute times](final/RESULTS.md)','',
        '## Untuned sharing control','']
    vd=gains['shared_food_control']-baseline;ci=np.quantile(vd[idx].mean(axis=1),[.025,.975]);test,tc=delta('shared_food_control')
    lines.append(f'Sharing only: held-out continuation {vd.mean():+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}], full game {test:+.1f} [{tc[0]:+.1f}, {tc[1]:+.1f}].')
    lines+=['','## Full-game death counts','','Counts per game, not exposure-adjusted rates. Energy includes aging/starvation.','','| Model | Predator deaths | Energy deaths |','|---|---:|---:|']
    deaths={n:[0,0,0]for n in final}
    for p in(ROOT/'final').glob('shard-*/games.jsonl'):
        for line in p.read_text().splitlines():
            r=json.loads(line);d=deaths[r['model']];d[0]+=r['predation_deaths'];d[1]+=r['energy_deaths'];d[2]+=1
    for n in final:
        a,b,c=deaths[n];assert c==1000;lines.append(f'| {n} | {a/c:.2f} | {b/c:.2f} |')
    for phase,pattern in [('training','pod-*/complete.json'),('validation','validation/shard-*/complete.json'),('full-game','final/shard-*/complete.json')]:
        secs=sum(json.loads(p.read_text())['elapsed_seconds']for p in ROOT.glob(pattern));lines+=['',f'{phase} active compute attribution: ${secs*.96/3600:.2f}, excludes idle/storage/setup.']
    (ROOT/'COMPARISON.md').write_text('\n'.join(lines)+'\n');(ROOT/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
