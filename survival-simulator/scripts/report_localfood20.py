"""Compare tail-checkpoint tuning with the frozen full-game panel."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / 'docs/localfood20'

def main():
    final = {r['model']: r for r in json.loads((ROOT/'final/summary.json').read_text())}
    pairs = json.loads((ROOT/'final/paired.json').read_text())
    def delta(a, b):
        r = next(r for r in pairs if {r['a'], r['b']} == {a, b})
        return (r['mean_difference'], r['ci95']) if r['a'] == a else (-r['mean_difference'], [-r['ci95'][1], -r['ci95'][0]])
    checkpoints = json.loads((ROOT/'pod-0/checkpoints.json').read_text())
    baseline = sum(r['baseline_score']-r['score'] for r in checkpoints)/len(checkpoints)
    files = sorted(ROOT.glob('pod-*/*-trials.json'))
    assert len(files) == 20
    fig, axes = plt.subplots(5,4,figsize=(16,15),sharex=True,sharey=True)
    lines = ['# Local-food successor research', '',
        '20 families, 32 Bayesian-optimization trials each, 200 shared local_food checkpoints 250 seconds before extinction. Winners were frozen before 1000 fresh full games per model (seeds 15001–16000).', '',
        f'Unchanged local_food gains {baseline:.2f} points after these checkpoints. Training gain below is relative to that continuation, not a full-game score.', '',
        '| Family | First trial continuation | Best continuation | Best trial | Training gain vs local_food | Full-game mean | Paired gain vs local_food (95% CI) | Paired gain vs scheduled breeding (95% CI) |',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for ax, p in zip(axes.flat, files):
        name = p.name.removesuffix('-trials.json')
        trials = json.loads(p.read_text())
        best = max(trials,key=lambda r:r['score'])
        x = [r['iteration'] for r in trials]; y = [r['score'] for r in trials]
        ax.scatter(x,y,s=12,color='#8c9bb3')
        ax.step(x,[r['best'] for r in trials],where='post',color='#2166ac')
        ax.axhline(baseline,color='#aa5030',ls='--',lw=1)
        ax.set_title(name,fontsize=11);ax.grid(alpha=.2);ax.set_xlim(1,32)
        a, ac = delta(name,'local_food_baseline'); b, bc = delta(name,'scheduled_breeding_baseline')
        lines.append(f"| {name} | {y[0]:.1f} | {best['score']:.1f} | {best['iteration']} | {best['score']-baseline:+.1f} | {final[name]['mean']:.1f} | {a:+.1f} [{ac[0]:+.1f}, {ac[1]:+.1f}] | {b:+.1f} [{bc[0]:+.1f}, {bc[1]:+.1f}] |")
    fig.suptitle('Local-food BO: trial continuation (dots), best so far (blue), unchanged baseline (dashed)')
    fig.supxlabel('Trial');fig.supylabel('Mean score gained after checkpoint');fig.tight_layout(rect=(.02,.02,1,.97))
    fig.savefig(ROOT/'training-evolution.png',dpi=150);plt.close(fig)
    lines += ['', '![Training histories](training-evolution.png)', '',
        'Every family retunes four breeding trait weights, so differences are effects of the complete tuned configurations, not isolated causal effects of the named feature. Each first trial is family-specific. Tail checkpoints cannot reliably distinguish activation thresholds that precede the checkpoint; fresh full games test the resulting early-game consequences.', '',
        'Confidence intervals are pointwise paired bootstrap intervals, not adjusted for searching across 20 families. Selecting the highest test score introduces selection uncertainty.', '',
        '## Recorded deaths', '',
        'Mean counts per full game. Energy deaths include aging/starvation. Counts depend on population and game duration; they are not exposure-adjusted death rates.', '',
        '| Model | Predator deaths | Energy deaths |', '|---|---:|---:|']
    deaths = {name: [0,0,0] for name in final}
    for p in (ROOT/'final').glob('shard-*/games.jsonl'):
        for line in p.read_text().splitlines():
            r=json.loads(line);d=deaths[r['model']]
            d[0]+=r['predation_deaths'];d[1]+=r['energy_deaths'];d[2]+=1
    for name in final:
        a,b,n=deaths[name];assert n==1000
        final[name]['mean_predation_deaths']=a/n;final[name]['mean_energy_deaths']=b/n
        lines.append(f'| {name} | {a/n:.2f} | {b/n:.2f} |')
    (ROOT/'final/summary.json').write_text(json.dumps(list(final.values()),indent=2)+'\n')
    seconds=sum(json.loads(p.read_text())['elapsed_seconds'] for p in ROOT.glob('pod-*/complete.json'))
    lines += ['', '[All means, confidence intervals and compute times](final/RESULTS.md)', '',
        f'Training active compute: {seconds/3600:.2f} pod-hours, approximately ${seconds*.96/3600:.2f}. Excludes setup, storage and idle time. Evaluation cost is reported separately in the full results.']
    (ROOT/'TRAINING.md').write_text('\n'.join(lines)+'\n')

if __name__ == '__main__':
    main()
