"""Render the frozen BO histories and compare with full-game evaluation."""
import json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1]/'docs/late250'
def main():
    final={r['model']:r for r in json.loads((ROOT/'final/summary.json').read_text())}
    checkpoints=json.loads((ROOT/'pod-0/checkpoints.json').read_text())
    baseline=sum(r['baseline_score']-r['score'] for r in checkpoints)/len(checkpoints)
    files=sorted(ROOT.glob('pod-*/*-trials.json'))
    fig,axes=plt.subplots(5,4,figsize=(16,15),sharex=True,sharey=True)
    lines=['# Late-game tuning evolution','',
      f'32 trials per family on the same 200 checkpoints. Unchanged scheduled breeding gains {baseline:.1f} score after the checkpoint on average. Scores below are additional score, not full-game scores. Trial 1 is a family-specific seeded configuration, which can differ from the baseline.','',
      '| Family | Trial 1 gain | Best gain | Best trial | Full-game mean (1000 maps) |','|---|---:|---:|---:|---:|']
    for ax,p in zip(axes.flat,files):
        name=p.name.removesuffix('-trials.json');trials=json.loads(p.read_text());winner=max(trials,key=lambda r:r['score'])
        x=[r['iteration'] for r in trials];y=[r['score'] for r in trials]
        ax.scatter(x,y,s=12,color='#8c9bb3');ax.step(x,[r['best'] for r in trials],where='post',color='#2166ac')
        ax.axhline(baseline,color='#aa5030',ls='--',lw=1)
        ax.set_title(name,fontsize=11);ax.grid(alpha=.2);ax.set_xlim(1,32)
        lines.append(f"| {name} | {y[0]:.1f} | {winner['score']:.1f} | {winner['iteration']} | {final[name]['mean']:.1f} |")
    fig.suptitle('Late-game BO: trial gain (dots), best so far (blue), unchanged baseline (dashed)',fontsize=14)
    fig.supxlabel('Trial');fig.supylabel('Mean additional score after checkpoint');fig.tight_layout(rect=(.02,.02,1,.97))
    fig.savefig(ROOT/'training-evolution.png',dpi=150);plt.close(fig)
    lines+=['','![All 20 training histories](training-evolution.png)','',
      'The tail-state objective does not guarantee full-game improvement. Retirement nearly matches local_food on training gain (435.1 versus 435.4), but its full-game mean is much lower (1521.7 versus 1598.2). All candidates were frozen before the fresh evaluation. Selecting a winner from these test results introduces selection uncertainty; the reported 95% intervals are pointwise.','',
      '[Full-game scores, confidence intervals, paired gains and timings](final/RESULTS.md)']
    seconds=sum(json.loads(p.read_text())['elapsed_seconds'] for p in ROOT.glob('pod-*/complete.json'))
    lines+=['',f'Training active pod time: {seconds/3600:.2f} pod-hours, approximately ${seconds*.96/3600:.2f} at $0.96/pod-hour. Includes checkpoint construction and verification; excludes setup, storage and idle time.']
    (ROOT/'TRAINING.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
