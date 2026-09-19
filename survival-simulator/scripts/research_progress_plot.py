"""Improvement-over-time plots for a tune study; reads results, starts no games.

Renders three panels from `study.json`. They are separate panels rather than one
chart with two y-axes on purpose: the objective is dimensionless, survival is in
seconds and score is in points, and overlaying different scales on twin axes
invites reading a crossing point that means nothing.

  1. Primary objective, best-so-far against completed trials.
  2. Mean survival seconds, best-so-far against completed trials.
  3. Primary objective, best-so-far against cumulative game wall-hours, which is
     what the Pod budget actually buys.

Each panel shows individual completed trials behind the best-so-far step so a
flat stretch is visibly "many trials, none better" rather than "nothing ran".
Colours are the first slots of the validated default categorical palette, used
in fixed order. Static PNG: no hover layer, so the per-stage tables in report.md
remain the readable-values view.

usage: python scripts/research_progress_plot.py --out logs/policy-search
"""
import argparse
import json
from pathlib import Path

INK = '#37352f'
MUTED = '#787066'
GRID = '#e8e6e1'
TRIALS = '#c3c2b7'
SERIES = ('#2a78d6', '#eb6834', '#1baf7a')


def series(trials, horizon):
    """Completed trials in order, carrying the incumbent best at each point.

    The incumbent is chosen by the full lexicographic rank the optimizer itself
    uses, then its survival and score are reported. Taking an independent
    maximum per metric would draw a champion that no single candidate achieved.
    """
    rows, incumbent, wall = [], None, 0.
    for index, trial in enumerate(trials):
        summary = trial.get('summary') or {}
        if summary.get('rank') is None:
            continue
        wall += summary.get('wall_seconds', 0.)/3600.
        if incumbent is None or tuple(summary['rank']) > tuple(incumbent['rank']):
            incumbent = summary
        rows.append(dict(index=index, label=trial.get('label', f'trial {index}'),
                         rank=summary['rank'][0], survival=summary['mean_survival'],
                         score=summary['mean_score'], wall_hours=wall,
                         best_rank=incumbent['rank'][0], best_survival=incumbent['mean_survival'],
                         best_score=incumbent['mean_score']))
    return rows


def panel(axes, x, points, best, color, title, xlabel, ylabel, point_label, best_label):
    axes.scatter(x, points, s=18, color=TRIALS, zorder=2, label=point_label)
    axes.step(x, best, where='post', color=color, linewidth=2, zorder=3, label=best_label)
    axes.set_title(title, color=INK, fontsize=11, loc='left', pad=8)
    axes.set_xlabel(xlabel, color=MUTED, fontsize=9)
    axes.set_ylabel(ylabel, color=MUTED, fontsize=9)
    axes.grid(True, color=GRID, linewidth=.8, zorder=0)
    axes.set_axisbelow(True)
    for side in ('top', 'right'):
        axes.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=MUTED, labelsize=8, length=0)
    if best:
        # One direct label on the current best; never a number on every point.
        axes.annotate(f'{best[-1]:.3g}', (x[-1], best[-1]), textcoords='offset points',
                      xytext=(6, 0), va='center', color=INK, fontsize=9, fontweight='bold')
    axes.legend(loc='lower right', frameon=False, fontsize=8, labelcolor=MUTED)


def render(study, horizon, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = series(study.get('trials', []), horizon)
    if not rows:
        return None
    figure, grid = plt.subplots(2, 2, figsize=(12.5, 8), dpi=110)
    figure.patch.set_facecolor('white')
    panels = grid.ravel()
    for axes in panels:
        axes.set_facecolor('white')
    completed = list(range(1, len(rows)+1))
    panel(panels[0], completed, [r['rank'] for r in rows], [r['best_rank'] for r in rows],
          SERIES[0], 'Primary objective', 'completed trials', 'mean/horizon + .25*worst/horizon',
          'trial', 'incumbent')
    panel(panels[1], completed, [r['survival'] for r in rows], [r['best_survival'] for r in rows],
          SERIES[1], 'Mean survival', 'completed trials', 'seconds', 'trial', 'incumbent')
    # Once candidates reach the horizon, survival saturates and score is the only
    # thing still separating them, so it gets its own panel rather than a footnote.
    panel(panels[2], completed, [r['score'] for r in rows], [r['best_score'] for r in rows],
          SERIES[2], 'Mean score (separates horizon-reaching candidates)', 'completed trials',
          'points', 'trial', 'incumbent')
    panel(panels[3], [r['wall_hours'] for r in rows], [r['rank'] for r in rows],
          [r['best_rank'] for r in rows], SERIES[0], 'Objective per compute spent',
          'cumulative game wall-hours', 'mean/horizon + .25*worst/horizon', 'trial', 'incumbent')
    survived = sum(r['survival'] >= horizon-1e-6 for r in rows)
    figure.suptitle(f'Search progress: {len(rows)} completed trials, {rows[-1]["wall_hours"]:.1f} game '
                    f'wall-hours, {survived} at full {horizon:.0f}s horizon',
                    color=INK, fontsize=12, x=.006, ha='left')
    figure.tight_layout(rect=(0, 0, 1, .95))
    figure.savefig(destination, facecolor='white')
    plt.close(figure)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path, help='Study directory holding study.json')
    parser.add_argument('--name', default='progress.png')
    args = parser.parse_args()
    study = json.loads((args.out/'study.json').read_text(encoding='utf-8'))
    protocol = json.loads((args.out/'protocol.json').read_text(encoding='utf-8'))
    written = render(study, protocol['seconds'], args.out/args.name)
    print(f'Wrote {written}' if written else 'No completed trials yet; no plot written.')


if __name__ == '__main__':
    main()
