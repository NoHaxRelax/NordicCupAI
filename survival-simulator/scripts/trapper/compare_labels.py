"""Compare run labels (e.g. society vs several trapper presets) over the same seeds.

    ../.venv/bin/python scripts/trapper/compare_labels.py results/trapper/remote ab-society ab-v6 ab-v5 ab-noreserve
"""
import glob, json, os, statistics, sys

folder, base, *others = sys.argv[1:]

def load(label):
    runs = {}
    for f in glob.glob(os.path.join(folder, f'*-{label}-r*-*.json')):
        r = json.load(open(f))
        runs.setdefault(r['seed'], []).append(r)
    return runs

B = load(base)
def seed_mean(runs, key):
    return {s: statistics.mean((r.get(key) or 0.0) for r in rs) for s, rs in runs.items()}
bm = seed_mean(B, 'score')
print(f'{base}: {len(B)} seeds, {sum(len(v) for v in B.values())} runs, mean score {statistics.mean(bm.values()):.1f}, '
      f'extinct runs {sum(r["alive"] == 0 for rs in B.values() for r in rs)}')
for lab in others:
    R = load(lab)
    if not R:
        print(lab, 'no runs'); continue
    rm = seed_mean(R, 'score'); hm = seed_mean(R, 'held_fraction')
    common = sorted(set(rm) & set(bm))
    diffs = [rm[s] - bm[s] for s in common]
    ext = sum(r['alive'] == 0 for rs in R.values() for r in rs)
    print(f'{lab}: {len(R)} seeds, mean score {statistics.mean(rm.values()):.1f}, diff vs {base}: mean {statistics.mean(diffs):+.1f} '
          f'median {statistics.median(diffs):+.1f} wins {sum(d > 0 for d in diffs)}/{len(diffs)}, held {statistics.mean(hm.values()):.3f}, extinct runs {ext}')
    print('   per seed diff:', {s: round(rm[s] - bm[s], 1) for s in common})
