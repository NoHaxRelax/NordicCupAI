#!/usr/bin/env python3
"""COCO-AP lattice solver for single-class validation scores (13 classes, IoU .5, 101-point interpolation).

Given the number of boxes answered (constant confidence, so ranking = frame order) and the score, list the
(truth frames T, hits, failing ranks) combinations that reproduce the score exactly.

    python3 lattice.py SCORE N_BOXES [--max-fails 3] [--tmin 1] [--tmax 300]
"""
import argparse, itertools
import numpy as np
recThrs = np.linspace(0, 1, 101)

def coco_ap(flags, T):
    flags = np.array(flags); tp = np.cumsum(flags); fp = np.cumsum(1 - flags); rc = tp / T; pr = (tp / np.maximum(tp + fp, 1e-9)).tolist()
    for i in range(len(pr) - 1, 0, -1): pr[i - 1] = max(pr[i - 1], pr[i])
    inds = np.searchsorted(rc, recThrs, side='left'); return float(np.mean([pr[i] if i < len(pr) else 0 for i in inds]))

def solve(score, n, max_fails=3, tmin=1, tmax=300, tol=5e-6, classes=13):
    target = score * classes; out = []
    for T in range(max(tmin, 1), tmax + 1):
        for nf in range(0, max_fails + 1):
            for fails in itertools.combinations(range(n), nf):
                flags = [0 if i in fails else 1 for i in range(n)]
                if abs(coco_ap(flags, T) - target) < tol: out.append((T, n - nf, fails))
    return target, out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('score', type=float); ap.add_argument('n', type=int)
    ap.add_argument('--max-fails', type=int, default=3); ap.add_argument('--tmin', type=int, default=1); ap.add_argument('--tmax', type=int, default=300)
    a = ap.parse_args(); target, sols = solve(a.score, a.n, a.max_fails, a.tmin, a.tmax)
    print(f'AP = {target:.5f}; {len(sols)} solutions (T, hits, failing ranks 0-based):')
    byT = {}
    for T, h, f in sols: byT.setdefault((T, h), []).append(f)
    for (T, h), fs in sorted(byT.items()): print(f'  T={T:4d} hits={h:4d}  fail-rank sets: {fs[:4]}{" ..." if len(fs) > 4 else ""}')


def feasible(score, n, tmin=1, tmax=300, classes=13, tol=5e-6):
    """Fast reading for large n: for each (T, hits) the AP lies between 'all failures ranked last' (max) and
    'all failures ranked first' (min); report the (T, hits) whose interval contains the target."""
    target = score * classes; out = []
    for T in range(max(tmin, 1), tmax + 1):
        for h in range(0, n + 1):
            last = coco_ap([1] * h + [0] * (n - h), T); first = coco_ap([0] * (n - h) + [1] * h, T)
            if first - tol <= target <= last + tol: out.append((T, h, round(first, 4), round(last, 4)))
    return target, out


if __name__ == '__main__' and False:
    pass
