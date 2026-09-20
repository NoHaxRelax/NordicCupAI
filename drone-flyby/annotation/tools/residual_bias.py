#!/usr/bin/env python3
"""Two checks on the scene-speed correction, from a replay's own observation histories.

1. Frozen and near-frozen frame pairs. The validation flight holds still between frames 2 and 3, and
   nearly so between 8/9 and 238/239. A refresh spanning such a pair measures a displacement of almost
   nothing over a tick gap of almost nothing, which is an ill-conditioned fit: the factor it reports is
   noise, and it is pooled with everything else. This prints the lessons by tick gap so the size of that
   contamination is visible, and what the pooled factor becomes once short gaps are dropped.

2. A vertical bias that grows down the frame. A point standing above the calibrated plane moves further
   down the image than the plane does, so a forecast that transports it along the plane leaves its box
   ABOVE the object, by more the further down it is. The scene-speed factor absorbs part of that, being
   the same kind of scalar, but it is one number for the whole scene and cannot depend on image row.
   This measures the signed vertical error of a forecast against where the object was actually seen,
   split by image row, before and after the correction, and per class.

    python3 residual_bias.py replays/hx-capture --endpoint /workspace/verifier/cp02/endpoint
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from forecast_study import best_factor, load, pooled  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('replay')
    ap.add_argument('--endpoint', required=True)
    ap.add_argument('--min-tick', type=float, default=0.5, help='shortest tick gap a lesson may span')
    ap.add_argument('--window', type=int, default=40)
    a = ap.parse_args()
    sys.path.insert(0, a.endpoint)
    from tracking import MotionModel

    tracks, model_data = load(a.replay)
    model = MotionModel(np.array(model_data['matrix']), (3840, 2160), model_data['origin_tick'],
                        model_data['origin_tick']+1)

    lessons = []                      # (tick known, class, factor, tick gap)
    for track, (label, obs) in tracks.items():
        for (ta, ba), (tb, bb) in zip(obs, obs[1:]):
            k = best_factor(model, [(ta, ba, tb, bb)])
            if k is not None:
                lessons.append((tb, label, k, tb-ta))
    lessons.sort()

    print('=== 1. lessons by the tick gap they span')
    gaps = np.array([g for _, _, _, g in lessons])
    factors = np.array([k for _, _, k, _ in lessons])
    for lo, hi in ((0, .25), (.25, .5), (.5, 1.), (1., 2.), (2., 4.), (4., 100.)):
        sel = (gaps >= lo) & (gaps < hi)
        if sel.sum():
            print(f'  gap {lo:5.2f}-{hi:<6.2f} n={sel.sum():4d}  median factor {np.median(factors[sel]):.4f}  '
                  f'p10 {np.percentile(factors[sel], 10):.4f}  p90 {np.percentile(factors[sel], 90):.4f}')
    keep = gaps >= a.min_tick
    print(f'  pooled over all lessons      {pooled(list(factors)):.4f}')
    print(f'  pooled over gaps >= {a.min_tick}      {pooled(list(factors[keep])):.4f}   '
          f'({(~keep).sum()} of {len(gaps)} lessons dropped)')
    print(f'  last {a.window}, all lessons       {pooled(list(factors[-a.window:])):.4f}')
    print(f'  last {a.window}, gaps >= {a.min_tick}       {pooled(list(factors[keep][-a.window:])):.4f}')

    print('\n=== 2. signed vertical error of a forecast, by where the object sits in the frame')
    print('    (predicted centre y minus observed centre y: negative = our box sits ABOVE the object)')
    speed = pooled(list(factors[keep][-a.window:]))
    rows = defaultdict(lambda: defaultdict(list))
    per_class = defaultdict(lambda: defaultdict(list))
    for track, (label, obs) in tracks.items():
        for i, (anchor_t, anchor_b) in enumerate(obs):
            for tick, truth in obs[i+1:]:
                if tick-anchor_t < 4:
                    continue
                row = (truth[1]+truth[3])/2
                for name, factor in (('ground', 1.), ('corrected', speed)):
                    try:
                        predicted = model.scaled(factor).box(anchor_b, anchor_t, tick)
                    except Exception:
                        continue
                    dy = (predicted[1]+predicted[3])/2-row
                    rows[name][int(row//360)].append(dy)
                    per_class[name][label].append((row, dy))
    print(f'    rolling-window factor in use: {speed:.4f}')
    print(f'    {"image row":14s} ' + '  '.join(f'{n:>22s}' for n in ('ground', 'corrected')))
    for band in sorted(rows['ground']):
        cells = []
        for name in ('ground', 'corrected'):
            v = rows[name].get(band, [])
            cells.append(f'{np.median(v):+7.1f} px  (n={len(v):4d})'.rjust(22) if v else f'{"-":>22s}')
        print(f'    y {band*360:4d}-{band*360+359:4d}   ' + '  '.join(cells))

    print('\n    per class, forecasts landing in the bottom third of the frame (y >= 1440):')
    print(f'    {"class":16s} {"ground":>12s} {"corrected":>12s}   n')
    for label in sorted(per_class['ground']):
        cells = []
        for name in ('ground', 'corrected'):
            v = [dy for row, dy in per_class[name][label] if row >= 1440]
            cells.append(f'{np.median(v):+11.1f}' if v else f'{"-":>12s}')
        n = len([1 for row, _ in per_class['ground'][label] if row >= 1440])
        print(f'    {label:16s} ' + ' '.join(cells) + f'   {n:4d}')

    height_study(model, tracks, speed)


def height_study(model, tracks, speed, strength=4.):
    """Does a per-class height factor, on top of the scene speed, explain what is left?

    The scene factor is one number for the whole camera; an object's own height above the calibrated
    plane is a property of the object, and plausibly of its class. This pools a height ratio per class
    from refreshes that had already happened and scores it on later, unseen sightings, against the
    speed-corrected forecast it has to beat.
    """
    from collections import defaultdict
    import numpy as np
    corrected = model.scaled(speed)
    lessons = []
    for track, (label, obs) in tracks.items():
        for (ta, ba), (tb, bb) in zip(obs, obs[1:]):
            h = best_factor(corrected, [(ta, ba, tb, bb)], upright=True)
            if h is not None:
                lessons.append((tb, label, h))
    lessons.sort()
    print('\n=== 3. per-class height ratio on top of the scene speed')
    for c in sorted({l for _, l, _ in lessons}):
        v = [h for _, lab, h in lessons if lab == c]
        print(f'    {c:16s} median {np.median(v):.3f}  n={len(v):3d}')
    rows = defaultdict(list)
    for track, (label, obs) in tracks.items():
        for i, (anchor_t, anchor_b) in enumerate(obs):
            seen = [h for t, lab, h in lessons if t <= anchor_t and lab == label]
            height = pooled(seen, prior=1., strength=strength)
            for tick, truth in obs[i+1:]:
                gap = tick-anchor_t
                if gap < 1:
                    continue
                for name, box in (('speed only', corrected.box(anchor_b, anchor_t, tick)),
                                  ('speed+height', corrected.box_upright(anchor_b, anchor_t, tick, height))):
                    err = float(np.hypot((box[0]+box[2])/2-(truth[0]+truth[2])/2,
                                         (box[1]+box[3])/2-(truth[1]+truth[3])/2))
                    rows[name].append((gap, err))
    print(f'    {"model":14s} ' + '  '.join(f'{f"gap {lo}-{hi}":>14s}' for lo, hi in ((1, 4), (5, 8), (9, 16), (17, 40))))
    for name in ('speed only', 'speed+height'):
        cells = []
        for lo, hi in ((1, 4), (5, 8), (9, 16), (17, 40)):
            sel = [e for g, e in rows[name] if lo <= g <= hi]
            cells.append(f'{np.median(sel):11.2f} px' if sel else f'{"-":>14s}')
        print(f'    {name:14s} ' + '  '.join(cells))


if __name__ == '__main__':
    main()
