#!/usr/bin/env python3
"""Which forecast model actually predicts where an object will be, measured on real track histories.

The scene is rigid and the camera motion is calibrated once, so an object's image motion is determined
by how near the camera it is compared with the plane that calibration used. That is a single scalar:
1.0 for a point on the calibrated plane, larger for something nearer. The same scalar also absorbs a
systematic error in the calibrated speed, which is why it is worth estimating even for flat objects.

Every model here is scored the way a live run would have to work: the factor is estimated only from
observations that had already been made when the forecast was issued, and it is scored against an
observation nobody had seen yet. Nothing is fitted on the future.

Models
  ground        transport every corner with the calibrated homography (what the tracker does today)
  global        one factor pooled over every refresh of every track so far, this flight
  class         the same, pooled within the object's own class, falling back to global while thin
  shrunk        the class estimate pulled towards the global one by its own sample count
  object        a factor fitted from this track's earlier observations alone
  upright       footprint edge on the calibrated plane, roof edge on the parallel plane at the factor

    python3 forecast_study.py replays/hx-capture --endpoint /workspace/verifier/cp02/endpoint
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

GRID = np.round(np.arange(0.90, 1.301, 0.002), 4)


def load(folder):
    """-> {track_id: (label, [(tick, box), ...])} and the flight's motion model."""
    histories, labels, model = {}, {}, None
    for f in sorted(Path(folder).glob('*.jsonl')):
        for line in open(f):
            r = json.loads(line)
            model = r.get('motion_model') or model
            for t in r.get('track_details') or []:
                if not t.get('history'):
                    continue
                labels[t['track_id']] = t['object_id']
                seen = histories.setdefault(t['track_id'], {})
                for tick, box in t['history']:
                    seen[float(tick)] = [float(v) for v in box]
    return {k: (labels[k], sorted(v.items())) for k, v in histories.items()}, model


def iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1]); x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.
    inter = (x1-x0)*(y1-y0)
    return float(inter/((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter))


def centre_error(a, b):
    return float(np.hypot((a[0]+a[2])/2-(b[0]+b[2])/2, (a[1]+a[3])/2-(b[1]+b[3])/2))


def best_factor(model, pairs, upright=False):
    """The factor minimising squared edge error over (tick_a, box_a, tick_b, box_b) pairs."""
    best, best_cost = None, np.inf
    for k in GRID:
        cost, used = 0., 0
        for ta, ba, tb, bb in pairs:
            if tb <= ta:
                continue
            try:
                predicted = (model.box_upright(ba, ta, tb, k) if upright
                             else model.scaled(k).box(ba, ta, tb))
            except Exception:
                cost = np.inf; break
            cost += float(np.sum((np.asarray(predicted)-np.asarray(bb))**2)); used += 1
        if used and cost < best_cost:
            best, best_cost = float(k), cost
    return best


def pooled(samples, prior=1., strength=4.):
    """A robust pooled factor: the median, pulled towards `prior` while the sample is thin."""
    if not samples:
        return prior
    estimate = float(np.median(samples))
    weight = len(samples)/(len(samples)+strength)
    return prior + weight*(estimate-prior)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('replay')
    ap.add_argument('--endpoint', required=True)
    ap.add_argument('--min-gap', type=float, default=1.)
    ap.add_argument('--strength', type=float, default=4., help='samples before a class estimate is trusted')
    a = ap.parse_args()
    sys.path.insert(0, a.endpoint)
    from tracking import MotionModel

    tracks, model_data = load(a.replay)
    if model_data is None:
        raise SystemExit('this replay has no motion_model in its log; rerun with the instrumented endpoint')
    model = MotionModel(np.array(model_data['matrix']), (3840, 2160), model_data['origin_tick'],
                        model_data['origin_tick']+1)

    # One factor per consecutive sighting: what a live run learns at the moment of each refresh.
    lessons = []          # (tick it became known, class, factor)
    for track, (label, obs) in tracks.items():
        for (ta, ba), (tb, bb) in zip(obs, obs[1:]):
            k = best_factor(model, [(ta, ba, tb, bb)])
            if k is not None:
                lessons.append((tb, label, k))
    lessons.sort()
    print(f'{len(lessons)} refresh lessons over {len(tracks)} tracks; '
          f'all-flight median factor {np.median([k for _, _, k in lessons]):.4f}')
    print('per class (median, count):  ' + '  '.join(
        f'{c} {np.median([k for _, lab, k in lessons if lab == c]):.4f} n={sum(1 for _, lab, _ in lessons if lab == c)}'
        for c in sorted({lab for _, lab, _ in lessons})))

    def known_before(tick, label=None, window=0):
        seen = [k for t, lab, k in lessons if t <= tick and (label is None or lab == label)]
        return seen[-window:] if window else seen

    # Does the drone's own speed drift over the flight? If it does, a factor pooled over everything
    # lags behind it and a rolling window is the right estimator.
    span = [t for t, _, _ in lessons]
    thirds = np.array_split(np.array([k for _, _, k in lessons]), 3)
    print('factor by third of the flight (tick %.0f-%.0f): %s' % (
        min(span), max(span), '  '.join(f'{np.median(t):.4f} (n={len(t)})' for t in thirds)))

    rows = defaultdict(list)
    for track, (label, obs) in tracks.items():
        for i, (anchor_t, anchor_b) in enumerate(obs):
            own_pairs = [(ta, ba, tb, bb) for (ta, ba), (tb, bb) in zip(obs[:i+1], obs[1:i+1])]
            own = best_factor(model, own_pairs) if len(own_pairs) >= 2 else None
            own_upright = best_factor(model, own_pairs, upright=True) if len(own_pairs) >= 2 else None
            g = pooled(known_before(anchor_t), strength=a.strength)
            w40 = pooled(known_before(anchor_t, window=40), strength=a.strength)
            w20 = pooled(known_before(anchor_t, window=20), strength=a.strength)
            w10 = pooled(known_before(anchor_t, window=10), strength=a.strength)
            c = pooled(known_before(anchor_t, label), prior=g, strength=a.strength)
            for tick, truth in obs[i+1:]:
                if tick-anchor_t < a.min_gap:
                    continue
                gap = tick-anchor_t
                for name, k in (('ground', 1.), ('global', g), ('class', c), ('object', own),
                                ('shrunk', None if own is None else pooled([own], prior=c, strength=1.)),
                                ('window40', w40), ('window20', w20), ('window10', w10)):
                    if k is None:
                        continue
                    try:
                        predicted = model.scaled(k).box(anchor_b, anchor_t, tick)
                    except Exception:
                        continue
                    rows[name].append((gap, centre_error(predicted, truth), iou(predicted, truth), label))
                if own_upright is not None:
                    try:
                        predicted = model.box_upright(anchor_b, anchor_t, tick, own_upright)
                        rows['upright'].append((gap, centre_error(predicted, truth), iou(predicted, truth), label))
                    except Exception:
                        pass

    names = ('ground', 'global', 'window40', 'window20', 'window10', 'class', 'shrunk', 'object', 'upright')
    print(f'\ncausal forecasts, {len(rows["ground"])} anchor/target pairs, nothing fitted on the future')
    buckets = [(1, 4), (5, 8), (9, 16), (17, 40)]
    print('model      ' + '  '.join(f'{f"gap {lo}-{hi}":>22s}' for lo, hi in buckets))
    print(' ' * 11 + '  '.join(f'{"centre px    IoU":>22s}' for _ in buckets))
    for name in names:
        if name not in rows:
            continue
        cells = []
        for lo, hi in buckets:
            sel = [(e, v) for g, e, v, _ in rows[name] if lo <= g <= hi]
            cells.append(f'{np.median([e for e, _ in sel]):9.2f} {np.mean([v for _, v in sel]):6.3f}'.rjust(22)
                         if sel else f'{"-":>22s}')
        print(f'{name:10s} ' + '  '.join(cells))

    print('\nlong gaps (>= 9 ticks) by class, median centre error px:')
    print(f'{"class":16s} ' + '  '.join(f'{n:>9s}' for n in ('ground', 'global', 'window20', 'shrunk', 'object', 'upright')) + '      n')
    for c in sorted({label for _, _, _, label in rows['ground']}):
        cells, count = [], 0
        for name in ('ground', 'global', 'window20', 'shrunk', 'object', 'upright'):
            sel = [e for g, e, _, label in rows.get(name, []) if label == c and g >= 9]
            count = max(count, len(sel))
            cells.append(f'{np.median(sel):9.2f}' if sel else f'{"-":>9s}')
        print(f'{c:16s} ' + '  '.join(cells) + f'  {count:5d}')


if __name__ == '__main__':
    main()
