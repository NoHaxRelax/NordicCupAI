#!/usr/bin/env python3
"""Per-class coverage from single-class validation attempts: did we discover every instance?

For each class with an attempt folder api-attempts/only-<class>-try1 (delivery.json + frames/only-<class>.jsonl):
  score -> AP = 13 x score (13 classes in the validation truth, proven by the hangar probes)
  E     = boxes we emitted for the class, and the instances they form (boxes projected onto the flight map and
          clustered; a cluster = one physical object)
  T_if_all_right = E / AP: the truth frame count implied if every box we emitted is a hit (COCO AP = recall then);
          a truth count far above E means whole instances were never answered, not just entry/exit frames
  annotated instances = v8 pseudo-label clusters + Elias's hidden tracks, for reference only (not truth)

    python3 class_coverage.py --attempts artifacts/drone-verifier-20260919/api-attempts --meta artifacts/drone-flight-map-20260919/validation/meta.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W, H = 3840, 2160
CLASSES = ['hangar', 'tank', 'helicopter', 'jet_plane', 'large_launcher', 'large_tower', 'mine_roller', 'small_tower',
           'small_plane', 'medium_plane', 'small_launcher', 'medium_launcher', 'ta-ta']


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def apply(M, x, y):
    X = M[0][0] * x + M[0][1] * y + M[0][2]; Y = M[1][0] * x + M[1][1] * y + M[1][2]; Z = M[2][0] * x + M[2][1] * y + M[2][2]
    return X / Z, Y / Z


def cluster(points, radius):
    out = []
    for x, y, f in points:
        for c in out:
            if (c['x'] - x) ** 2 + (c['y'] - y) ** 2 <= radius * radius:
                n = c['n']; c['x'] = (c['x'] * n + x) / (n + 1); c['y'] = (c['y'] * n + y) / (n + 1); c['n'] += 1; c['frames'].append(f); break
        else:
            out.append({'x': x, 'y': y, 'n': 1, 'frames': [f]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--attempts', default=str(ROOT / 'artifacts/drone-verifier-20260919/api-attempts'))
    ap.add_argument('--meta', default=str(ROOT / 'artifacts/drone-flight-map-20260919/validation/meta.json'))
    ap.add_argument('--v8', default=str(ROOT / 'artifacts/drone-api-tests/score-anchored-v8-full-20260918/full-validation-plan.json'))
    ap.add_argument('--hidden2', default=str(ROOT / 'drone/crop_picker/elias-data/validation_hidden2.json'))
    ap.add_argument('--radius', type=float, default=110.0, help='cluster radius in ground px (about 25 m: tall objects drift on the map as they cross the frame)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    meta = json.loads(Path(a.meta).read_text())
    G = {f['frame']: f['ground_from_image'] for f in meta['frames']}
    v8 = json.loads(Path(a.v8).read_text())['predictions_by_frame']
    hidden = json.loads(Path(a.hidden2).read_text())['tracks']

    ann = defaultdict(list)
    for fr, rows in v8.items():
        if int(fr) not in G:
            continue
        for o in rows:
            b = o['bbox']; x, y = apply(G[int(fr)], (b[0] + b[2]) / 2 * W, (b[1] + b[3]) / 2 * H)
            ann[o['object_id']].append((x, y, int(fr)))
    ann_clusters = {c: cluster(sorted(pts, key=lambda p: p[2]), a.radius) for c, pts in ann.items()}
    hidden_by_class = defaultdict(int)
    for t in hidden:
        hidden_by_class[t['what']] += 1

    rows = []
    for c in CLASSES:
        tries = sorted(Path(a.attempts).glob(f'only-{c}-try*'))
        delivs = [(json.loads((t / 'delivery.json').read_text()), t) for t in tries if (t / 'delivery.json').exists()]
        if not delivs:
            rows.append({'class': c, 'status': 'not run'}); continue
        # the attempt that received the most frames (the organiser skips frames on its clock); ties -> highest score
        deliv, folder = max(delivs, key=lambda dt: (int(dt[0].get('frames') or 0), float(dt[0]['score'])))
        score = float(deliv['score']); ap_ = 13 * score
        logf = next(iter(sorted((Path(a.attempts) / f'only-{c}-try1' / 'frames').glob('*.jsonl'))), folder / 'frames' / f'only-{c}.jsonl')
        E = 0; pts = []; conf = []
        if logf.exists():
            for line in open(logf):
                r = json.loads(line); f = int(r['frame'])
                for x in r.get('response') or []:
                    if x['object_id'] != c:
                        continue
                    E += 1; conf.append(float(x['confidence']))
                    if f in G:
                        b = x['bbox']; px, py = apply(G[f], (b[0] + b[2]) / 2 * W, (b[1] + b[3]) / 2 * H); pts.append((px, py, f))
        ours = cluster(sorted(pts, key=lambda p: p[2]), a.radius)
        ours_big = [k for k in ours if k['n'] >= 3]
        annc = ann_clusters.get(c, []); annc_big = [k for k in annc if k['n'] >= 3]
        rows.append({'class': c, 'score': round(score, 5), 'ap': round(ap_, 3), 'frames_delivered': deliv.get('frames'),
                     'emitted': E, 'emitted_instances': len(ours_big), 'emitted_instance_frames': sorted((min(k['frames']), max(k['frames']), k['n']) for k in ours_big),
                     'truth_frames_if_all_right': round(E / ap_, 1) if ap_ > 0 else None,
                     'annotated_frames': sum(k['n'] for k in annc), 'annotated_instances': len(annc_big), 'hidden_tracks': hidden_by_class.get(c, 0),
                     'conf_median': round(sorted(conf)[len(conf) // 2], 3) if conf else None})
    print(f"{'class':16s} {'score':>8s} {'AP':>6s} {'emit':>5s} {'inst':>4s} {'T if all right':>14s} {'annot frames':>12s} {'annot inst':>10s} {'hidden':>6s}")
    for r in rows:
        if r.get('status'):
            print(f"{r['class']:16s} {r['status']}"); continue
        print(f"{r['class']:16s} {r['score']:8.5f} {r['ap']:6.3f} {r['emitted']:5d} {r['emitted_instances']:4d} {str(r['truth_frames_if_all_right']):>14s} {r['annotated_frames']:12d} {r['annotated_instances']:10d} {r['hidden_tracks']:6d}   instances(frames): {r['emitted_instance_frames']}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
