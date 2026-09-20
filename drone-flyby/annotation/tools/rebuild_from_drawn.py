#!/usr/bin/env python3
"""Rebuild the plan from the boxes drawn in the review page, and add the tracks the drawings revealed.

Each instance gets its own straight-line fit of width, height and centre offset against frame number, taken
through the frames that were drawn and evaluated on every frame of the track. A straight line is used rather
than the perspective scale alone because the drawings show these objects rotating as the drone passes - large
tower 1 goes 59x66 to 70x54, almost the same area - which a single reference size cannot follow.

Objects the drawings found that the tracker never saw are added as new tracks: the ground spot is fixed, so
the flight-map homography places the box on every frame it is visible for and the perspective scale sizes it.

    python3 rebuild_from_drawn.py --answers box-answers.json --out probe-drawn.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_best_plan import H, W, load_geometry, to_ground
from persp_probe import cluster, scale_at

ROOT = Path(__file__).resolve().parents[2]

# Objects the review page turned up that no track covers. The spiky launcher at installation 2 stands beside
# the lattice tower and only separates from it near the bottom of the frame, which is why the tracker never
# split them. Installation 1 has only the one launcher (the drawings there land on the spot we already track)
# and installation 3 has none - the nine-position ring probe scored exactly 0 over 249 delivered frames.
NEW_TRACKS = [
    {'cls': 'medium_launcher', 'ground': (-70.5, -8600.0), 'ref': 110, 'size': (33.0, 28.0),
     'frames': (92, 124), 'note': 'spiky launcher beside large tower 2'},
]


def img_of(G, f, g):
    p = np.linalg.inv(G[f]) @ np.array([g[0], g[1], 1.0]); return p[:2] / p[2]


def fit(xs, ys):
    """straight line through the drawn samples; a flat line when only one was drawn"""
    if len(xs) == 1:
        return 0.0, ys[0]
    a, b = np.polyfit(np.array(xs, float), np.array(ys, float), 1)
    return float(a), float(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--answers', required=True)
    ap.add_argument('--plan', default=str(ROOT / 'artifacts/drone-verifier-20260919/probes/probe-pruned.json'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--match', type=float, default=70.0, help='px a drawing may sit from the box it corrects')
    ap.add_argument('--gamma', type=float, default=0.9)
    a = ap.parse_args()
    G, _ = load_geometry(ROOT / 'artifacts/drone-flight-map-20260919/validation/meta.json')
    plan = json.loads(Path(a.plan).read_text())
    drawn = [d for d in json.loads(Path(a.answers).read_text())['answers'] if not d['absent'] and d['box']]

    byc = defaultdict(list)
    for f, boxes in plan['predictions_by_frame'].items():
        for b in boxes:
            if b['confidence'] >= 0.8 and int(f) in G:
                g = to_ground(G[int(f)], (b['bbox'][0] + b['bbox'][2]) / 2 * W,
                              (b['bbox'][1] + b['bbox'][3]) / 2 * H)
                byc[b['object_id']].append((int(f), g[0], g[1], b))
    inst = {c: [k for k in cluster(rows, 90.0) if k['n'] >= 8] for c, rows in byc.items()}

    # attach every drawing to the track box it is closest to on that frame; a drawing too far from any of them
    # is a different object and is reported rather than folded in
    samples, orphans = defaultdict(list), []
    for d in drawn:
        cx, cy = (d['box'][0] + d['box'][2]) / 2, (d['box'][1] + d['box'][3]) / 2
        best, bestd = None, 1e9
        for i, k in enumerate(inst.get(d['cls'], [])):
            for f, b in k['rows']:
                if f != d['frame']:
                    continue
                ox = (b['bbox'][0] + b['bbox'][2]) / 2 * W
                oy = (b['bbox'][1] + b['bbox'][3]) / 2 * H
                dd = float(np.hypot(ox - cx, oy - cy))
                if dd < bestd:
                    best, bestd = (i, b, ox, oy), dd
        if best is None or bestd > a.match:
            orphans.append((d, round(bestd, 1))); continue
        i, b, ox, oy = best
        samples[(d['cls'], i)].append({'f': d['frame'], 'w': d['box'][2] - d['box'][0],
                                       'h': d['box'][3] - d['box'][1], 'dx': cx - ox, 'dy': cy - oy})

    pbf = {f: [dict(b) for b in v] for f, v in plan['predictions_by_frame'].items()}
    rewritten = 0
    print(f'{len(drawn)} drawings -> {len(samples)} tracks corrected')
    for (cls, i), ss in sorted(samples.items()):
        ss.sort(key=lambda s: s['f'])
        fs = [s['f'] for s in ss]
        aw, bw = fit(fs, [s['w'] for s in ss])
        ah, bh = fit(fs, [s['h'] for s in ss])
        ax_, bx = fit(fs, [s['dx'] for s in ss])
        ay_, by = fit(fs, [s['dy'] for s in ss])
        k = inst[cls][i]
        lo, hi = min(f for f, _ in k['rows']), max(f for f, _ in k['rows'])
        for f, b in k['rows']:
            fc = min(max(f, fs[0] - 12), fs[-1] + 12)      # do not extrapolate the line far past the drawings
            w, h = max(6.0, aw * fc + bw), max(6.0, ah * fc + bh)
            x0, y0, x1, y1 = b['bbox']
            cx = (x0 + x1) / 2 * W + ax_ * fc + bx
            cy = (y0 + y1) / 2 * H + ay_ * fc + by
            nb = [max(0.0, cx - w / 2) / W if x0 > 0 else 0.0, max(0.0, cy - h / 2) / H if y0 > 0 else 0.0,
                  min(W, cx + w / 2) / W if x1 < 1 else 1.0, min(H, cy + h / 2) / H if y1 < 1 else 1.0]
            for t in pbf[str(f)]:
                if t['object_id'] == cls and t['bbox'] == b['bbox']:
                    t['bbox'] = [round(v, 6) for v in nb]; rewritten += 1; break
        print(f"  {cls:16s} #{i+1} frames {lo}-{hi}: drawn " +
              ' '.join(f"f{s['f']}:{s['w']:.0f}x{s['h']:.0f}" for s in ss) +
              f"  -> {aw*lo+bw:.0f}x{ah*lo+bh:.0f} at {lo}, {aw*hi+bw:.0f}x{ah*hi+bh:.0f} at {hi}")

    added = 0
    for t in NEW_TRACKS:
        s0 = scale_at(G, t['ref'], t['ground'])
        f0, f1 = t['frames']
        for f in range(f0, f1 + 1):
            if f not in G:
                continue
            x, y = img_of(G, f, t['ground'])
            r = (scale_at(G, f, t['ground']) / s0) ** a.gamma
            w, h = t['size'][0] * r, t['size'][1] * r
            nb = [max(0.0, x - w / 2) / W, max(0.0, y - h / 2) / H,
                  min(W, x + w / 2) / W, min(H, y + h / 2) / H]
            if nb[2] - nb[0] < 0.001 or nb[3] - nb[1] < 0.001 or x < -w or x > W + w or y < -h or y > H + h:
                continue
            pbf.setdefault(str(f), []).append({'object_id': t['cls'], 'bbox': [round(v, 6) for v in nb],
                                               'confidence': 0.9})
            added += 1
        print(f"  NEW {t['cls']:12s} {t['note']}: {added} boxes over frames {f0}-{f1}")

    plan['predictions_by_frame'] = {f: v for f, v in sorted(pbf.items(), key=lambda kv: int(kv[0])) if v}
    plan['name'] = Path(a.out).stem
    Path(a.out).write_text(json.dumps(plan))
    print(f'{a.out}: {rewritten} boxes rewritten, {added} added, '
          f'{sum(len(v) for v in plan["predictions_by_frame"].values())} total')
    for d, dd in orphans:
        print(f"  not folded in (nearest track box {dd} px away): {d['cls']} f{d['frame']} "
              f"{d['box'][2]-d['box'][0]:.0f}x{d['box'][3]-d['box'][1]:.0f}  [{d['inst']}]")


if __name__ == '__main__':
    main()
