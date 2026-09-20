#!/usr/bin/env python3
"""Measure each instance's real extent off the frames, by the one property these objects reliably have.

The earlier Lab-colour attempt latched onto shadows and neighbouring objects. Every one of these targets is
near-black against grass, tarmac or dirt, so the test that works is simple brightness: take the crop's own
median brightness as the background level and keep pixels far below it. Trees are dark but green, so the blob
is also required to be unsaturated. The largest such blob touching the centre is the object.

Each measurement is rendered back over the crop so it can be checked by eye rather than trusted.

    python3 eye_measure.py --cls large_tower --out-json extents.json --sheet gal.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_best_plan import H, W, load_geometry, to_ground

ROOT = Path(__file__).resolve().parents[2]
FRAMES = ROOT / 'artifacts/drone-validation-box-review-20260919/backgrounds'


def measure(im, cx, cy, half, dark=0.62, sat=0.34):
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    a = np.asarray(im.crop((x0, y0, x0 + 2 * half, y0 + 2 * half)), dtype=np.float32) / 255.0
    if a.size == 0:
        return None
    v = a.max(2)
    mn = a.min(2)
    s = np.where(v > 1e-6, (v - mn) / np.maximum(v, 1e-6), 0.0)
    bg = float(np.median(v))
    mask = (v < dark * bg) & (s < sat)
    if mask.sum() < 12:
        return None
    lbl = np.zeros(mask.shape, np.int32)
    cur = 0
    idx = np.argwhere(mask)
    pos = {tuple(p): i for i, p in enumerate(idx)}
    parent = list(range(len(idx)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    for i, (r, c) in enumerate(idx):
        for dr, dc in ((-1, 0), (0, -1), (-1, -1), (-1, 1)):
            j = pos.get((r + dr, c + dc))
            if j is not None:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups = defaultdict(list)
    for i in range(len(idx)):
        groups[find(i)].append(i)
    ctr = np.array([half, half], float)
    best, bestscore = None, None
    for g in groups.values():
        pts = idx[g]
        if len(pts) < 10:
            continue
        d = np.hypot(*(pts - ctr).T).min()
        score = len(pts) / (1 + d) ** 1.5
        if bestscore is None or score > bestscore:
            best, bestscore = pts, score
    if best is None:
        return None
    r0, c0 = (int(v) for v in best.min(0)); r1, c1 = (int(v) for v in best.max(0))
    return [x0 + c0, y0 + r0, x0 + c1 + 1, y0 + r1 + 1], len(best)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', default=str(ROOT / 'artifacts/drone-verifier-20260919/probes/probe-pruned.json'))
    ap.add_argument('--cls', required=True)
    ap.add_argument('--per-instance', type=int, default=7)
    ap.add_argument('--dark', type=float, default=0.62)
    ap.add_argument('--sat', type=float, default=0.34)
    ap.add_argument('--out-json', default=None)
    ap.add_argument('--sheet', default=None)
    a = ap.parse_args()
    G, _ = load_geometry(ROOT / 'artifacts/drone-flight-map-20260919/validation/meta.json')
    plan = json.loads(Path(a.plan).read_text())
    rows = []
    for f, boxes in plan['predictions_by_frame'].items():
        for b in boxes:
            if b['object_id'] == a.cls and b['confidence'] >= 0.8 and int(f) in G:
                bb = [b['bbox'][0] * W, b['bbox'][1] * H, b['bbox'][2] * W, b['bbox'][3] * H]
                if bb[1] < 4 or bb[3] > H - 4:
                    continue
                g = to_ground(G[int(f)], (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
                rows.append((int(f), g[0], g[1], bb))
    inst = []
    for f, gx, gy, bb in sorted(rows):
        for k in inst:
            if (k['x'] - gx) ** 2 + (k['y'] - gy) ** 2 <= 90 ** 2:
                n = k['n']; k['x'] = (k['x'] * n + gx) / (n + 1); k['y'] = (k['y'] * n + gy) / (n + 1)
                k['n'] += 1; k['rows'].append((f, bb)); break
        else:
            inst.append({'x': gx, 'y': gy, 'n': 1, 'rows': [(f, bb)]})
    inst = sorted([q for q in inst if q['n'] >= 8], key=lambda d: min(f for f, _ in d['rows']))
    out, cells = [], []
    for i, k in enumerate(inst):
        k['rows'].sort()
        step = max(1, len(k['rows']) // a.per_instance)
        picks = k['rows'][::step][:a.per_instance]
        meas = []
        for f, bb in picks:
            cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
            half = int(max(45, max(bb[2] - bb[0], bb[3] - bb[1]) * 1.6))
            im = Image.open(FRAMES / f'frame_{f:06d}.jpg')
            m = measure(im, cx, cy, half, a.dark, a.sat)
            if not m:
                continue
            ob, npx = m
            meas.append({'frame': f, 'obj': [round(v, 1) for v in ob], 'ours': [round(v, 1) for v in bb],
                         'w': round(ob[2] - ob[0], 1), 'h': round(ob[3] - ob[1], 1),
                         'dx': round((ob[0] + ob[2]) / 2 - cx, 1), 'dy': round((ob[1] + ob[3]) / 2 - cy, 1),
                         'ow': round(bb[2] - bb[0], 1), 'oh': round(bb[3] - bb[1], 1), 'px': npx})
            if a.sheet and len(cells) < 12 and f in (picks[1][0], picks[-2][0]):
                Zm = 6
                c = im.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half))).resize(
                    (half * 2 * Zm, half * 2 * Zm), Image.NEAREST)
                d = ImageDraw.Draw(c)
                d.rectangle([(bb[0] - cx + half) * Zm, (bb[1] - cy + half) * Zm,
                             (bb[2] - cx + half) * Zm, (bb[3] - cy + half) * Zm], outline=(255, 150, 40), width=2)
                d.rectangle([(ob[0] - cx + half) * Zm, (ob[1] - cy + half) * Zm,
                             (ob[2] - cx + half) * Zm, (ob[3] - cy + half) * Zm], outline=(60, 230, 120), width=2)
                d.text((6, 6), f'{a.cls} #{i+1} f{f}  ours {bb[2]-bb[0]:.0f}x{bb[3]-bb[1]:.0f} '
                               f'measured {ob[2]-ob[0]:.0f}x{ob[3]-ob[1]:.0f}', fill=(255, 255, 0))
                cells.append(c)
        if len(meas) < 3:
            out.append({'instance': i + 1, 'n': len(meas), 'status': 'too few'}); continue
        med = lambda key: float(np.median([m[key] for m in meas]))
        out.append({'instance': i + 1, 'frames': [meas[0]['frame'], meas[-1]['frame']], 'n': len(meas),
                    'ground': [round(float(k['x']), 1), round(float(k['y']), 1)],
                    'obj': [round(med('w'), 1), round(med('h'), 1)],
                    'ours': [round(med('ow'), 1), round(med('oh'), 1)],
                    'offset': [round(med('dx'), 1), round(med('dy'), 1)],
                    'per_frame': meas})
    for o in out:
        if o.get('status'):
            print(f"  #{o['instance']}: {o['status']}")
        else:
            print(f"  #{o['instance']} n={o['n']} frames {o['frames']} ground {o['ground']}  "
                  f"object {o['obj']} vs ours {o['ours']}  centre offset {o['offset']}")
    if a.out_json:
        Path(a.out_json).write_text(json.dumps({a.cls: out}, indent=1))
    if a.sheet and cells:
        cols = 2
        Wt = max(c.width for c in cells); Ht = max(c.height for c in cells)
        rowsn = (len(cells) + cols - 1) // cols
        sheet = Image.new('RGB', (cols * (Wt + 6), rowsn * (Ht + 6)), (10, 10, 12))
        for i, c in enumerate(cells):
            sheet.paste(c, ((i % cols) * (Wt + 6), (i // cols) * (Ht + 6)))
        if sheet.width > 1500:
            sheet = sheet.resize((1500, int(sheet.height * 1500 / sheet.width)), Image.LANCZOS)
        sheet.save(a.sheet, quality=92)
        print(f'  sheet {a.sheet} {sheet.size}  orange = ours, green = measured')


if __name__ == '__main__':
    main()
