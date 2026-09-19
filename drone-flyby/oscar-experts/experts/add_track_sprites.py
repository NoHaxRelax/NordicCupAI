"""Cut auto-masked sprites of object tracks the bank does not cover, from TRAINING-split grid tiles only.

Pass 7 misses were concentrated in tracks without a sprite (large-launcher-c-047-078: 41 of 48 misses,
large-tower-038-067: 9 of 9, the two medium-plane tracks, two tank tracks). One frame gives three sprites
(the L0/L1/L2 tiles of that frame share the geometry; the mask is cut on the sharp L2 tile and reused).
Masks come from GrabCut seeded by the organizer box (fallback: ground-colour distance), so every row is
marked review_status='claude-auto' for Oscar to replace with a reviewed outline; a contact sheet is written
next to the bank for that review.

  python -m drone.experts.add_track_sprites --tracks large_launcher:large-launcher-c-047-078 ... [--frames-per-track 2]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .build_bank import sha, ROOT


def pick_records(manifest, cls, track):
    """{frame: {zoom: record}} for frames where the track is fully contained at all three zooms (train split only)."""
    by_frame = {}
    for r in manifest['records']:
        if r['split'] != 'train':
            continue
        ann = [a for a in r['annotations'] if a['track_id'] == track and a['class_name'] == cls and a['fully_contained']]
        if not ann:
            continue
        b = ann[0]['bbox_xyxy']
        margin = min(b[0], b[1], 384 - b[2], 384 - b[3])  # distance to the tile border: prefer the tile that holds the object best
        cur = by_frame.setdefault(r['frame'], {}).get(r['zoom'])
        if cur is None or margin > cur[1]:
            by_frame[r['frame']][r['zoom']] = (r, margin, ann[0])
    return {f: z for f, z in by_frame.items() if set(z) == {0, 1, 2}}


def auto_mask(crop, box):
    """Foreground of the object inside `box` (crop coordinates): GrabCut seeded by the box, ground-colour fallback."""
    x1, y1, x2, y2 = box
    h, w = crop.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        rect = (max(0, x1 - 1), max(0, y1 - 1), min(w, x2 + 1) - max(0, x1 - 1), min(h, y2 + 1) - max(0, y1 - 1))
        cv2.grabCut(crop, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        fg = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    except cv2.error:
        fg = np.zeros((h, w), bool)
    inside = np.zeros((h, w), bool); inside[y1:y2, x1:x2] = True
    fraction = fg[inside].mean() if inside.any() else 0.
    method = 'grabcut'
    if not .2 <= fraction <= .97:
        # ground colour from the ring outside the box; object = pixels that differ from it
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        ring = lab[~inside]
        med = np.median(ring, 0); spread = np.median(np.abs(ring - med), 0) * 1.4826 + 2.
        z = np.sqrt((((lab - med) / spread) ** 2).sum(2))
        fg = (z > 2.5) & inside
        method = 'colour'
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8))
    if count > 1:
        keep = [i for i in range(1, count) if stats[i, 4] >= max(12, .02 * stats[1:, 4].max())]
        fg = np.isin(labels, keep)
    # fill enclosed holes (windows, dark panels)
    flood = fg.astype(np.uint8).copy(); ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 1)
    fg = fg | (flood == 0)
    return fg & inside, method


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid', type=Path, default=ROOT / 'data/drone/grid384-20260918-v1')
    p.add_argument('--bank', type=Path, default=ROOT / 'data/drone/expert-bank-20260918-v1')
    p.add_argument('--tracks', nargs='+', required=True, help='class:track_id entries')
    p.add_argument('--frames-per-track', type=int, default=2)
    p.add_argument('--margin', type=int, default=10)
    a = p.parse_args()
    manifest = json.loads((a.grid / 'manifest.json').read_text())
    doc = json.loads((a.bank / 'manifest.json').read_text())
    existing = {r['id'] for r in doc['sprites']}
    sheet, added = [], []
    for entry in a.tracks:
        cls, track = entry.split(':', 1)
        frames = pick_records(manifest, cls, track)
        if not frames:
            print(f'{entry}: no training frame fully contained at all zooms'); continue
        order = sorted(frames)
        picks = [order[int(round(i))] for i in np.linspace(0, len(order) - 1, min(a.frames_per_track, len(order)))]
        for frame in dict.fromkeys(picks):
            recs = frames[frame]
            rec2, _, ann2 = recs[2]
            tile2 = cv2.imread(str(a.grid / rec2['file']))
            b = [int(round(v)) for v in ann2['bbox_xyxy']]
            cx1, cy1 = max(0, b[0] - a.margin), max(0, b[1] - a.margin)
            cx2, cy2 = min(384, b[2] + a.margin), min(384, b[3] + a.margin)
            box = (b[0] - cx1, b[1] - cy1, b[2] - cx1, b[3] - cy1)
            mask, method = auto_mask(tile2[cy1:cy2, cx1:cx2], box)
            if mask.sum() < 30:
                print(f'{entry} frame {frame}: mask too small ({int(mask.sum())} px), skipped'); continue
            out = a.bank / cls; out.mkdir(exist_ok=True)
            for zoom in (0, 1, 2):
                rec, _, ann = recs[zoom]
                bz = [int(round(v)) for v in ann['bbox_xyxy']]
                ox, oy = bz[0] - b[0], bz[1] - b[1]  # same frame geometry; tiles may be cut at a different origin
                tile = cv2.imread(str(a.grid / rec['file']))
                crop = tile[cy1 + oy:cy2 + oy, cx1 + ox:cx2 + ox]
                if crop.shape[:2] != mask.shape:
                    print(f'{entry} frame {frame} L{zoom}: crop geometry differs, skipped'); continue
                sid = f"auto:{rec['id']}:{ann.get('instance', 0)}"
                if sid in existing:
                    print(f'{sid} already in bank'); continue
                stem = 'auto__' + rec['id'].replace(':', '__')
                rgba = np.dstack([crop, (mask * 255).astype(np.uint8)])
                cv2.imwrite(str(out / f'{stem}.png'), rgba)
                cv2.imwrite(str(out / f'{stem}-mask.png'), (mask * 255).astype(np.uint8))
                doc['sprites'].append(dict(id=sid, class_name=cls, file=f'{cls}/{stem}.png', mask_file=f'{cls}/{stem}-mask.png',
                                           sha256=sha(out / f'{stem}.png'), mask_sha256=sha(out / f'{stem}-mask.png'), zoom=zoom,
                                           source='validation' if rec['id'].startswith('validation') else 'reference', frame=rec['frame'],
                                           tile=rec['id'], tile_sha256=rec.get('sha256'), tracks=[track], split='train', review_status='claude-auto',
                                           size=[int(rgba.shape[1]), int(rgba.shape[0])], organizer_box_in_sprite=list(box), foreground_pixels=int(mask.sum()),
                                           alpha_threshold=128, note=f'auto mask ({method}) from the L2 tile of frame {frame}; not reviewed by Oscar'))
                doc['classes'][cls] = doc['classes'].get(cls, 0) + 1
                doc['by_zoom'][str(zoom)] = doc['by_zoom'].get(str(zoom), 0) + 1
                existing.add(sid); added.append(sid)
                if zoom == 2:
                    vis = crop.copy(); vis[~mask] = (vis[~mask] * .35).astype(np.uint8)
                    cv2.rectangle(vis, box[:2], box[2:], (0, 255, 255), 1)
                    cell = np.zeros((160, 160, 3), np.uint8); s_ = min(150 / max(vis.shape[:2]), 4.)
                    small = cv2.resize(np.hstack([crop, vis]), None, fx=s_ / 2 if vis.shape[1] * 2 * s_ > 150 else s_, fy=s_ / 2 if vis.shape[1] * 2 * s_ > 150 else s_, interpolation=cv2.INTER_NEAREST)
                    cell[:min(160, small.shape[0]), :min(160, small.shape[1])] = small[:160, :160]
                    cv2.putText(cell, f'{cls[:12]} f{frame} {method[:4]} {int(mask.mean() * 100)}%', (2, 156), cv2.FONT_HERSHEY_SIMPLEX, .32, (255, 255, 255), 1)
                    sheet.append(cell)
            print(json.dumps(dict(track=track, frame=frame, method=method, foreground=int(mask.sum()), box_fill=round(float(mask.sum()) / max(1, (box[2] - box[0]) * (box[3] - box[1])), 2))))
    (a.bank / 'manifest.json').write_text(json.dumps(doc, indent=1) + '\n')
    if sheet:
        cols = 6; rows_ = [np.hstack(sheet[i:i + cols] + [np.zeros((160, 160, 3), np.uint8)] * (cols - len(sheet[i:i + cols]))) for i in range(0, len(sheet), cols)]
        cv2.imwrite(str(a.bank / 'auto-review-sheet.png'), np.vstack(rows_))
    print(json.dumps(dict(added=len(added), classes=doc['classes'])))


if __name__ == '__main__':
    main()
