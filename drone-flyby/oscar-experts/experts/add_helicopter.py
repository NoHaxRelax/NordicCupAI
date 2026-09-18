"""Add a helicopter template to the expert bank from the v4 library sprite (training tile, reference frame 0).

Oscar rejected the automatic helicopter mask in his review, so this row is marked
review_status='claude-auto' and must be replaced when a reviewed outline exists.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .build_bank import sha, ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, default=ROOT / 'data/drone/sprite-library-20260918-v4')
    p.add_argument('--grid', type=Path, default=ROOT / 'data/drone/grid-comparison-20260918-v1/256')
    p.add_argument('--bank', type=Path, default=ROOT / 'data/drone/expert-bank-20260918-v1')
    a = p.parse_args()
    lib = json.loads((a.library / 'library.json').read_text())
    entry = next(e for e in lib['entries'] if e['class_name'] == 'helicopter')
    manifest = json.loads((a.grid / 'manifest.json').read_text())
    rec = next(r for r in manifest['records'] if r['id'] == entry['source_tile'])
    assert rec['split'] == 'train' and rec['source'] == 'reference'
    ann = next(x for x in rec['annotations'] if x['class_name'] == 'helicopter' and x['fully_contained'])
    sprite = cv2.imread(str(a.library / 'helicopter/sprite.png'))
    mask = cv2.imread(str(a.library / 'helicopter/mask.png'), 0) > 127
    # keep the largest component plus anything within 3 px of it (blades)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    main = labels == (1 + int(np.argmax(stats[1:, 4])))
    near = cv2.dilate(main.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    mask = mask & (main | near)
    rgba = np.dstack([sprite, (mask * 255).astype(np.uint8)])
    X, Y = entry['crop_xyxy'][:2]
    bx1, by1, bx2, by2 = [int(round(v)) for v in ann['bbox_xyxy']]
    out = a.bank / 'helicopter'
    out.mkdir(exist_ok=True)
    stem = 'auto__' + entry['source_tile'].replace(':', '__')
    cv2.imwrite(str(out / f'{stem}.png'), rgba)
    cv2.imwrite(str(out / f'{stem}-mask.png'), (mask * 255).astype(np.uint8))
    doc = json.loads((a.bank / 'manifest.json').read_text())
    doc['sprites'] = [r for r in doc['sprites'] if r['class_name'] != 'helicopter']
    doc['sprites'].append(dict(id='auto:' + entry['source_tile'] + ':0', class_name='helicopter', file=f'helicopter/{stem}.png', mask_file=f'helicopter/{stem}-mask.png',
                               sha256=sha(out / f'{stem}.png'), mask_sha256=sha(out / f'{stem}-mask.png'), zoom=rec['zoom'], source='reference', frame=rec['frame'],
                               tile=entry['source_tile'], tile_sha256=rec['sha256'], tracks=[ann['track_id']], split='train', review_status='claude-auto',
                               size=[int(rgba.shape[1]), int(rgba.shape[0])], organizer_box_in_sprite=[bx1 - X, by1 - Y, bx2 - X, by2 - Y],
                               foreground_pixels=int(mask.sum()), alpha_threshold=128, note='v4 library mask (Higgsfield + blade growth); not reviewed by Oscar'))
    doc['classes']['helicopter'] = 1
    (a.bank / 'manifest.json').write_text(json.dumps(doc, indent=1) + '\n')
    print(json.dumps(dict(added=stem, foreground=int(mask.sum()), box=[bx1 - X, by1 - Y, bx2 - X, by2 - Y], size=rgba.shape[1::-1])))


if __name__ == '__main__':
    main()
