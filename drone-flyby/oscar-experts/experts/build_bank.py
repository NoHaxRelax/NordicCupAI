"""Build a training-only sprite bank for the per-object experts.

Source: the reviewed sprite export (Oscar approved or redrew every outline).
Every sprite is checked against the grid dataset's track splits: reference
frames are training data, and validation sprites must come from training-split
tracks. Sprites from tracks or frames in label_exclusions.json are dropped.
Output per sprite: the BGRA cutout, a hard mask from its alpha, and a manifest
row carrying the organiser box, track, zoom, source frame and hashes.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(reviewed, grid, splits_manifest, exclusions, output, alpha_threshold=128):
    reviewed, grid, output = Path(reviewed), Path(grid), Path(output)
    bank = json.loads((reviewed / 'bank.json').read_text())
    manifest = json.loads((grid / 'manifest.json').read_text())
    if hashlib.sha256((grid / 'manifest.json').read_bytes()).hexdigest() != bank['source_manifest_sha256']:
        raise ValueError('Reviewed bank was cut from a different tile manifest')
    records = {r['id']: r for r in manifest['records']}
    splits = json.loads((Path(splits_manifest) / 'manifest.json').read_text())['track_splits']
    excluded_rules = json.loads(Path(exclusions).read_text()) if exclusions else []
    output.mkdir(parents=True, exist_ok=False)
    rows, dropped = [], []
    for s in bank['sprites']:
        rec = records[s['tile']]
        if s['split'] != 'train' or rec['split'] != 'train':
            raise ValueError(f"Non-training sprite in reviewed bank: {s['id']}")
        if sha(grid / rec['file']) != s['tile_sha256']:
            raise ValueError(f"Tile hash mismatch for {s['tile']}")
        tracks = sorted({a['track_id'] for a in rec['annotations'] if a['class_name'] == s['class_name']})
        # Reference-flight objects are training data by contract; validation tracks must be in the training split.
        if s['source'] == 'reference':
            if not tracks or not all(t.startswith('reference-') for t in tracks):
                raise ValueError(f"Reference sprite {s['id']} has unexpected tracks: {tracks}")
        elif any(splits.get(t) != 'train' for t in tracks):
            raise ValueError(f"Sprite {s['id']} touches a non-training track: {tracks}")
        rule = next((e for e in excluded_rules if e['track_id'] in tracks and ('frames' not in e or s['frame'] in e['frames'])), None)
        if rule is not None:
            dropped.append(dict(id=s['id'], class_name=s['class_name'], track=rule['track_id'], reason=rule['reason']))
            continue
        src = reviewed / s['file']
        if sha(src) != s['sha256']:
            raise ValueError(f"Sprite hash mismatch: {s['file']}")
        rgba = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError(f"Sprite is not BGRA: {s['file']}")
        mask = (rgba[:, :, 3] >= alpha_threshold).astype(np.uint8) * 255
        if np.count_nonzero(mask) < 4:
            raise ValueError(f"Empty mask: {s['file']}")
        stem = s['id'].replace(':', '__')
        class_dir = output / s['class_name']
        class_dir.mkdir(exist_ok=True)
        sprite_path, mask_path = class_dir / f'{stem}.png', class_dir / f'{stem}-mask.png'
        cv2.imwrite(str(sprite_path), rgba)
        cv2.imwrite(str(mask_path), mask)
        rows.append(dict(id=s['id'], class_name=s['class_name'], file=str(sprite_path.relative_to(output)),
                         mask_file=str(mask_path.relative_to(output)), sha256=sha(sprite_path), mask_sha256=sha(mask_path),
                         zoom=s['zoom'], source=s['source'], frame=s['frame'], tile=s['tile'], tile_sha256=s['tile_sha256'],
                         tracks=tracks, split='train', review_status=s['review_status'], size=s['size'],
                         organizer_box_in_sprite=s['box_in_sprite'], foreground_pixels=int(np.count_nonzero(mask)),
                         alpha_threshold=alpha_threshold))
    doc = dict(format='expert-sprite-bank-v1', policy='Training split only: reference frames plus training-split validation tracks. '
               'Pixels are the reviewed cutouts; masks are alpha >= threshold. Dev and reserved tracks never enter.',
               reviewed_bank=str(reviewed.relative_to(ROOT)), reviewed_bank_sha256=sha(reviewed / 'bank.json'),
               grid_manifest=str(grid.relative_to(ROOT)), grid_manifest_sha256=sha(grid / 'manifest.json'),
               splits_manifest=str(Path(splits_manifest).relative_to(ROOT)), splits_manifest_sha256=sha(Path(splits_manifest) / 'manifest.json'),
               exclusions=str(Path(exclusions).relative_to(ROOT)) if exclusions else None,
               exclusions_sha256=sha(exclusions) if exclusions else None,
               classes=dict(sorted(Counter(r['class_name'] for r in rows).items())),
               by_zoom=dict(sorted(Counter(r['zoom'] for r in rows).items())),
               by_source=dict(Counter(r['source'] for r in rows)), dropped=dropped, sprites=rows)
    (output / 'manifest.json').write_text(json.dumps(doc, indent=1) + '\n')
    return doc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviewed', type=Path, default=ROOT / 'data/drone/sprite-bank-reviewed-20260918')
    p.add_argument('--grid', type=Path, default=ROOT / 'data/drone/grid-comparison-20260918-v1/256', help='tile dataset the reviewed bank was cut from')
    p.add_argument('--splits', type=Path, default=ROOT / 'data/drone/grid384-20260918-v1', help='dataset whose track_splits define train/dev/reserved')
    p.add_argument('--exclusions', type=Path, default=ROOT / 'data/drone/sprite-mask-review-20260918/label_exclusions.json')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    doc = build(a.reviewed, a.grid, a.splits, a.exclusions, a.output)
    print(json.dumps(dict(sprites=len(doc['sprites']), classes=doc['classes'], by_zoom=doc['by_zoom'],
                          by_source=doc['by_source'], dropped=[d['id'] for d in doc['dropped']])))


if __name__ == '__main__':
    main()
