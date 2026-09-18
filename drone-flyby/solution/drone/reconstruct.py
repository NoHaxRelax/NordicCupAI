#!/usr/bin/env python3
"""Combine same-frame native validation crops, preserving missing-pixel masks."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from PIL import Image


def reconstruct(records):
    first = records[0][1]
    width, height = first['original_width'], first['original_height']
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    coverage = np.zeros((height, width), dtype=bool)
    for path, record in records:
        if (record['frame'], record['original_width'], record['original_height']) != (first['frame'], width, height):
            raise ValueError('Mismatched frame or source dimensions')
        view = record['view']
        if view['resolution_level'] != 2:
            continue  # Never represent interpolated pixels as original pixels.
        x1, y1, x2, y2 = view['source_region_xyxy']
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('Invalid source region')
        with Image.open(path.parent / record['image_file']) as im:
            tile = np.asarray(im.convert('RGB'))
        if tile.shape != (y2 - y1, x2 - x1, 3):
            raise ValueError('Native crop size does not match source region')
        region = rgb[y1:y2, x1:x2]
        seen = coverage[y1:y2, x1:x2]
        if np.any(region[seen] != tile[seen]):
            raise ValueError(f'Conflicting pixels for frame {first["frame"]}; sequences may differ')
        region[:] = tile
        coverage[y1:y2, x1:x2] = True
    return rgb, coverage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('captures', type=Path, nargs='+', help='Explicitly select runs from the same scene')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    groups = defaultdict(list)
    for root in a.captures:
        for path in sorted(root.rglob('*.json')):
            record = json.loads(path.read_text())
            if 'view' in record and 'image_file' in record:
                groups[record['frame']].append((path, record))
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for frame, records in sorted(groups.items()):
        rgb, mask = reconstruct(records)
        stem = f'frame_{frame:06d}'
        Image.fromarray(np.dstack((rgb, mask.astype(np.uint8) * 255))).save(a.output / f'{stem}.png')
        entry = {'frame': frame, 'native_coverage': float(mask.mean()), 'complete': bool(mask.all()),
                 'image_file': f'{stem}.png', 'unknown_pixels': int((~mask).sum()),
                 'label_status': 'not supplied by validation API',
                 'capture_records': [str(path) for path, _ in records]}
        manifest.append(entry)
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'frames': len(manifest), 'complete_frames': sum(x['complete'] for x in manifest)}, indent=2))


if __name__ == '__main__':
    main()
