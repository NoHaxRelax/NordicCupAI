#!/usr/bin/env python3
"""Measure capture coverage and exact shared pixels across two validation replays."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from PIL import Image


def load(root):
    return [(p, json.loads(p.read_text())) for p in sorted(root.rglob('*.json'))]


def stats(records):
    indices = {r['frame_index'] for _, r in records}
    native_frames = [r['frame'] for _, r in records if r['view']['resolution_level'] == 2]
    return {
        'requests': len(records),
        'sequence_count': len({r['sequence_id'] for _, r in records}),
        'frame_range': [min(r['frame'] for _, r in records), max(r['frame'] for _, r in records)] if records else [],
        'missing_frame_indices': sorted(set(range(249)) - indices),
        'levels': dict(Counter(r['view']['resolution_level'] for _, r in records)),
        'native_frame_range': [min(native_frames), max(native_frames)] if native_frames else [],
        'camera_rejections': sum(r.get('camera_command_feedback') is not None for _, r in records),
        'capture_ms_max': max((r['capture_ms'] for _, r in records), default=None),
        'request_fields': sorted({k for _, r in records for k in r if k not in {
            'image_file', 'image_sha256', 'response', 'target_center', 'received_at', 'capture_ms'}}),
    }


def pixels(path, record):
    with Image.open(path.parent / record['image_file']) as im:
        return np.asarray(im.convert('RGB'))


def compare(a, b):
    by_frame = {r['frame']: (p, r) for p, r in b}
    overlaps = []
    identical_views = []
    for pa, ra in a:
        if ra['frame'] not in by_frame:
            continue
        pb, rb = by_frame[ra['frame']]
        va, vb = ra['view'], rb['view']
        if va['resolution_level'] == vb['resolution_level'] and va['source_region_xyxy'] == vb['source_region_xyxy']:
            identical_views.append({'frame': ra['frame'], 'level': va['resolution_level'],
                                    'pixels_identical': bool(np.array_equal(pixels(pa, ra), pixels(pb, rb)))})
        if va['resolution_level'] != 2 or vb['resolution_level'] != 2:
            continue
        aa, bb = va['source_region_xyxy'], vb['source_region_xyxy']
        x1, y1, x2, y2 = max(aa[0], bb[0]), max(aa[1], bb[1]), min(aa[2], bb[2]), min(aa[3], bb[3])
        if x2 <= x1 or y2 <= y1:
            continue
        xa = pixels(pa, ra)[y1-aa[1]:y2-aa[1], x1-aa[0]:x2-aa[0]]
        xb = pixels(pb, rb)[y1-bb[1]:y2-bb[1], x1-bb[0]:x2-bb[0]]
        changed = np.any(xa != xb, axis=2)
        overlaps.append({'frame': ra['frame'], 'pixels_compared': int(changed.size), 'different_pixels': int(changed.sum())})
    return {'identical_geometry_views': identical_views, 'native_overlap_frames': len(overlaps),
            'native_pixels_compared': sum(r['pixels_compared'] for r in overlaps),
            'native_pixels_different': sum(r['different_pixels'] for r in overlaps), 'overlaps': overlaps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('probe_a', type=Path)
    parser.add_argument('probe_b', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    a, b = load(args.probe_a), load(args.probe_b)
    result = {'probe_a': stats(a), 'probe_b': stats(b), 'comparison': compare(a, b)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({**result, 'comparison': {k: v for k, v in result['comparison'].items() if k != 'overlaps'}}, indent=2))


if __name__ == '__main__':
    main()
