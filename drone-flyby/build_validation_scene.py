"""Build a local scene from reconstructed validation frames and participant labels.

The organizer's local_evaluator.py can then replay the whole validation flight
offline. The labels are participant pseudo-labels (reviewed tracks, score-
anchored projections and algorithmic completions), not organizer ground truth,
and they are incomplete: the resulting mAP is a proxy for comparing our own
configurations, not an estimate of the official score.

    python build_validation_scene.py --frames /path/to/reconstructed-validation \
        --labels /path/to/score-anchored-validation-v8 --output src/validation
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames', type=Path, required=True, help='Directory with frame_NNNNNN.png source frames')
    p.add_argument('--labels', type=Path, required=True, help='Directory with per-track JSON files (bbox_source_xyxy rows)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--exclude-status', nargs='*', default=[], help='review_status values to drop, e.g. algorithmic_bottom_completion')
    p.add_argument('--copy', action='store_true', help='Copy frames instead of symlinking them')
    p.add_argument('--width', type=int, default=3840)
    p.add_argument('--height', type=int, default=2160)
    a = p.parse_args()
    images = a.output/'images'; annotations = a.output/'annotations'
    images.mkdir(parents=True, exist_ok=True); annotations.mkdir(parents=True, exist_ok=True)
    frames = {}
    for path in sorted(a.frames.glob('frame_*.png')):
        frame = int(path.stem.split('_')[1])
        target = images/path.name
        if not target.exists():
            if a.copy:
                target.write_bytes(path.read_bytes())
            else:
                target.symlink_to(path.resolve())
        frames[frame] = target
    boxes = defaultdict(list); status = Counter(); tracks = 0
    for path in sorted(a.labels.glob('*.json')):
        doc = json.loads(path.read_text())
        if 'annotations' not in doc or 'track_id' not in doc:
            continue
        tracks += 1
        for row in doc['annotations']:
            if row.get('review_status') in a.exclude_status:
                continue
            x1, y1, x2, y2 = row['bbox_source_xyxy']
            x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
            x2, y2 = min(a.width-1, int(round(x2))), min(a.height-1, int(round(y2)))
            if x2 <= x1 or y2 <= y1 or int(row['frame']) not in frames:
                continue
            boxes[int(row['frame'])].append({'object_id': row['class'], 'bbox': [x1, y1, x2, y2],
                                             'review_status': row.get('review_status'), 'track_id': doc['track_id']})
            status[row.get('review_status')] += 1
    counts = Counter()
    for frame in frames:
        rows = boxes.get(frame, [])
        for r in rows:
            counts[r['object_id']] += 1
        (annotations/f'frame_{frame:06d}.json').write_text(json.dumps({
            'frame': frame, 'annotations': [{'object_id': r['object_id'], 'bbox': r['bbox']} for r in rows],
            'provenance': [{'track_id': r['track_id'], 'review_status': r['review_status']} for r in rows]}, indent=1)+'\n')
    meta = {'source_frames': str(a.frames.resolve()), 'labels': str(a.labels.resolve()), 'frames': len(frames),
            'tracks': tracks, 'boxes': sum(len(v) for v in boxes.values()), 'boxes_per_class': dict(counts),
            'review_status_counts': dict(status), 'excluded_status': a.exclude_status,
            'note': 'Participant pseudo-labels, incomplete; a proxy for comparing configurations, not the official score.'}
    (a.output/'run_metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
