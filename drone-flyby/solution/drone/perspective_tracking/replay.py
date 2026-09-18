"""Offline reference replay using the runtime predictor and one supplied box.

Only calibration images 0/1 and each object's first complete visible box enter
the tracker. Future labels score predictions; they never refresh tracks.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from . import DeterministicTracker, MotionModel, ViewGeometry, calibrate_images


ROOT = Path(__file__).resolve().parents[2]


def view_at(frame, camera):
    size = (3840, 2160)
    if camera == 'full':
        return ViewGeometry(size, (0, 0, *size), size)
    if camera == 'overview' or frame == 0:
        return ViewGeometry(size, (0, 0, *size), (960, 540))
    x = [0, 960, 1920, 960][(frame-1) % 4]
    return ViewGeometry(size, (x, 0, x+1920, 1080), (960, 540))


def read_view(folder, frame, view):
    image = cv2.imread(str(folder/f'frame_{frame:06d}.png'), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Missing calibration image {frame}')
    x1, y1, x2, y2 = map(int, view.region)
    return cv2.resize(image[y1:y2, x1:x2], view.image_size, interpolation=cv2.INTER_AREA)


def iou(predicted, truth):
    # Match the historical reference benchmark's last-pixel clipping convention.
    limit = [3839., 2159., 3839., 2159.]
    a, b = np.clip(predicted, 0, limit), np.clip(truth, 0, limit)
    overlap = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    union = (a[2:]-a[:2]).prod() + (b[2:]-b[:2]).prod() - overlap
    return float(overlap/union) if union else 0.


def replay(reference, output, camera='full', cached_matches=None):
    output.mkdir(parents=True, exist_ok=True)
    views = [view_at(f, camera) for f in (0, 1)]
    started = time.perf_counter()
    if cached_matches:
        if camera != 'full':
            raise ValueError('Cached full-image calibration cannot stand in for cropped images')
        raw = np.load(cached_matches)
        model = MotionModel.from_matches(raw['x'][raw['train']], raw['y'][raw['train']])
    else:
        images = [read_view(reference/'images', f, views[f]) for f in (0, 1)]
        model = calibrate_images(*images, *views)
    calibration_ms = 1000*(time.perf_counter()-started)
    tracker = DeterministicTracker(model, 'reference-replay')
    rows, responses, runtimes = [], [], []
    seen, anchors = set(), {}
    unmatched = 0
    for path in sorted((reference/'annotations').glob('*.json')):
        annotations = json.loads(path.read_text())
        frame = annotations['frame']
        if frame < 1:
            continue
        view = view_at(frame, camera)
        truth = {a['object_id']: np.array(a['bbox'], float) for a in annotations['annotations']}
        for label, box in truth.items():
            # Supplying exact first boxes isolates the implemented geometry.
            complete = np.all(box[:2] > view.region[:2]) and np.all(box[2:] < view.region[2:])
            complete = complete and np.all(box[2:] < [3839, 2159])
            if label not in seen and complete:
                tracker.add_view_detection(label, label, view.box_from_source(box), frame, view)
                seen.add(label); anchors[label] = frame
        started = time.perf_counter()
        predictions = {p['track_id']: p for p in tracker.predictions(frame)}
        runtimes.append(1000*(time.perf_counter()-started))
        responses.append({'frame': frame, 'predictions': list(predictions.values())})
        unmatched += sum(label not in truth for label in predictions)
        for label, box in truth.items():
            if label not in seen or frame <= anchors[label]:
                continue
            prediction = predictions.get(label)
            overlap = iou(prediction['bbox_unclipped_xyxy'], box) if prediction else 0.
            rows.append({'frame': frame, 'track_id': label, 'anchor_frame': anchors[label],
                         'prediction': prediction, 'truth_source_xyxy': box.tolist(), 'iou': overlap})
    evaluated = sorted(set(row['track_id'] for row in rows))
    summary = {'camera': camera, 'calibration': 'cached initial-pair matches' if cached_matches else 'actual initial-pair images',
               'initial_boxes': 'one exact complete visible reference box per object; no later updates',
               'added_noise': False, 'future_images_used': False, 'anchors': anchors,
               'calibration_ms': calibration_ms, 'calibration_diagnostics': model.diagnostics,
               'future_boxes': len(rows), 'passing_iou50': sum(r['iou'] >= .5 for r in rows),
               'future_iou50_fraction': float(np.mean([r['iou'] >= .5 for r in rows])) if rows else None,
               'evaluated_tracks': len(evaluated),
               'whole_track_passes': sum(all(r['iou'] >= .5 for r in rows if r['track_id'] == label) for label in evaluated),
               'predictions_without_matching_label': unmatched,
               'prediction_ms_median': float(np.median(runtimes)),
               'prediction_ms_p90': float(np.percentile(runtimes, 90)),
               'limitations': ['Known starting boxes and identities; not detector/classification accuracy.',
                               'Twenty-five-frame recording; some object lifetimes extend beyond it.',
                               'Camera policies change visibility and scoring cohorts; percentages are not paired comparisons.',
                               'IoU50 diagnostic, not competition mAP; unmatched predictions counted separately.']}
    (output/'model.json').write_text(json.dumps(model.to_dict(), indent=2, allow_nan=False)+'\n')
    (output/'tracker-state.json').write_text(json.dumps(tracker.to_dict(), indent=2, allow_nan=False)+'\n')
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    for filename, values in (('scored-predictions.jsonl', rows), ('runtime-predictions.jsonl', responses)):
        (output/filename).write_text(''.join(json.dumps(value, allow_nan=False)+'\n' for value in values))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT/'data/drone/reference/helsinki')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--camera', choices=('full', 'overview', 'sweep'), default='full')
    parser.add_argument('--cached-matches', type=Path, help='Only for reproducing the historical full-image calibration')
    args = parser.parse_args()
    replay(args.reference, args.output, args.camera, args.cached_matches)


if __name__ == '__main__':
    main()
