"""Bounded local discovery pilot. Never submits to the competition API.

Reference results measure fit to known training instances. Validation results
measure agreement with sparse score-confirmed boxes, not scene-wide recall.
"""
import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from common import ROOT, W, H, atomic, iou, now, read


def starts(total, extent):
    step = round(extent * .75)
    return sorted(set(list(range(0, total - extent + 1, step)) + [total - extent]))


def windows(mode):
    if mode == 'full':
        return [(0, 0, W, H)]
    w, h = (1920, 1080) if mode == 'L1' else (960, 540)
    return [(x, y, x + w, y + h) for y in starts(H, h) for x in starts(W, w)]


def deduplicate(rows):
    # Cross-tile duplicate suppression only. Keep alternate class hypotheses.
    kept = []
    for row in sorted(rows, key=lambda r: r['confidence'], reverse=True):
        if not any(row['class'] == k['class'] and iou(row['box'], k['box']) > .5 for k in kept):
            kept.append(row)
    return kept


def measure(predictions, targets):
    matches = []
    for target in targets:
        spatial = [p for p in predictions if iou(p['box'], target['box']) >= .5]
        correct = [p for p in spatial if p['class'] == target['class']]
        matches.append({**target, 'located': bool(spatial), 'located_and_classified': bool(correct),
                        'best_iou': max((iou(p['box'], target['box']) for p in predictions), default=0),
                        'matching_classes': sorted({p['class'] for p in spatial})})
    return {'targets': len(targets), 'located': sum(m['located'] for m in matches),
            'located_and_classified': sum(m['located_and_classified'] for m in matches),
            'proposals': len(predictions), 'matches': matches}


def inputs():
    examples = []
    for frame in (5, 12, 20):
        ann = read(ROOT / f'data/drone/reference/helsinki/annotations/frame_{frame:06d}.json')
        examples.append(('reference', frame, ROOT / f'data/drone/reference/helsinki/images/frame_{frame:06d}.png',
                         [{'class': a['object_id'], 'box': a['bbox']} for a in ann['annotations']]))
    state = read(ROOT / 'data/drone/discovery/state.json')
    for frame in (5, 45, 66, 140):
        targets = [{'class': s['class'], 'box': s['bbox_source_xyxy'], 'seed_id': s['seed_id']}
                   for s in state['seeds'] if s['frame'] == frame]
        examples.append(('validation', frame, ROOT / f'data/drone/reconstructed-validation/frame_{frame:06d}.png', targets))
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)  # Do not allow automatic downloads.
    os.environ.setdefault('YOLO_CONFIG_DIR', '/private/tmp/drone-discovery-audit-config')
    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/drone-discovery-audit-matplotlib')
    Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
    Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS
    SETTINGS.update({'sync': False})
    torch.set_num_threads(4)
    device = ('mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device
    model = YOLO(str(args.weights))
    report = {
        'created_at': now(), 'status': 'running', 'weights': str(args.weights.resolve()),
        'weights_sha256': hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        'runtime': {'torch': torch.__version__, 'ultralytics': ultralytics.__version__, 'device': device},
        'checkpoint_training_metrics': model.ckpt.get('train_metrics'),
        'checkpoint_epochs_in_results': len(model.ckpt.get('train_results', {}).get('epoch', [])),
        'settings': {'confidence': .02, 'input_size': 960, 'max_detections_per_view': 300,
                     'overlap': .25, 'tile_counts': {mode: len(windows(mode)) for mode in ('full', 'L1', 'L2')}},
        'limitations': [
            'Reference frames were used in training: these are fit diagnostics, not independent accuracy.',
            'Validation targets are sparse score-confirmed boxes with IoU >= .5 to hidden labels, not exact ground truth.',
            'Unmatched validation proposals are unreviewed; precision and full-scene recall cannot be measured here.',
            'No proposal from this pilot is a verified annotation or automatically added to training.'
        ], 'frames': []}
    started = time.monotonic()
    for source, frame, path, targets in inputs():
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != (H, W):
            raise ValueError(f'Expected complete 4K image: {path}')
        entry = {'source': source, 'frame': frame, 'modes': {}, 'predictions': {}}
        combined = []
        for mode in ('full', 'L1', 'L2'):
            mode_start = time.monotonic()
            predictions = []
            for x1, y1, x2, y2 in windows(mode):
                crop = image[y1:y2, x1:x2]
                # The camera always delivers 960x540, including lower zooms.
                delivered = cv2.resize(crop, (960, 540), interpolation=cv2.INTER_AREA)
                result = model.predict(delivered, imgsz=960, conf=.02, max_det=300,
                                       device=device, verbose=False, save=False)[0]
                for box in result.boxes.data.cpu().tolist():
                    a, b, c, d, confidence, label = box
                    predictions.append({'class': model.names[int(label)], 'confidence': confidence,
                                        'box': [x1 + a * (x2-x1)/960, y1 + b * (y2-y1)/540,
                                                x1 + c * (x2-x1)/960, y1 + d * (y2-y1)/540],
                                        'view': [x1, y1, x2, y2], 'zoom': mode})
            predictions = deduplicate(predictions)
            combined.extend(predictions)
            entry['modes'][mode] = {**measure(predictions, targets), 'seconds': time.monotonic()-mode_start}
            entry['predictions'][mode] = predictions
            print(json.dumps({'source': source, 'frame': frame, 'mode': mode,
                              **{k: v for k, v in entry['modes'][mode].items() if k != 'matches'}}), flush=True)
        combined = deduplicate(combined)
        entry['modes']['combined'] = measure(combined, targets)
        entry['predictions']['combined'] = combined
        report['frames'].append(entry)
        report['elapsed_seconds'] = time.monotonic()-started
        atomic(args.output, report)
    report['summary'] = {}
    for source in ('reference', 'validation'):
        report['summary'][source] = {}
        for mode in ('full', 'L1', 'L2', 'combined'):
            counter = Counter()
            for entry in report['frames']:
                if entry['source'] == source:
                    counter.update({k: entry['modes'][mode][k] for k in ('targets', 'located', 'located_and_classified', 'proposals')})
            report['summary'][source][mode] = dict(counter)
    report['status'] = 'complete'
    atomic(args.output, report)
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
