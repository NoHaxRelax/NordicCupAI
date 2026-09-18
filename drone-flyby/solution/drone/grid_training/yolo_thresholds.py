"""Development-only confidence and localization diagnosis for a saved detector."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    u, v = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, u-x) * max(0, v-y)
    return intersection / max(1e-6, (a[2]-a[0])*(a[3]-a[1]) +
                              (b[2]-b[0])*(b[3]-b[1]) - intersection)


def score(records, predictions, threshold, require_class):
    groups, negatives = defaultdict(list), []
    for row, detections in zip(records, predictions):
        detections = [p for p in detections if p['score'] >= threshold]
        if row['kind'] == 'background':
            negatives.append(bool(detections))
        used = set()
        for ann in row['annotations']:
            if not ann['fully_contained']:
                continue
            candidates = [(iou(ann['bbox_xyxy'], pred['box']), j)
                          for j, pred in enumerate(detections) if j not in used
                          and (not require_class or pred['class_id'] == ann['class_id'])]
            overlap, index = max(candidates, default=(0, -1))
            found = overlap >= .5
            if found:
                used.add(index)
            groups[ann['class_name'], row['zoom'], ann['group']].append(found)
    per_class = defaultdict(list)
    for (name, zoom, group), values in groups.items():
        per_class[name].append(sum(values)/len(values))
    return {'recall_per_class': {k: sum(v)/len(v) for k,v in per_class.items()},
            'background_false_positive_rate': sum(negatives)/len(negatives)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from ultralytics import YOLO
    from .train import sha, write
    if args.output.exists():
        raise ValueError('Refusing to overwrite diagnostics')
    manifest = json.loads((args.data/'manifest.json').read_text())
    rows = [r for r in manifest['records'] if r['split'] == 'dev']
    model = YOLO(str(args.checkpoint))
    predictions = []
    for row in rows:
        assert sha(args.data/row['file']) == row['sha256']
        result = model.predict(str(args.data/row['file']), imgsz=row['input_size'],
                               conf=.001, device=0, verbose=False)[0]
        predictions.append([{'box': b, 'class_id': int(c), 'score': float(s)}
                            for b,c,s in zip(result.boxes.xyxy.cpu().tolist(),
                                             result.boxes.cls.cpu().tolist(),
                                             result.boxes.conf.cpu().tolist())])
    write(args.output, {'checkpoint_sha256': sha(args.checkpoint),
          'manifest_sha256': sha(args.data/'manifest.json'),
          'limitation': 'Reused partial development labels; not independent threshold selection or mAP.',
          'thresholds': [{'threshold': t, 'class_and_box': score(rows,predictions,t,True),
                          'box_only': score(rows,predictions,t,False)}
                         for t in [.001,.003,.01,.03,.1,.25,.5]],
          'predictions': [{'id': r['id'], 'detections': p} for r,p in zip(rows,predictions)]})


if __name__ == '__main__':
    main()
