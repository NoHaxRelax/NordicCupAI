"""Harvest verifier training crops from expert candidates on TRAINING tiles.

Positives: candidates whose box overlaps a training label of the expert's class (IoU >= .3) -> that class.
Negatives: candidates on verified-empty training tiles, and candidates on labelled tiles that overlap no
label of any class (IoU < .05 with every label) -> 'background'. Candidates that overlap another class's
label are labelled with that class (the verifier must also separate classes). Crops are square windows
around the candidate box with 25% context, resized to 96 px, saved with their source ids and hashes.
Only records with split == 'train' are read. Dev and reserved tracks never enter.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .common import iou, sha


def crop(image, box, size=96, context=.25):
    x1, y1, x2, y2 = box
    w, h = max(4., x2 - x1), max(4., y2 - y1)
    side = max(w, h) * (1 + 2 * context)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    X1, Y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    X2, Y2 = int(round(cx + side / 2)), int(round(cy + side / 2))
    H, W = image.shape[:2]
    pad = [max(0, -Y1), max(0, Y2 - H), max(0, -X1), max(0, X2 - W)]
    patch = image[max(0, Y1):min(H, Y2), max(0, X1):min(W, X2)]
    if any(pad):
        patch = cv2.copyMakeBorder(patch, *pad, cv2.BORDER_REFLECT)
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA if patch.shape[0] > size else cv2.INTER_LINEAR)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True, help='directory holding one evaluate.py output dir per class')
    p.add_argument('--grid', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--size', type=int, default=96)
    a = p.parse_args()
    manifest = json.loads((a.grid / 'manifest.json').read_text())
    records = {r['id']: r for r in manifest['records']}
    a.output.mkdir(parents=True, exist_ok=False)
    images, labels, meta = [], [], []
    counts = Counter()
    for report_path in sorted(a.runs.glob('*/report.json')):
        report = json.loads(report_path.read_text())
        expert_class = report['class_name']
        for row in report['crops'] + [dict(id=e['id'], zoom=e['zoom'], candidates=e.get('candidates', []), empty=True) for e in report.get('empty', [])]:
            rec = records.get(row['id'])
            if rec is None or rec['split'] != 'train':
                continue
            image = None
            all_labels = [(x['class_name'], x['bbox_xyxy']) for x in rec['annotations']] if rec['kind'] == 'positive' else []
            for cand in row.get('candidates', []):
                if cand.get('rejected_by') is not None or 'bbox' not in cand:
                    continue
                overlaps = [(iou(cand['bbox'], b), c) for c, b in all_labels]
                best = max(overlaps, default=(0., None))
                if best[0] >= .3:
                    label = best[1]
                elif best[0] < .05:
                    label = 'background'
                else:
                    continue  # ambiguous overlap: skip
                if image is None:
                    image = cv2.imread(str(a.grid / rec['file']))
                images.append(crop(image, cand['bbox'], a.size))
                labels.append(label)
                counts[(expert_class, label)] += 1
                meta.append(dict(tile=row['id'], expert=expert_class, label=label, bbox=cand['bbox'], iou=best[0], zoom=rec['zoom'],
                                 scores={k: v for k, v in cand.items() if isinstance(v, (int, float)) and k not in ('cx', 'cy')}))
    if not images:
        raise SystemExit('No crops harvested')
    np.savez_compressed(a.output / 'crops.npz', images=np.stack(images), labels=np.array(labels))
    classes = sorted(set(labels))
    doc = dict(format='expert-crops-v1', size=a.size, classes=classes, count=len(labels), by_label=dict(Counter(labels)),
               by_expert_label={f'{e}:{l}': n for (e, l), n in sorted(counts.items())}, grid_manifest_sha256=sha(a.grid / 'manifest.json'),
               crops_sha256=sha(a.output / 'crops.npz'), records=meta, policy='training split only')
    (a.output / 'manifest.json').write_text(json.dumps(doc, indent=1))
    print(json.dumps(dict(count=len(labels), by_label=doc['by_label'])))


if __name__ == '__main__':
    main()
