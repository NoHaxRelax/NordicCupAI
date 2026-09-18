"""Cut verifier crops from synthetic sprite composites (training-split sprites on training-split empty tiles).

Positives: every pasted sprite's box, jittered slightly (the verifier sees expert boxes, not exact ones).
Negatives: random windows on the same composites that overlap no box (IoU < .02), sized like real
candidates. Output matches crops.py so train_verifier.py can concatenate them.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .common import iou, sha
from .crops import crop


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--size', type=int, default=96)
    p.add_argument('--negatives-per-image', type=int, default=2)
    p.add_argument('--seed', type=int, default=1731)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    m = json.loads((a.synthetic / 'manifest.json').read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    images, labels, meta = [], [], []
    for rec in m['records']:
        if rec.get('background_split', 'train') != 'train':
            continue
        image = cv2.imread(str(a.synthetic / rec['file']))
        if image is None:
            continue
        H, W = image.shape[:2]
        boxes = []
        for ann in rec['annotations']:
            x1, y1, x2, y2 = ann['bbox_xyxy']
            w, h = x2 - x1, y2 - y1
            jitter = rng.uniform(-.08, .08, 4) * [w, h, w, h]
            box = [x1 + jitter[0], y1 + jitter[1], x2 + jitter[2], y2 + jitter[3]]
            images.append(crop(image, box, a.size)); labels.append(ann['class_name']); boxes.append(ann['bbox_xyxy'])
            meta.append(dict(tile=rec['id'], label=ann['class_name'], bbox=box, zoom=rec['zoom'], synthetic=True))
        for _ in range(a.negatives_per_image):
            for _ in range(20):
                side = float(rng.uniform(16, 90))
                x, y = rng.uniform(0, W - side), rng.uniform(0, H - side)
                box = [x, y, x + side, y + side]
                if all(iou(box, b) < .02 for b in boxes):
                    images.append(crop(image, box, a.size)); labels.append('background')
                    meta.append(dict(tile=rec['id'], label='background', bbox=box, zoom=rec['zoom'], synthetic=True))
                    break
    np.savez_compressed(a.output / 'crops.npz', images=np.stack(images), labels=np.array(labels))
    doc = dict(format='expert-crops-v1', size=a.size, classes=sorted(set(labels)), count=len(labels), by_label=dict(Counter(labels)),
               synthetic_manifest_sha256=sha(a.synthetic / 'manifest.json'), crops_sha256=sha(a.output / 'crops.npz'), records=meta, policy='training-split synthetic only')
    (a.output / 'manifest.json').write_text(json.dumps(doc, indent=1))
    print(json.dumps(dict(count=len(labels), by_label=doc['by_label'])))


if __name__ == '__main__':
    main()
