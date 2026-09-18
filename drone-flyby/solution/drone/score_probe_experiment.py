#!/usr/bin/env python3
"""Offline proof of class-group score probes using the public Helsinki labels.

This supplies a known candidate box, then infers its label from aggregate scores.
It does not discover candidate boxes or send requests to the live portal.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'artifacts/drone-source-2026-09-17'))
import local_evaluator
from dtos import OBJECT_CLASSES


def main():
    annotations = {
        int(p.stem.split('_')[-1]): json.loads(p.read_text())['annotations']
        for p in (ROOT / 'data/drone/reference/helsinki/annotations').glob('*.json')
    }
    # The scorer only needs annotations. Include all 25 annotated frames without
    # requiring all 25 image files to be downloaded for this score-only test.
    local_evaluator.frame_numbers = lambda scene: sorted(annotations)
    local_evaluator.load_annotations = lambda frame, scene: annotations[frame]
    candidate = annotations[0][0]
    box = candidate['bbox']

    def probe(labels, bbox=box):
        predictions = {0: [{'object_id': label, 'bbox': bbox, 'confidence': 1.0} for label in labels]}
        with redirect_stdout(io.StringIO()):
            value, _ = local_evaluator.score('offline-reference', predictions)
        return value

    baseline = probe([])
    positive = probe(OBJECT_CLASSES)
    bits = []
    for bit in range(4):
        labels = [name for index, name in enumerate(OBJECT_CLASSES) if index & (1 << bit)]
        value = probe(labels)
        bits.append({'bit': bit, 'labels_tested': labels, 'score': value, 'positive': value > baseline})
    inferred_index = sum((1 << row['bit']) for row in bits if row['positive'])
    inferred = OBJECT_CLASSES[inferred_index]
    x1, y1, x2, y2 = box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    geometry = []
    for scale in [0.6, 0.8, 1.0, 1.2, 1.5]:
        w, h = (x2-x1)*scale, (y2-y1)*scale
        trial = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
        geometry.append({'scale_about_center': scale, 'iou': min(scale**2, 1/scale**2),
                         'score': probe([inferred], trial)})
    result = {'dataset': 'public Helsinki reference, NOT validation',
              'candidate_box_source': 'known reference ground truth for this controlled experiment',
              'frame': 0, 'bbox_source_xyxy': box, 'baseline': baseline,
              'all_labels_control': positive, 'class_group_queries': bits,
              'inferred_label': inferred, 'actual_label': candidate['object_id'],
              'geometry_probes': geometry,
              'live_queries_used': 0}
    assert positive > 0 and inferred == candidate['object_id']
    assert geometry[0]['score'] == geometry[4]['score'] == 0
    assert geometry[1]['score'] == geometry[2]['score'] == geometry[3]['score'] > 0
    destination = ROOT / 'artifacts/drone-api-tests/offline-score-probe.json'
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
