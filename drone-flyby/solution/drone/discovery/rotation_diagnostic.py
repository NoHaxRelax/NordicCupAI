"""Known-location diagnostic only: does rotating a view recover missed sprites?"""
import json
import os
import time
from pathlib import Path
from common import ROOT, atomic, read
from offline_proposal_pilot import measure, deduplicate

os.environ['YOLO_CONFIG_DIR'] = '/private/tmp/drone-discovery-audit-config'
os.environ['MPLCONFIGDIR'] = '/private/tmp/drone-discovery-audit-matplotlib'
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import SETTINGS


def main():
    SETTINGS.update({'sync': False})
    torch.set_num_threads(4)
    pilot = read(ROOT / 'artifacts/drone-api-tests/discovery-audit/model-pilot.json')
    model = YOLO(pilot['weights'])
    report = {'method': 'Native 960x540 crop centred on each known seed; rotate by multiples of 45 degrees.',
              'limitation': 'Known locations are supplied. This isolates orientation sensitivity; it does not measure discovery recall.',
              'weights_sha256': pilot['weights_sha256'], 'targets': []}
    start = time.monotonic()
    for frame in pilot['frames']:
        if frame['source'] != 'validation':
            continue
        image = cv2.imread(str(ROOT / f'data/drone/reconstructed-validation/frame_{frame["frame"]:06d}.png'))
        for target in frame['modes']['combined']['matches']:
            b = target['box']
            cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
            # Padding allows a genuinely centred crop for objects near the top.
            crop = cv2.warpAffine(image, np.float32([[1, 0, 480-cx], [0, 1, 270-cy]]),
                                 (960, 540), borderMode=cv2.BORDER_REFLECT_101)
            item = {'frame': frame['frame'], 'seed_id': target['seed_id'], 'class': target['class'], 'angles': []}
            for angle in range(0, 360, 45):
                matrix = cv2.getRotationMatrix2D((480, 270), angle, 1)
                rotated = cv2.warpAffine(crop, matrix, (960, 540), borderMode=cv2.BORDER_REFLECT_101)
                result = model.predict(rotated, imgsz=960, conf=.02, max_det=300,
                                       device=pilot['runtime']['device'], verbose=False, save=False)[0]
                inverse = cv2.invertAffineTransform(matrix)
                predictions = []
                for a, y, c, d, confidence, label in result.boxes.data.cpu().tolist():
                    corners = cv2.transform(np.float32([[[a, y], [c, y], [c, d], [a, d]]]), inverse)[0]
                    minimum, maximum = corners.min(axis=0), corners.max(axis=0)
                    predictions.append({'class': model.names[int(label)], 'confidence': confidence,
                                        'box': [float(minimum[0]+cx-480), float(minimum[1]+cy-270),
                                                float(maximum[0]+cx-480), float(maximum[1]+cy-270)]})
                metrics = measure(predictions, [target])
                item['angles'].append({'angle': angle, 'located': metrics['located'],
                                       'classified': metrics['located_and_classified'],
                                       'best_iou': metrics['matches'][0]['best_iou']})
            report['targets'].append(item)
            print(json.dumps(item), flush=True)
    report['seconds'] = time.monotonic()-start
    report['summary'] = {'targets': len(report['targets']),
                         'unrotated_classified': sum(t['angles'][0]['classified'] for t in report['targets']),
                         'any_rotation_classified': sum(any(a['classified'] for a in t['angles']) for t in report['targets']),
                         'any_rotation_located': sum(any(a['located'] for a in t['angles']) for t in report['targets'])}
    atomic(ROOT / 'artifacts/drone-api-tests/discovery-audit/rotation-diagnostic.json', report)
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
