"""Run the deterministic detector on one delivered camera image."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import cv2

from .detector import Settings, TemplateDetector, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--engine',choices=['pixels','features','ensemble'],default='pixels')
    p.add_argument('--feature-matching',choices=['per_template','global','global_multiclass'],default='global')
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--source-region', type=float, nargs=4, required=True, metavar=('X1','Y1','X2','Y2'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=.8)
    p.add_argument('--scales', type=float, nargs='+', default=[.85,1.,1.18])
    p.add_argument('--angles', type=float, nargs='+', default=[-12.,0.,12.])
    a = p.parse_args()
    cv2.setNumThreads(4)
    image = cv2.imread(str(a.image), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError('Image cannot be read')
    if image.ndim != 3 or image.shape[2] not in (3,4):
        raise ValueError('Expected RGB/RGBA image')
    if image.shape[2] == 4 and (image[:,:,3] != 255).any():
        raise ValueError('Image includes unobserved transparent pixels')
    if a.engine=='ensemble':
        from .features import FeatureEnsemble,EnsembleSettings
        settings=EnsembleSettings(score_threshold=a.threshold,matching=a.feature_matching)
        detector=FeatureEnsemble(a.bank,settings)
    elif a.engine=='features':
        from .features import FeatureDetector,FeatureSettings
        settings=FeatureSettings(score_threshold=a.threshold,matching=a.feature_matching)
        detector=FeatureDetector(a.bank,settings)
    else:
        settings = Settings(score_threshold=a.threshold, scales=tuple(a.scales), angles=tuple(a.angles))
        detector = TemplateDetector(a.bank, settings)
    start = time.perf_counter()
    predictions = detector.predict(image[:,:,:3], a.source_region)
    result = {'predictions': predictions, 'seconds': time.perf_counter()-start,
              'image_sha256': sha(a.image), 'bank_sha256': sha(a.bank/'manifest.json'),
              'source_region': a.source_region, 'settings': asdict(settings)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'detections':len(predictions),'seconds':result['seconds'],'output':str(a.output)}))


if __name__ == '__main__':
    main()
