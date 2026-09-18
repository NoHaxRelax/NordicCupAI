"""Offline blind discovery benchmark. Does not call competition APIs."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time

import cv2

from .build import ROOT, REFERENCE_HOLDOUT
from .detector import TemplateDetector, Settings, iou, nms, sha


def measure(predictions, targets):
    used, matches = set(), []
    for prediction in sorted(predictions, key=lambda p: -p['score']):
        candidates = [(iou(prediction['bbox'], t['bbox']), index) for index, t in enumerate(targets)
                      if index not in used and t['class'] == prediction['class']]
        overlap, index = max(candidates, default=(0., -1))
        if overlap >= .5:
            used.add(index)
            matches.append({'target': index, 'iou': overlap, 'score': prediction['score']})
    return {'targets': len(targets), 'matched': len(used), 'proposals': len(predictions),
            'unmatched_proposals': len(predictions)-len(used), 'matches': matches,
            'per_class': {label: {'targets': sum(t['class'] == label for t in targets),
                                  'matched': sum(targets[i]['class'] == label for i in used)}
                          for label in sorted({t['class'] for t in targets})}}


def validation_truth(root, frames):
    result = {frame: [] for frame in frames}
    hashes = {}
    aliases = json.loads((root/'drone/overnight/config.json').read_text())['track_aliases']
    # Use the corrected completion package, but NEVER its projected annotations.
    for path in sorted((root/'data/drone/training/algorithmic-full-validation').glob('*.json')):
        doc = json.loads(path.read_text())
        relevant = [a for a in doc.get('annotations', []) if a['frame'] in result
                    and a.get('provenance') == 'reviewed_positive'
                    and a.get('review_status', '').startswith('direct')]
        if relevant:
            hashes[str(path.relative_to(root))] = sha(path)
        for a in relevant:
            b = a['bbox_source_xyxy']
            # Fully visible objects only, matching the detector's declared contract.
            if 0 < b[0] < b[2] < 3839 and 0 < b[1] < b[3] < 2159:
                track = doc.get('track_id',path.stem)
                row = {'class': a['class'], 'bbox': b, 'track': aliases.get(track,track)}
                if not any(row['class'] == r['class'] and iou(b, r['bbox']) > .9 for r in result[a['frame']]):
                    result[a['frame']].append(row)
    if not any(result.values()):
        raise ValueError('No corrected reviewed validation labels found')
    return result, hashes


def starts(total, size):
    return sorted(set(list(range(0, total-size+1, size*3//4)) + [total-size]))


def scan(detector, image, zoom, top_band=False, band_height=1080):
    width, height = 3840//2**zoom, 2160//2**zoom
    predictions = []
    for y in starts(band_height if top_band else 2160, height):
        for x in starts(3840, width):
            crop = image[y:y+height, x:x+width]
            if crop.shape[2] == 4 and (crop[:,:,3] != 255).any():
                raise ValueError('Evaluation view contains unobserved transparent pixels')
            view = cv2.resize(crop[:,:,:3], (960, 540), interpolation=cv2.INTER_AREA)
            for row in detector.detect(view, 960/width):
                b = row['bbox']
                predictions.append({**row, 'bbox': [x+b[0]*width/960, y+b[1]*height/540,
                                                   x+b[2]*width/960, y+b[3]*height/540]})
    if hasattr(detector,'suppress'):
        return detector.suppress(predictions)
    return nms(predictions, detector.settings.nms_iou, detector.settings.max_detections)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--engine', choices=['pixels','features','ensemble'], default='pixels')
    p.add_argument('--feature-descriptor', choices=['sift','root'], default='sift')
    p.add_argument('--feature-geometry', choices=['similarity','affine'], default='similarity')
    p.add_argument('--feature-extent', choices=['rectangle','foreground','source_aware'], default='source_aware')
    p.add_argument('--foreground-padding', type=float, default=.1)
    p.add_argument('--feature-matching', choices=['per_template','global','global_multiclass'], default='per_template')
    p.add_argument('--template-scales', type=float, nargs='+', default=[1.])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--validation-frames', type=int, nargs='+', default=[100, 127])
    p.add_argument('--zoom', type=int, choices=(0,1,2), default=0)
    p.add_argument('--threshold', type=float, default=.8)
    p.add_argument('--angles', type=float, nargs='+', default=[-12,0,12])
    p.add_argument('--scales', type=float, nargs='+', default=[.85,1.,1.18])
    p.add_argument('--mask-mode', choices=['context_penalty','masked_ncc','photometric'], default='context_penalty')
    p.add_argument('--classes', nargs='+', help='Optional fixed class subset for a specialist diagnostic')
    p.add_argument('--reference-frames', type=int, nargs='*', default=list(REFERENCE_HOLDOUT))
    p.add_argument('--top-band', action='store_true', help='Search only source y=0..1080, score fully contained targets')
    p.add_argument('--band-height', type=int, choices=[540,1080], default=1080)
    p.add_argument('--reference-only', action='store_true')
    a = p.parse_args()
    cv2.setNumThreads(4)
    if a.engine=='ensemble':
        from .features import FeatureEnsemble,EnsembleSettings
        settings=EnsembleSettings(score_threshold=a.threshold,foreground_padding=a.foreground_padding,matching=a.feature_matching,template_scales=tuple(a.template_scales))
        detector=FeatureEnsemble(a.bank,settings)
    elif a.engine=='features':
        from .features import FeatureDetector, FeatureSettings
        settings=FeatureSettings(score_threshold=a.threshold,descriptor=a.feature_descriptor,geometry=a.feature_geometry,extent=a.feature_extent,foreground_padding=a.foreground_padding,matching=a.feature_matching,template_scales=tuple(a.template_scales))
        detector=FeatureDetector(a.bank,settings)
    else:
        settings = Settings(score_threshold=a.threshold, angles=tuple(a.angles), scales=tuple(a.scales), mask_mode=a.mask_mode)
        detector = TemplateDetector(a.bank, settings)
    if a.classes:
        unknown=set(a.classes)-set(detector.manifest['classes'])
        if unknown:
            raise ValueError('Unknown classes: '+str(unknown))
        detector.templates=[t for t in detector.templates if t[0]['class'] in a.classes]
        if a.engine=='features':
            detector.feature_bank=[t for t in detector.feature_bank if t[0]['class'] in a.classes]
        elif a.engine=='ensemble':
            for matcher in detector.matchers:
                matcher.feature_bank=[t for t in matcher.feature_bank if t[0]['class'] in a.classes]
    if a.top_band and 2160//2**a.zoom>a.band_height:
        raise ValueError('Camera view is taller than the requested band')
    cutoff = detector.manifest.get('validation_train_through')
    if not a.reference_only and cutoff is not None and any(f <= cutoff for f in a.validation_frames):
        raise ValueError('Evaluation frames overlap validation training interval')
    a.output.mkdir(parents=True, exist_ok=False)
    report = {'settings': asdict(settings), 'bank_sha256': sha(a.bank/'manifest.json'), 'zoom': a.zoom,
              'engine':a.engine, 'classes':a.classes, 'top_band':a.top_band, 'band_height':a.band_height,
              'code_hashes':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
              'validation_templates': detector.manifest['validation_templates'], 'results': [],
              'limitations': ['Reference holdouts share instances/background with training.',
                             'Validation labels are participant-reviewed, incomplete positive annotations.',
                             'Unmatched validation proposals are unverified, not established false positives.',
                             'Only fully visible reviewed objects are scored; no competition mAP claim.',
                             'Evaluated validation frames become development data; remaining frames are not accessed.',
                             'Late validation frames 181-249 remain reserved by the default benchmark.']}
    examples = []
    for frame in a.reference_frames:
        if frame not in detector.manifest['reference_holdout_frames']:
            raise ValueError('Reference test frame was not excluded from template construction')
        path = a.root/f'data/drone/reference/helsinki/annotations/frame_{frame:06d}.json'
        doc = json.loads(path.read_text())
        truth = [{'class': r['object_id'], 'bbox': r['bbox']} for r in doc['annotations']
                 if 0 < r['bbox'][0] < r['bbox'][2] < 3839 and 0 < r['bbox'][1] < r['bbox'][3] < 2159]
        examples.append(('reference', frame, a.root/f'data/drone/reference/helsinki/images/frame_{frame:06d}.png', truth))
    if not a.reference_only:
        truth, hashes = validation_truth(a.root, a.validation_frames)
        trained = set(detector.manifest.get('validation_train_tracks', []))
        if any(t['track'] in trained for rows in truth.values() for t in rows):
            raise ValueError('Physical track overlaps bank and evaluation')
        report['validation_annotation_hashes'] = hashes
        for frame in a.validation_frames:
            examples.append(('validation', frame, a.root/f'data/drone/reconstructed-validation/frame_{frame:06d}.png', truth[frame]))
    for split, frame, path, truth in examples:
        if a.classes:
            truth=[t for t in truth if t['class'] in a.classes]
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[:2] != (2160, 3840):
            raise ValueError('Invalid evaluation image '+str(path))
        image_hash = sha(path)
        if image_hash in detector.manifest['source_hashes'].values():
            raise ValueError('Evaluation image duplicates a bank source')
        start = time.perf_counter()
        if a.top_band:
            if a.zoom == 0:
                raise ValueError('Top-band search requires zoom 1 or 2')
            truth = [t for t in truth if t['bbox'][3] < a.band_height]
        predictions = scan(detector, image, a.zoom, top_band=a.top_band, band_height=a.band_height)
        row = dict(split=split, frame=frame, seconds=time.perf_counter()-start, image_sha256=image_hash,
                   **measure(predictions, truth), predictions=predictions, truth=truth)
        report['results'].append(row)
        overlay = cv2.resize(image[:,:,:3], (1920,1080))
        for target in truth:
            x1,y1,x2,y2 = [round(v/2) for v in target['bbox']]
            cv2.rectangle(overlay, (x1,y1), (x2,y2), (255,180,0), 1)
        for pred in predictions:
            x1,y1,x2,y2 = [round(v/2) for v in pred['bbox']]
            cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 1)
            cv2.putText(overlay, f"{pred['class']} {pred['score']:.2f}", (x1,max(12,y1-3)), cv2.FONT_HERSHEY_SIMPLEX, .35, (0,255,0), 1)
        cv2.imwrite(str(a.output/f'{split}-{frame:06d}.jpg'), overlay)
        report['summary'] = {s: dict(Counter({key: sum(r[key] for r in report['results'] if r['split'] == s)
                                            for key in ('targets','matched','proposals','unmatched_proposals')}))
                             for s in sorted({r['split'] for r in report['results']})}
        (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k: row[k] for k in ('split','frame','seconds','targets','matched','proposals')}), flush=True)


if __name__ == '__main__':
    main()
