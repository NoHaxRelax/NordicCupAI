"""Develop on reference objects, then audit a separate pseudo-labelled flight.

No validation labels enter fitting, hyperparameter selection or pose extraction.
Validation boxes are tracker-assisted proposals, not official scoring truth.
"""
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .conditioned_shape import ConditionedShapeModel, axis_from_crop, features, fit_residual_factors
from .motion import MotionModel
from .replay import iou
from .shape_study import full


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/drone-conditioned-shape'
STEP_M = 13.8888889


@lru_cache(maxsize=4)
def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Cannot read image {path}')
    return image


def pose(image_root, frame, box):
    image = read_image(str(image_root/f'frame_{frame:06d}.png'))
    x1, y1 = np.floor(np.maximum(box[:2], 0)).astype(int)
    x2, y2 = np.ceil(np.minimum(box[2:], [3840, 2160])).astype(int)
    return axis_from_crop(image[y1:y2, x1:x2])


def reference_tracks():
    tracks = {}
    for path in sorted((ROOT/'data/drone/reference/helsinki/annotations').glob('*.json')):
        data = json.loads(path.read_text())
        for row in data['annotations']:
            name = row['object_id']
            tracks.setdefault(name, {})[data['frame']] = np.array(row['bbox'], float)
    return tracks


def make_samples(tracks, model):
    samples, failures = [], []
    image_root = ROOT/'data/drone/reference/helsinki/images'
    for label, track in tracks.items():
        frames = [f for f, box in track.items() if f >= 1 and full(box)]
        # Local windows supply contexts; one physical track still has total
        # training weight one, regardless of how many overlapping windows exist.
        for start in range(0, len(frames)-4, 3):
            window = frames[start:start+6]
            try:
                factors = fit_residual_factors(model, [(f, track[f]) for f in window])
                anchor = window[0]
                angle, score = pose(image_root, anchor, track[anchor])
                even, odd = features(model, track[anchor], anchor, angle, score)
                samples.append({'instance_id': 'reference:'+label, 'sequence_id': 'reference',
                    'label': label, 'anchor': anchor, 'training_frames': window,
                    'even': even.tolist(), 'odd': odd.tolist(), 'factors': factors,
                    'orientation': angle, 'axis_score': score})
            except ValueError as exc:
                failures.append({'class': label, 'frames': window, 'error': str(exc)})
    return samples, failures


def evaluate(bank, model, track, label, instance_id, sequence, image_root, clock):
    frames = [f for f, box in track.items() if full(box) and clock[f] >= model.calibrated_until]
    if not frames:
        return []
    anchor = frames[0]
    angle, score = pose(image_root, anchor, track[anchor])
    rows = []
    for frame, truth in sorted(track.items()):
        if frame <= anchor:
            continue
        elapsed = clock[frame]-clock[anchor]
        common = {'class': label, 'instance_id': instance_id, 'sequence': sequence,
                  'frame': frame, 'anchor': anchor, 'orientation': angle, 'axis_score': score}
        baseline = model.box(track[anchor], clock[anchor], clock[frame])
        for method in ('shared_plane', 'conditioned_candidate', 'guarded'):
            if method == 'shared_plane':
                predicted, reason = baseline, None
            else:
                result = bank.predict(label, model, track[anchor], clock[anchor],
                    distance_m=elapsed*STEP_M, orientation=angle, axis_score=score,
                    diagnostic=method == 'conditioned_candidate')
                predicted, reason = result['box'], result['reason']
            rows.append({**common, 'method': method, 'prediction': predicted.tolist(),
                         'truth': truth.tolist(), 'iou': iou(predicted, truth), 'fallback_reason': reason})
    return rows


def summary(rows):
    identities = sorted({r['instance_id'] for r in rows})
    return {'frames': len(rows), 'instances': len(identities),
            'passing_iou50': sum(r['iou'] >= .5 for r in rows),
            'whole_tracks': sum(all(r['iou'] >= .5 for r in rows if r['instance_id'] == name) for name in identities),
            'mean_iou': float(np.mean([r['iou'] for r in rows])) if rows else None,
            'macro_mean_iou': float(np.mean([np.mean([r['iou'] for r in rows if r['instance_id'] == name]) for name in identities])) if rows else None,
            'minimum_iou': min((r['iou'] for r in rows), default=None)}


def validation_tracks():
    tracks, labels = {}, {}
    for path in sorted((ROOT/'data/drone/training/manual-validation').glob('*crops/manifest.json')):
        name = path.parent.name.removesuffix('-crops').replace('-early', '')
        if name == 'crops':
            name = 'large-launcher-early'
        for row in json.loads(path.read_text()).get('records', []):
            labels[name] = row['class']
            tracks.setdefault(name, {}).setdefault(row['frame'], np.array(row['object_bbox_source_xyxy'], float))
    return tracks, labels


def validation_model():
    path = OUT/'validation-calibration.json'
    if path.exists():
        return MotionModel.from_dict(json.loads(path.read_text()))
    images = [read_image(str(ROOT/f'data/drone/reconstructed-validation/frame_{f:06d}.png')) for f in (5, 6)]
    sift = cv2.SIFT_create(nfeatures=6500)
    features_and_desc = [sift.detectAndCompute(image, None) for image in images]
    (k0, d0), (k1, d1) = features_and_desc
    matcher = cv2.BFMatcher()
    reverse = {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(d1, d0, k=2)
               if len(pair) == 2 for m, n in [pair] if m.distance < .7*n.distance}
    matches = [m for pair in matcher.knnMatch(d0, d1, k=2) if len(pair) == 2
               for m, n in [pair] if m.distance < .7*n.distance and reverse.get(m.trainIdx) == m.queryIdx]
    x = np.array([k0[m.queryIdx].pt for m in matches]); y = np.array([k1[m.trainIdx].pt for m in matches])
    h, _ = cv2.findHomography(x, y, cv2.RANSAC, 6.)
    if h is None:
        raise ValueError('Validation first-pair calibration failed')
    hp = np.c_[x, np.ones(len(x))]@h.T
    keep = np.linalg.norm(hp[:, :2]/hp[:, 2, None]-y, axis=1) < 40.
    model = MotionModel.from_matches(x[keep], y[keep])
    path.write_text(json.dumps(model.to_dict(), indent=2, allow_nan=False)+'\n')
    return model


def symmetry_audit(model):
    w, h = model.source_size
    points = np.array([[x, y] for x in np.linspace(100, w-100, 9) for y in np.linspace(100, h-100, 5)])
    reflected = points.copy(); reflected[:, 0] = w-reflected[:, 0]
    result = {}
    for ticks in (1, 10, 20):
        first = model.points(points, 1, 1+ticks); first[:, 0] = w-first[:, 0]
        second = model.points(reflected, 1, 1+ticks)
        errors = np.linalg.norm(first-second, axis=1)
        result[str(ticks)] = {'median_source_px': float(np.median(errors)), 'max_source_px': float(max(errors))}
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = MotionModel.from_dict(json.loads((ROOT/'artifacts/drone-runtime-tracker/diagnosis/native-calibration.json').read_text())['model'])
    tracks = reference_tracks()
    samples, failures = make_samples(tracks, model)
    (OUT/'training-samples.json').write_text(json.dumps({'samples': samples, 'failures': failures}, indent=2, allow_nan=False)+'\n')
    reference_clock = {f: float(f) for f in range(25)}
    cv_results, cv_rows = {}, []
    for ridge in (.1, 1., 10.):
        rows = []
        for label, track in tracks.items():
            training = [s for s in samples if s['instance_id'] != 'reference:'+label]
            bank = ConditionedShapeModel.fit(training, reference_step_m=STEP_M, ridge=ridge)
            if 'reference:'+label in bank.training['instances']:
                raise ValueError('Held-out object leaked into model')
            rows += evaluate(bank, model, track, label, 'reference:'+label, 'reference',
                             ROOT/'data/drone/reference/helsinki/images', reference_clock)
        cv_results[str(ridge)] = {method: summary([r for r in rows if r['method'] == method])
                                 for method in ('shared_plane', 'conditioned_candidate', 'guarded')}
        cv_rows += [{**row, 'ridge': ridge} for row in rows]
        print('reference CV', ridge, cv_results[str(ridge)], flush=True)
    # Select using reference only; do not revisit after seeing validation.
    selected = max(cv_results, key=lambda k: cv_results[k]['conditioned_candidate']['macro_mean_iou'])
    bank = ConditionedShapeModel.fit(samples, reference_step_m=STEP_M, ridge=float(selected))
    input_path = OUT/'validation-inputs.json'
    if input_path.exists():
        # Freeze this experiment's audit inputs; other annotation work may
        # continue in the workspace without silently changing the test set.
        snapshot = json.loads(input_path.read_text())['tracks']
        validation = {name: {int(f): np.array(b, float) for f, b in t['boxes'].items()}
                      for name, t in snapshot.items()}
        labels = {name: t['label'] for name, t in snapshot.items()}
    else:
        validation, labels = validation_tracks()
        input_path.write_text(json.dumps({
            'label_source': 'tracker-assisted participant pseudo-labels, not organizer truth',
            'tracks': {name: {'label': labels[name], 'boxes': {str(f): b.tolist() for f, b in track.items()}}
                       for name, track in validation.items()}}, indent=2, allow_nan=False)+'\n')
    vm = validation_model()
    timing = json.loads((ROOT/'artifacts/drone-camera-pixels/validation-clocks.json').read_text())['rows']['L0_full']
    clock = {5: 0.}
    for transition in timing:
        if transition['motion_ticks'] is None:
            raise ValueError('Unknown motion time in validation audit')
        clock[transition['to']] = clock[transition['from']]+transition['motion_ticks']
    validation_rows, unscored = [], []
    for name, track in validation.items():
        rows = evaluate(bank, vm, track, labels[name], 'validation:'+name, 'validation',
                        ROOT/'data/drone/reconstructed-validation', clock)
        validation_rows += rows
        if not rows:
            unscored.append(name)
    records = []
    for identity in sorted({r['instance_id'] for r in validation_rows}):
        candidate = [r for r in validation_rows if r['instance_id'] == identity and r['method'] == 'conditioned_candidate']
        baseline = [r for r in validation_rows if r['instance_id'] == identity and r['method'] == 'shared_plane']
        c, b = summary(candidate), summary(baseline)
        records.append({'instance_id': identity, 'sequence_id': 'validation', 'label': candidate[0]['class'],
            'label_source': 'tracker_assisted_pseudo_labels', 'frames': c['frames'],
            'candidate_passes': c['passing_iou50'], 'candidate_min_iou': c['minimum_iou'],
            'candidate_mean_iou': c['mean_iou'], 'baseline_mean_iou': b['mean_iou']})
    bank, reasons = bank.qualify(records)
    report = {'metadata': {
        'training': 'Official reference labels only. Whole physical objects excluded for reference model selection.',
        'validation': 'Separate flight, approximate tracker-assisted pseudo-labels. Not organizer accuracy or mAP.',
        'motion_clock': 'Validation uses measured background timing to isolate shape; future background timing is supplied, not a blind frame-clock forecast.',
        'pose': 'Single initial crop edge-spatial principal axis modulo pi, anisotropy weighted. Not recovered 3D heading.',
        'no_artificial_noise': True, 'initial_boxes': 'Known source-pixel boxes, no detector accuracy claim.',
        'position_symmetry': 'Residual equivariance is enforced under reflecting camera, box and axis together. Physical asset symmetry is not established.',
        'selection': 'Three fixed ridge values chosen on reference object-held-out macro mean IoU before scoring validation. Development audit, not a final untouched test.'},
        'training_samples': len(samples), 'training_instances': bank.instance_counts,
        'training_fit_failures': failures, 'selected_ridge': selected, 'reference_model_selection': cv_results,
        'recommended_model': 'shared_plane' if cv_results[selected]['conditioned_candidate']['macro_mean_iou'] <= cv_results[selected]['shared_plane']['macro_mean_iou'] else 'candidate_requires_independent_qualification',
        'validation_pseudo_label_audit': {method: summary([r for r in validation_rows if r['method'] == method])
                                        for method in ('shared_plane', 'conditioned_candidate', 'guarded')},
        'validation_instances': records, 'unscored_validation_instances': unscored,
        'symmetry': {'reference': symmetry_audit(model), 'validation': symmetry_audit(vm)},
        'validated_classes': list(bank.validated_classes), 'qualification_rejections': reasons}
    (OUT/'model.json').write_text(json.dumps(bank.to_dict(), indent=2, allow_nan=False)+'\n')
    (OUT/'measurements.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for name, rows in [('reference-model-selection', cv_rows), ('validation-diagnostic', validation_rows)]:
        (OUT/f'{name}.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    print(json.dumps({k: report[k] for k in ('training_samples', 'selected_ridge', 'validation_pseudo_label_audit', 'symmetry', 'validated_classes')}, indent=2))


if __name__ == '__main__':
    main()
