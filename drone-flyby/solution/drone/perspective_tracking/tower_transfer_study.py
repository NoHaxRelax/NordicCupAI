"""Same-class transfer from reference to independently reannotated validation.

The validation boxes are assistant visual annotations, not organizer labels.
They are frozen before forecast scoring. No validation target enters fitting.
"""
import json
from pathlib import Path

import numpy as np

from .conditioned_shape import ConditionedShapeModel, fit_residual_factors
from .conditioned_study import pose, reference_tracks
from .edge_profile import EdgeMotionProfile
from .motion import MotionModel
from .replay import iou
from .shape_study import full, forecast


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/drone-transfer-review'
STEP_M = 13.8888889


def main():
    ref_model = MotionModel.from_dict(json.loads((ROOT/'artifacts/drone-runtime-tracker/diagnosis/native-calibration.json').read_text())['model'])
    val_model = MotionModel.from_dict(json.loads((ROOT/'artifacts/drone-conditioned-shape/validation-calibration.json').read_text()))
    reference = reference_tracks()['large_tower']
    training = [(f, b) for f, b in reference.items() if full(b)]
    direct = EdgeMotionProfile.fit(ref_model, {'reference:large_tower': training},
                                   label='large_tower', reference_step_m=STEP_M)
    residual = EdgeMotionProfile('large_tower', tuple(fit_residual_factors(ref_model, training)), STEP_M,
                                  {'instance_id': 'reference:large_tower', 'frames': [f for f, b in training]})
    bank = ConditionedShapeModel.from_dict(json.loads((ROOT/'artifacts/drone-conditioned-shape/model.json').read_text()))
    annotations = json.loads((OUT/'independent-tower-boxes.json').read_text())
    observations = [(r['frame'], np.array(r['bbox_source_xyxy'], float)) for r in annotations['observations']]
    anchor_frame, anchor_box = observations[0]
    clock = {5: 0.}
    for row in json.loads((ROOT/'artifacts/drone-camera-pixels/validation-clocks.json').read_text())['rows']['L0_full']:
        clock[row['to']] = clock[row['from']]+row['motion_ticks']
    angle, score = pose(ROOT/'data/drone/reconstructed-validation', anchor_frame, anchor_box)
    rows = []
    for frame, truth in observations[1:]:
        start, end = clock[anchor_frame], clock[frame]
        distance = (end-start)*STEP_M
        baseline = val_model.box(anchor_box, start, end)
        base_midpoints = EdgeMotionProfile('nominal', (1.,)*4, STEP_M).predict(val_model, anchor_box, start, distance_m=distance)
        predictions = {
            'shared_plane': baseline,
            'same_class_direct': direct.predict(val_model, anchor_box, start, distance_m=distance),
            'same_class_residual': baseline+(residual.predict(val_model, anchor_box, start, distance_m=distance)-base_midpoints),
            'conditioned_candidate': bank.predict('large_tower', val_model, anchor_box, start,
                distance_m=distance, orientation=angle, axis_score=score, diagnostic=True)['box']}
        for method, pred in predictions.items():
            rows.append({'frame': frame, 'anchor_frame': anchor_frame, 'method': method,
                         'prediction': pred.tolist(), 'truth': truth.tolist(), 'iou': iou(pred, truth),
                         'center_error_px': float(np.linalg.norm((pred[:2]+pred[2:]-truth[:2]-truth[2:])/2)),
                         'width_error_px': float(pred[2]-pred[0]-truth[2]+truth[0]),
                         'height_error_px': float(pred[3]-pred[1]-truth[3]+truth[1])})
    # Separate online-adaptation diagnostic; compare the same later anchor and
    # frames, never call these one-observation transfer measurements.
    start_frame = observations[2][0]
    early = [(clock[f], b) for f, b in observations[:3]]
    for frame, truth in observations[3:]:
        for method in ('one_box_shared', 'rational_centre_fixed_size', 'rational_edges'):
            pred = forecast(method, early, clock[frame], val_model)
            rows.append({'frame': frame, 'anchor_frame': start_frame, 'method': 'three_views:'+method,
                         'prediction': pred.tolist() if pred is not None else None,
                         'truth': truth.tolist(), 'iou': iou(pred, truth) if pred is not None else 0.})
    summary = {}
    for method in sorted({r['method'] for r in rows}):
        selected = [r for r in rows if r['method'] == method]
        summary[method] = {'future_sampled_frames': len(selected),
            'passing_iou50': sum(r['iou'] >= .5 for r in selected),
            'mean_iou': float(np.mean([r['iou'] for r in selected])),
            'minimum_iou': min(r['iou'] for r in selected), 'last_frame': selected[-1]}
    old = {r['frame']: r['object_bbox_source_xyxy'] for r in json.loads((ROOT/'data/drone/training/manual-validation/large-tower-crops/manifest.json').read_text())['records']}
    report = {'metadata': {
        'scope': 'One physical held-out tower in a different flight, with participant visual reannotation.',
        'labels': 'Assistant inspected each sampled box before scoring; not exact organizer ground truth.',
        'training': 'All complete official reference large_tower boxes, frames 7 through 24. No validation target views used by transferred profiles.',
        'initial_observation': 'One manually supplied box at validation frame 38. No automatic detector or pose ground truth.',
        'calibration': 'Native validation frames 5 and 6, before the target enters. No later image geometry fitting.',
        'timing': 'Measured background motion clock supplied. Steps from target frame 38 through 67 are all normal.',
        'limits': 'Twelve sampled boxes on one instance; not whole-track per-frame success or coverage of all orientations.',
        'no_noise_added': True, 'no_test_fitting': True},
        'profiles': {'same_class_direct': direct.to_dict(), 'same_class_residual': residual.to_dict()},
        'initial_axis_proxy': {'radians': angle, 'anisotropy': score}, 'results': summary,
        'old_boxes_vs_visual_reannotation': [{'frame': f, 'iou': iou(old[f], b)} for f, b in observations]}
    (OUT/'tower-transfer-measurements.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (OUT/'tower-transfer-predictions.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    print(json.dumps({method: {k: v for k, v in result.items() if k != 'last_frame'} for method, result in summary.items()}, indent=2))
    print('Final frame predictions:', json.dumps({method: result['last_frame']['prediction'] for method, result in summary.items()}))


if __name__ == '__main__':
    main()
