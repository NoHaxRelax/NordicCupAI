"""Test one-box predictions using profiles learned before the target forecast.

Separates within-instance temporal holdout from cross-instance transfer.
No trained class transfer claim can be made from one reference instance/class.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .edge_profile import EdgeMotionProfile
from .motion import MotionModel, ProjectionError
from .replay import iou
from .shape_study import full, summarize


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/drone-edge-profiles'
STEP_M = 13.8888889


def evaluate(track, name, anchor, model, profile, mode):
    rows = []
    box = track[anchor]
    for frame, truth in track.items():
        if frame <= anchor:
            continue
        for method in ['shared_plane', 'learned_profile']:
            try:
                pred = (model.box(box, anchor, frame) if method == 'shared_plane' else
                        profile.predict(model, box, anchor, distance_m=(frame-anchor)*STEP_M))
            except ProjectionError:
                pred = None
            rows.append({'class': name, 'frame': frame, 'anchor': anchor, 'mode': mode,
                         'method': method, 'prediction': pred.tolist() if pred is not None else None,
                         'truth': truth.tolist(), 'iou': iou(pred, truth) if pred is not None else 0.})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', choices=('native', 'overview'), default='native')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tracks = {}
    for path in sorted((ROOT/'data/drone/reference/helsinki/annotations').glob('*.json')):
        data = json.loads(path.read_text())
        for a in data['annotations']:
            tracks.setdefault(a['object_id'], {})[data['frame']] = np.array(a['bbox'], float)
    if args.calibration == 'native':
        model = MotionModel.from_dict(json.loads((ROOT/'artifacts/drone-runtime-tracker/diagnosis/native-calibration.json').read_text())['model'])
    else:
        model = MotionModel.from_dict(json.loads((ROOT/'artifacts/drone-runtime-tracker/overview/model.json').read_text()))
    rows, profiles, failures = [], {}, []
    for name, track in tracks.items():
        complete = [f for f, b in track.items() if f >= 1 and full(b)]
        if not complete:
            continue
        anchor = complete[0]
        if not any(f > anchor for f in track):
            continue
        # Strict target exclusion: all views of this physical instance are left out.
        modes = {'leave_target_out_pooled': {n: list(t.items()) for n, t in tracks.items() if n != name}}
        if name in ['small_tower', 'large_tower']:
            other = 'small_tower' if name == 'large_tower' else 'large_tower'
            modes['other_tower_only'] = {other: list(tracks[other].items())}
        for mode, training in modes.items():
            profile = EdgeMotionProfile.fit(model, training, label=mode, reference_step_m=STEP_M)
            profiles[f'{mode}:{name}'] = profile.to_dict()
            rows += evaluate(track, name, anchor, model, profile, mode)
        # Fit earlier views of the SAME object, then take one later anchor box.
        # This tests temporal extrapolation only, not unseen-instance transfer.
        for training_count in [5, 8]:
            if len(complete) <= training_count+1:
                continue
            training_frames = complete[:training_count]
            anchor = complete[training_count]
            mode = f'within_instance_temporal_{training_count}'
            try:
                profile = EdgeMotionProfile.fit(model, {name: [(f, track[f]) for f in training_frames]},
                                                label=name, reference_step_m=STEP_M)
            except ValueError as exc:
                failures.append({'mode': mode, 'class': name, 'error': str(exc)})
                continue
            profiles[f'{mode}:{name}'] = {**profile.to_dict(), 'training_frames': training_frames,
                                         'forecast_anchor': anchor}
            rows += evaluate(track, name, anchor, model, profile, mode)
    summaries = []
    for mode in sorted({r['mode'] for r in rows}):
        for method in ['shared_plane', 'learned_profile']:
            selected = [r for r in rows if r['mode'] == mode and r['method'] == method]
            summary = {'mode': mode, 'method': method, 'all': summarize(selected),
                       'towers': summarize([r for r in selected if r['class'] in ['small_tower', 'large_tower']]),
                       'per_class': {name: summarize([r for r in selected if r['class'] == name])
                                     for name in sorted({r['class'] for r in selected})}}
            summaries.append(summary)
            print(mode, method, summary['all'], 'towers', summary['towers'], flush=True)
    data = {'metadata': {
        'scope': 'Reference-only exploratory model of effective height/shape impact; not physical height estimation.',
        'calibration': f'First-pair {args.calibration} camera motion. Profiles reuse that calibration; cross-flight camera changes are not tested.',
        'leave_target_out': 'No target box beyond its first complete box enters fitting. Other reference instances supply profile training views. Training uses their entire recorded tracks.',
        'other_tower_only': 'Train on the small tower and test the large tower, then reverse. These are different classes and different image locations.',
        'within_instance_temporal': 'Earlier views train the profile; one later box initializes the forecast; only still-later labels are scored. Same instance, so not unseen-object transfer.',
        'one_reference_instance_per_class': True, 'artificial_noise': False,
        'speed': 'distance_m is integrated travel; constant speed uses speed_m_s * elapsed_seconds.'},
        'profiles': profiles, 'summaries': summaries, 'fit_failures': failures}
    (args.output/'measurements.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    (args.output/'predictions.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))


if __name__ == '__main__':
    main()
