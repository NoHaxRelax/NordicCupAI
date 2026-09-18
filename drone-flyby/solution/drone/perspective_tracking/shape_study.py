"""Causal early-observation study for object-dependent box transformations.

Uses known reference boxes. Later labels score forecasts only; no added noise.
Does not establish class-prior transfer: reference has one instance per class.
"""
import json
from pathlib import Path

import numpy as np

from .motion import MotionModel
from .replay import iou


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/drone-shape-study'
TOWERS = {'small_tower', 'large_tower'}


def full(box):
    return bool(np.all(box[:2] > 0) and np.all(box[2:] < [3839, 2159]))


def summarize(rows):
    classes = sorted({r['class'] for r in rows})
    return {'future_boxes': len(rows), 'tracks': len(classes),
            'passing_iou50': sum(r['iou'] >= .5 for r in rows),
            'fraction_iou50': float(np.mean([r['iou'] >= .5 for r in rows])) if rows else None,
            'whole_tracks': sum(all(r['iou'] >= .5 for r in rows if r['class'] == name) for name in classes)}


def forecast(method, observations, future, model):
    anchor, box = observations[-1]
    if method == 'one_box_shared':
        return model.box(box, anchor, future)
    times = np.array([f for f, b in observations], float)-anchor
    boxes = np.array([b for f, b in observations])
    e = np.array(model.diagnostics['epipole_homogeneous'])
    ep = e[:2]/e[2]
    dt = future-anchor
    if method == 'rational_edges':
        # Each stable projected 3D extremum has reciprocal distance from the
        # epipole linear in time. Extrema can switch; this is an approximation.
        offsets = boxes - np.tile(ep, 2)
        if np.any(np.abs(offsets) < 1e-5):
            return None
        coeff = np.polyfit(times, 1/offsets, 1)
        denominator = coeff[0]*dt+coeff[1]
        if np.any(denominator*offsets[-1] <= 0):
            return None
        result = np.tile(ep, 2)+1/denominator
    else:
        centres = (boxes[:, :2]+boxes[:, 2:])/2
        vectors = centres-ep
        radii = np.linalg.norm(vectors, axis=1)
        coeff = np.polyfit(times, 1/radii, 1)
        denominator = coeff[0]*dt+coeff[1]
        if denominator <= 0:
            return None
        direction = np.mean(vectors/radii[:, None], axis=0)
        direction /= np.linalg.norm(direction)
        centre = ep+direction/denominator
        sizes = boxes[:, 2:]-boxes[:, :2]
        size = sizes[-1].copy()
        if method == 'rational_centre_linear_size':
            size += np.polyfit(times, sizes, 1)[0]*dt
        if np.any(size <= 0):
            return None
        result = np.r_[centre-size/2, centre+size/2]
    return result if np.all(result[2:] > result[:2]) and np.all(np.isfinite(result)) else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = ROOT/'data/drone/reference/helsinki/annotations'
    tracks = {}
    for path in sorted(base.glob('*.json')):
        data = json.loads(path.read_text())
        for a in data['annotations']:
            tracks.setdefault(a['object_id'], {})[data['frame']] = np.array(a['bbox'], float)
    native_data = json.loads((ROOT/'artifacts/drone-runtime-tracker/diagnosis/native-calibration.json').read_text())
    model = MotionModel.from_dict(native_data['model'])
    original = MotionModel.from_dict(json.loads((ROOT/'artifacts/drone-runtime-tracker/full/model.json').read_text()))
    original_rows = [json.loads(line) for line in (ROOT/'artifacts/drone-runtime-tracker/full/scored-predictions.jsonl').read_text().splitlines()]
    strata = []
    for tag, candidate in [('original_half_size_calibration', original), ('native_calibration', model)]:
        rows = []
        for r in original_rows:
            pred = candidate.box(tracks[r['track_id']][r['anchor_frame']], r['anchor_frame'], r['frame'])
            rows.append({'class': r['track_id'], 'frame': r['frame'], 'iou': iou(pred, tracks[r['track_id']][r['frame']])})
        strata.append({'calibration': tag, 'all': summarize(rows),
                       'excluding_named_towers': summarize([r for r in rows if r['class'] not in TOWERS]),
                       'named_towers': summarize([r for r in rows if r['class'] in TOWERS]),
                       'per_class': {name: summarize([r for r in rows if r['class'] == name]) for name in tracks}})
    summaries, all_rows = [], []
    methods = ['one_box_shared', 'rational_centre_fixed_size', 'rational_centre_linear_size', 'rational_edges']
    for spacing in [1, 4]:
        for count in [2, 3, 5]:
            by_method = {method: [] for method in methods}
            for name, track in tracks.items():
                complete = [f for f, b in track.items() if f >= 1 and full(b)]
                if not complete:
                    continue
                frames = [complete[0]+spacing*i for i in range(count)]
                if any(f not in complete for f in frames):
                    continue
                observations = [(f, track[f]) for f in frames]
                for future, truth in track.items():
                    if future <= frames[-1]:
                        continue
                    for method in methods:
                        pred = forecast(method, observations, future, model)
                        by_method[method].append({'class': name, 'frame': future, 'anchor_frame': frames[-1],
                                                 'initial_frames': frames if method != 'one_box_shared' else frames[-1:],
                                                 'prediction': pred.tolist() if pred is not None else None,
                                                 'truth': truth.tolist(), 'iou': iou(pred, truth) if pred is not None else 0.})
            for method, rows in by_method.items():
                summary = {'observation_count': count, 'spacing_frames': spacing, 'method': method,
                           'all': summarize(rows),
                           'named_towers': summarize([r for r in rows if r['class'] in TOWERS]),
                           'ta_ta': summarize([r for r in rows if r['class'] == 'ta-ta'])}
                summaries.append(summary)
                all_rows.extend({**r, 'count': count, 'spacing': spacing, 'method': method} for r in rows)
                print(count, spacing, method, summary['all'], 'towers', summary['named_towers'], flush=True)
    inventory = []
    for name, track in tracks.items():
        frames = [f for f, box in track.items() if full(box)]
        first, last = frames[0], frames[-1]
        sizes = [track[f][2:]-track[f][:2] for f in (first, last)]
        inventory.append({'class': name, 'labelled_views': len(track), 'complete_views': len(frames),
                          'first_complete_frame': first, 'last_complete_frame': last,
                          'first_wh': sizes[0].tolist(), 'last_wh': sizes[1].tolist()})
    data = {'metadata': {
        'scope': 'Reference-only exploratory geometry study; exact initial boxes, one physical instance per class.',
        'height_groups': 'Tower exclusion uses class names small_tower and large_tower, not measured physical height or prediction success.',
        'causality': 'Native calibration uses first image pair only; each forecast uses only declared initial boxes. Future labels score only.',
        'matching': 'Methods within one observation-count/spacing cell have identical future labels and latest anchor. One-box baseline receives only that latest box.',
        'four_tick_spacing': 'Temporal revisit proxy only, not a legal crop-visibility replay.',
        'added_noise': False}, 'inventory': inventory, 'stratified_results': strata, 'early_observation_results': summaries}
    (OUT/'measurements.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    (OUT/'predictions.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in all_rows))


if __name__ == '__main__':
    main()
