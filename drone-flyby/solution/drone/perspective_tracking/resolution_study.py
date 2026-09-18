"""Compare actual calibration resolutions and flag classes needing attention.

IoU itself is invariant to uniform image scaling. These experiments change
feature measurements through real image resizing, not fabricated box errors.
"""
import json
from pathlib import Path

from .shape_study import summarize


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/drone-edge-profiles'


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def describe(rows):
    return {**summarize(rows), 'minimum_iou': min((r['iou'] for r in rows), default=None),
            'passing_iou75': sum(r['iou'] >= .75 for r in rows)}


def main():
    sources = {}
    for name in ('native', 'overview'):
        folder = OUT if name == 'native' else OUT/'overview'
        sources[name] = read_rows(folder/'predictions.jsonl')
        data = json.loads((folder/'measurements.json').read_text())
        for key, profile in data['profiles'].items():
            mode, target = key.split(':')
            if mode in ('leave_target_out_pooled', 'other_tower_only'):
                if target in profile['training']['instance_views']:
                    raise ValueError(f'Target leaked into training: {name}/{key}')
            elif max(profile['training_frames']) >= profile['forecast_anchor']:
                raise ValueError(f'Future target view leaked into training: {name}/{key}')
    baselines = {name: [r for r in rows if r['mode'] == 'leave_target_out_pooled'
                       and r['method'] == 'shared_plane'] for name, rows in sources.items()}
    baselines['half_resolution'] = [
        {**r, 'class': r['track_id'], 'anchor': r['anchor_frame']}
        for r in read_rows(ROOT/'artifacts/drone-runtime-tracker/full/scored-predictions.jsonl')]
    cohorts = [{(r['class'], r['anchor'], r['frame']) for r in rows} for rows in baselines.values()]
    if any(cohort != cohorts[0] for cohort in cohorts):
        raise ValueError('Resolution comparison must use identical targets and forecast horizons')
    classes = sorted({r['class'] for r in baselines['native']})
    by_class = {name: {resolution: describe([r for r in rows if r['class'] == name])
                       for resolution, rows in baselines.items()} for name in classes}
    overview_failures = [name for name in classes
                         if by_class[name]['overview']['fraction_iou50'] < 1.]
    any_failures = [name for name in classes
                   if any(s['fraction_iou50'] < 1. for s in by_class[name].values())]
    adaptive = {}
    for name in overview_failures:
        adaptive[name] = {}
        for method in ('shared_plane', 'learned_profile'):
            selected = [r for r in sources['overview'] if r['class'] == name
                        and r['mode'] == 'within_instance_temporal_5' and r['method'] == method]
            adaptive[name][method] = describe(selected) if selected else None
    result = {'metadata': {
        'resolutions': {'native': [3840, 2160], 'half_resolution': [1920, 1080], 'overview': [960, 540]},
        'degradation': 'Real image resizing changes first-pair feature calibration. No artificial box errors.',
        'initial_boxes': 'Exact official source boxes for all resolutions; detector localization degradation is not measured.',
        'flatness': 'A miss does not identify height: calibration error, silhouette change and clipping can all contribute.',
        'adaptive': 'Five earlier views of each same target fit a separate profile; sixth view anchors a later forecast. Temporal holdout only.',
        'held_out_instances': 'Pooled transfer excludes all views of each target from fitting, but training is still from the same reference flight.'},
        'baselines': {name: describe(rows) for name, rows in baselines.items()},
        'by_class': by_class, 'overview_failure_classes': overview_failures,
        'failure_classes_at_any_resolution': any_failures,
        'per_object_temporal_correction': adaptive,
        'pooled_overview_diagnostic': describe([r for r in sources['overview']
            if r['mode'] == 'leave_target_out_pooled' and r['method'] == 'learned_profile'])}
    (OUT/'resolution-stress.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'baselines': result['baselines'], 'overview_failure_classes': overview_failures,
                      'failure_classes_at_any_resolution': any_failures,
                      'per_object_temporal_correction': adaptive}, indent=2))


if __name__ == '__main__':
    main()
