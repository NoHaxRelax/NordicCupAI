#!/usr/bin/env python3
"""Verify bounded track proposals with isolated validation score probes."""
import json
from pathlib import Path
import subprocess
import sys

from oracle import normalized_prediction

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / 'artifacts/drone-api-tests/score-probes'
CAPTURES = ROOT / 'data/drone/capture/mining'
SOURCE = ROOT / 'data/drone/mined/large-launcher-track-proposals.json'
BASELINE = 0.0015232292460015233
TILES = [(2400,270),(3360,270),(1440,270),(2400,270),(3360,270),(1440,270)]


def main():
    document = json.loads(SOURCE.read_text())
    for annotation, tile in zip(document['annotations'], TILES):
        frame = annotation['frame']
        name = f'track-large-launcher-{frame}'
        prediction = normalized_prediction(annotation['bbox_source_xyxy'], annotation['object_id'], 1.0)
        plan = {'name': name, 'target': list(tile), 'candidate_frame': frame,
                'candidate_bbox_source': annotation['bbox_source_xyxy'],
                'predictions_by_frame': {str(frame): [prediction]}}
        plan_path = PROBES / f'{name}-plan.json'
        folder = PROBES / name
        if not (folder / 'queue.json').exists():
            if len(list(PROBES.glob('*/queue.json'))) >= 30:
                raise RuntimeError('Approved 30-run session cap reached')
            plan_path.write_text(json.dumps(plan, indent=2) + '\n')
            subprocess.run([sys.executable, str(ROOT/'drone/run_probe.py'), str(plan_path)], check=True)
        result = json.loads((folder/'result.json').read_text())
        if result.get('errors'):
            raise RuntimeError(f'{name} returned API errors')
        records = [json.loads(p.read_text()) for p in (CAPTURES/name).rglob('*.json')]
        target = [r for r in records if r['frame'] == frame]
        if len(target) != 1 or target[0]['response']['annotations'] != [prediction]:
            annotation['score_verification'] = {'status':'unusable_target_capture','score':result['score']}
        else:
            matched = abs(result['score']-BASELINE) <= max(1e-12, BASELINE*1e-7)
            annotation['score_verification'] = {
                'status': 'score_confirmed_match' if matched else 'score_did_not_confirm_match',
                'score': result['score'], 'reference_score': BASELINE, 'probe': name,
                'provenance': 'isolated frame and class prediction; confirms an IoU >= 0.50 match, not exact ground truth coordinates'
            }
        annotation['status'] = annotation['score_verification']['status']
        SOURCE.write_text(json.dumps(document, indent=2) + '\n')
        print(json.dumps({'frame':frame, 'score':result['score'],
                          'status':annotation['score_verification']['status'],
                          'recorded_requests':len(records)}), flush=True)


if __name__ == '__main__':
    main()
