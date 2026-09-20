#!/usr/bin/env python3
"""Publish a served validation attempt (endpoint frame log + captures + portal result) to the shared tracker app.

Same conversion as experiments/pipeline300-20260919/checkpoint-02/publish_api_review.py, with arguments:
    python3 publish_api_run.py --frames-dir <run>/frames --result <attempt>/result.json --run-id verifier-g05-api-20260919 \
        --title "Checkpoint 02 + verifier · validation API" --description "..." [--device "L40S (Helsinki)"]
The frame log with the most rows in --frames-dir is the attempt; captures are copied into the run's inputs/.
run.json is written last by atomic rename.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames-dir', type=Path, required=True)
    ap.add_argument('--result', type=Path, required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--title', required=True)
    ap.add_argument('--description', required=True)
    ap.add_argument('--device', default='L40S')
    ap.add_argument('--classification', default='Real organizer validation output; detector confidences are the verifier-adjusted values the endpoint used. Fresh/projected from logged track events.')
    a = ap.parse_args()
    OUT = ROOT / 'drone/tracker_app/runs' / a.run_id
    result = json.loads(a.result.read_text())
    files = list(a.frames_dir.glob('*.jsonl'))
    source = max(files, key=lambda p: len(p.read_text().splitlines()))
    rows = {r['frame']: r for r in map(json.loads, source.read_text().splitlines())}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'inputs').mkdir(exist_ok=True)
    frames = []; size = np.array([3840, 2160, 3840, 2160])
    for frame in range(1, 250):
        r = rows.get(frame)
        background = ROOT / f'artifacts/drone-validation-box-review-20260919/backgrounds/frame_{frame:06d}.jpg'
        f = dict(frame=frame, available=bool(r), background=os.path.relpath(background, OUT), detections=[], tracker=[], forecasts=[], pending=[])
        if r:
            for k in ['level', 'region', 'total_ms', 'detector_ms', 'events', 'timing']:
                f[k] = r.get(k)
            p = a.frames_dir / 'captures' / f'{source.stem}-{frame:06d}.png'
            if p.exists():
                shutil.copy2(p, OUT / 'inputs' / p.name); f['exact_capture'] = 'inputs/' + p.name
            scale = np.tile((np.array(r['region'][2:]) - r['region'][:2]) / [960, 540], 2); origin = np.tile(r['region'][:2], 2)
            f['detections'] = [dict(id=f'd{i}', class_name=d['label'], confidence=d['confidence'], box=(np.array(d['box']) * scale + origin).tolist())
                               for i, d in enumerate(r.get('raw_detections', []))]
            def disp(t):
                return dict(id=t['track_id'], class_name=t['object_id'], confidence=t['confidence'], box=t['bbox_source_xyxy'])
            fresh = {e.get('track_id') for e in r.get('events', []) if e['event'] in ('birth', 'refresh', 'entry_birth')}
            f['forecasts'] = [disp(t) for t in r.get('pre_update_predictions', [])]
            f['tracker'] = [dict(disp(t), kind='refreshed' if t['track_id'] in fresh or t['track_id'].startswith('transient-') else 'projected')
                            for t in r.get('track_details', [])]
            if not r.get('track_details'):
                f['tracker'] = [dict(id=f'w{i}', class_name=x['object_id'], confidence=x['confidence'], box=(np.array(x['bbox']) * size).tolist(), kind='refreshed')
                                for i, x in enumerate(r['response'])]
            assert len(f['tracker']) == len(r['response']), (frame, len(f['tracker']), len(r['response']))
        frames.append(f)
    ms = sorted(r['total_ms'] for r in rows.values())
    summary = dict(api_score=result.get('score'), sequence_id=source.stem, frames=len(rows), missing_frames=sorted(set(range(1, 250)) - set(rows)),
                   levels=dict(collections.Counter(r['level'] for r in rows.values())), p95_ms=ms[int(.95 * (len(ms) - 1))], max_ms=max(ms),
                   over_300=sum(v > 300 for v in ms), device=a.device,
                   background='Reconstructed validation full-scene context; exact API camera captures provided separately',
                   classification=a.classification)
    (OUT / 'review.json').write_text(json.dumps(dict(summary=summary, frames=frames)))
    shutil.copy2(a.result, OUT / 'api-result.json'); shutil.copy2(source, OUT / 'run.jsonl')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    created = result.get('finished_at') or datetime.now(timezone.utc).isoformat()
    tmp = OUT / 'run.json.tmp'
    tmp.write_text(json.dumps(dict(title=a.title, created_at=created, description=a.description, data='review.json'), indent=2))
    tmp.replace(OUT / 'run.json')
    print(json.dumps({k: v for k, v in summary.items() if k != 'missing_frames'}, indent=1), 'missing', len(summary['missing_frames']))
    print('published', OUT)


if __name__ == '__main__':
    main()
