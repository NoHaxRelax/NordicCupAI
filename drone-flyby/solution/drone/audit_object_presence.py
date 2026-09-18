"""Independent structural/coverage audit of the offline location artifacts.

Does not establish visual recall or organizer ground-truth correctness.
"""
import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def audit(require_complete=False):
    base = ROOT / 'data/drone/mined'
    dataset = json.loads((base / 'object-presence-pass.json').read_text())
    progress = json.loads((base / 'progress.json').read_text())
    manifest = json.loads((ROOT / 'data/drone/reconstructed-validation/manifest.json').read_text())
    annotations = dataset['annotations']
    issues = []
    def check(condition, message):
        if not condition:
            issues.append(message)
    coverage = dataset['coverage']
    check(len(coverage) == 249 and {r['frame'] for r in coverage} == set(range(1, 250)), 'Coverage must contain each frame exactly once')
    keys = set()
    for index, row in enumerate(annotations):
        tag = f'Annotation {index}'
        check(row.get('frame') in range(1,250), tag + ' has invalid frame')
        key = (row.get('frame'), row.get('track_id'))
        check(key not in keys, tag + ' duplicates a frame/track pair')
        keys.add(key)
        b = row.get('bbox_source_xyxy', [])
        if len(b) != 4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in b):
            issues.append(tag + ' has invalid coordinates')
            continue
        check(0 <= b[0] < b[2] <= 3840 and 0 <= b[1] < b[3] <= 2160, tag + ' lies outside source image')
        normalized = row.get('bbox_normalized_xyxy', [])
        check(len(normalized) == 4 and all(abs(x-y/z) < 2e-5 for x,y,z in zip(normalized,b,[3840,2160,3840,2160])), tag + ' normalized coordinates disagree')
        check(isinstance(row.get('unverified'), bool), tag + ' lacks explicit uncertainty')
        check(bool(row.get('provenance')) and bool(row.get('evidence')), tag + ' lacks provenance')
        if row.get('unverified') is False:
            check(row.get('evidence') == 'score_confirmed', tag + ' claims verification without score evidence')
        if row.get('frame', 250) < 5:
            from PIL import Image
            with Image.open(ROOT / f'data/drone/reconstructed-validation/frame_{row["frame"]:06d}.png') as im:
                alpha = im.getchannel('A').crop((math.floor(b[0]), math.floor(b[1]), math.ceil(b[2]), math.ceil(b[3])))
                check(alpha.getextrema() == (255,255), tag + ' overlaps missing pixels')
    with (base / 'object-presence-pass.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    check(len(rows) == len(annotations), 'CSV row count differs from JSON')
    csv_map = {(int(r['frame']), r['track_id']):r for r in rows}
    check(set(csv_map) == keys, 'CSV frame/track pairs differ from JSON')
    for a in annotations:
        r = csv_map.get((a['frame'], a['track_id']))
        if r:
            check(json.loads(r['bbox_source_xyxy']) == a['bbox_source_xyxy'], 'CSV coordinates differ from JSON')
    confirmed = [a for a in annotations if a.get('unverified') is False]
    check(len(confirmed) == 7 and {a['frame'] for a in confirmed} == set(range(140,147)), 'Seven original score-confirmed records were not preserved')
    for row in coverage:
        check(row['annotation_count'] == sum(a['frame'] == row['frame'] for a in annotations), f'Frame {row["frame"]} annotation count stale')
    reviewed = set(progress.get('frames_reviewed', []))
    available = {m['frame'] for m in manifest if m['native_coverage'] > 0}
    remaining = sorted(available - reviewed)
    if require_complete:
        check(not remaining, f'Available frames remain unreviewed: {remaining}')
        check(progress.get('status') == 'complete', 'Worker has not marked completion')
        for row in coverage:
            check(row['frame'] not in reviewed or row['inspection_status'] != 'unreviewed', f'Frame {row["frame"]} coverage contradicts review log')
    updated = datetime.fromisoformat(progress['updated_at'].replace('Z','+00:00'))
    check((updated-datetime.now(timezone.utc)).total_seconds() < 60, 'Worker timestamp is in the future')
    result = {'passed': not issues, 'completion_required': require_complete, 'issues': issues,
              'box_instances': len(annotations), 'track_ids': len({a['track_id'] for a in annotations}),
              'score_confirmed': len(confirmed), 'reviewed_frames': len(reviewed),
              'available_frames_without_review': remaining,
              'unavailable_native_frames': sorted(set(range(1,250))-available),
              'limitations': 'Structural checks do not prove visual recall or ground-truth accuracy.'}
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.require_complete)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
