#!/usr/bin/env python3
"""Marks (crop_picker export format) for the verifier's real positives, other-object crops and the test set.

  real-marks.json   kind 'real'  : the four small classes from Helsinki organiser labels, validation v8 pseudo-labels
                                   and Elias's hidden-object tracks (validation_hidden2.json, class per track)
                    kind 'other' : every other labelled class (Helsinki all frames, validation v8), one mark per box
  test-marks.json   kind 'test'  : the detector's raw boxes for the four small classes from a pipeline300 replay log,
                                   centre converted to source px, with det_conf / det_label / det_level and a truth:
                                   object (same-class label or hidden track), other_object, terrain (no label, no
                                   hidden zone, no label centre within 64 px), plus the frame's team split.
Splits: helsinki -> train; validation 1-90 train, 91-99 gap, 100-180 dev, 181+ reserved (dev/reserved real marks are
kept with their split for evaluation). Every scene's dir_from_root is relative to a root that holds
data/drone/reconstructed-validation and data/drone/reference/helsinki/images (symlinks on the pod).

    python3 build_marks.py --out artifacts/drone-verifier-20260919/marks
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W, H = 3840, 2160
SMALL = ['small_launcher', 'medium_launcher', 'ta-ta', 'jammer']
FACTOR = {0: 4, 1: 2, 2: 1}
SCENES = {'validation': {'dir_from_root': 'data/drone/reconstructed-validation', 'pattern': 'frame_%06d.png'},
          'helsinki': {'dir_from_root': 'data/drone/reference/helsinki/images', 'pattern': 'frame_%06d.png'}}


def split_of(scene, frame):
    if scene == 'helsinki':
        return 'train'
    if frame <= 90:
        return 'train'
    if frame <= 99:
        return 'gap'
    if frame <= 180:
        return 'dev'
    return 'reserved'


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def load_labels(v8_path, hidden2_path):
    """{(scene, frame): [(class, box_source_xyxy, source)]}"""
    labels = defaultdict(list)
    for f in sorted((ROOT / 'data/drone/reference/helsinki/annotations').glob('frame_*.json')):
        d = json.loads(f.read_text())
        for a in d['annotations']:
            labels[('helsinki', int(d['frame']))].append((a['object_id'], [float(v) for v in a['bbox']], 'organiser'))
    v8 = json.loads(Path(v8_path).read_text())['predictions_by_frame']
    for fr, rows in v8.items():
        for o in rows:
            b = o['bbox']; labels[('validation', int(fr))].append((o['object_id'], [b[0] * W, b[1] * H, b[2] * W, b[3] * H], 'v8'))
    h2 = json.loads(Path(hidden2_path).read_text())
    tracks, zones, mism = h2['tracks'], h2['zones'], 0
    for fr, boxes in zones.items():
        f = int(fr)
        active = [t for t in tracks if t['kept_out'][0] <= f <= t['kept_out'][1]]
        if len(active) != len(boxes):
            mism += 1
            for b in boxes:  # class unknown: keep as a keep-out only
                labels[('validation', f)].append(('unknown', b, 'hidden2'))
            continue
        for t, b in zip(active, boxes):
            what = t['what'] if t['what'] in SMALL or t['what'] in ('small_tower', 'tank') else 'unknown'
            # zone is twice the object: shrink to the object's size around the centre for a fair box
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; w, h = t['size']
            labels[('validation', f)].append((what, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 'hidden2'))
    print(f'hidden2: {len(zones)} frames, {mism} with a track/zone count mismatch (kept as unknown keep-out)')
    return labels


def mark(scene, frame, cls, kind, x, y, mid, **extra):
    m = {'id': mid, 'scene': scene, 'frame': frame, 'x': round(float(x), 1), 'y': round(float(y), 1), 'cls': cls, 'kind': kind,
         'window_source_px': 128, 'near_label': False, 'split': split_of(scene, frame)}
    m.update(extra)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--v8', default=str(ROOT / 'artifacts/drone-api-tests/score-anchored-v8-full-20260918/full-validation-plan.json'))
    ap.add_argument('--hidden', default=str(ROOT / 'drone/crop_picker/elias-data/validation_hidden.json'))
    ap.add_argument('--hidden2', default=str(ROOT / 'drone/crop_picker/elias-data/validation_hidden2.json'))
    ap.add_argument('--replay', default=str(ROOT / 'experiments/pipeline300-20260919/checkpoint-02/replay/checkpoint-02-local.jsonl'))
    ap.add_argument('--other-every', type=int, default=2, help='keep every Nth frame of an other-class track')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    labels = load_labels(a.v8, a.hidden2)
    hidden1 = json.loads(Path(a.hidden).read_text())['zones']  # small_plane row, one zone per frame

    real, other = [], []
    per_track_seen = Counter()
    for (scene, frame), rows in sorted(labels.items()):
        for k, (cls, b, src) in enumerate(rows):
            if cls == 'unknown':
                continue
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if not (16 <= cx <= W - 16 and 16 <= cy <= H - 16):
                continue
            mid = f'{scene}-{frame}-{cls}-{k}-{src}'
            if cls in SMALL:
                real.append(mark(scene, frame, cls, 'real', cx, cy, mid, label_source=src))
            else:
                key = (scene, cls, round(cx / 400), round(cy / 400))  # coarse spatial key: thin dense tracks
                per_track_seen[key] += 1
                if (per_track_seen[key] - 1) % a.other_every == 0:
                    other.append(mark(scene, frame, cls, 'other', cx, cy, mid, label_source=src))
    real_marks = {'version': 1, 'tool': 'build_marks.py', 'crop_spec': {'window_source_px': 128}, 'scenes': SCENES, 'marks': real + other}
    (out / 'real-marks.json').write_text(json.dumps(real_marks))
    print('real:', dict(Counter((m['cls'], m['split']) for m in real)))
    print('other:', dict(Counter(m['split'] for m in other)), 'classes', len({m['cls'] for m in other}))

    tests = []; truth_count = Counter()
    for line in open(a.replay):
        r = json.loads(line)
        frame, level, region = int(r['frame']), int(r['level']), r['region']
        f = FACTOR[level]
        rows = labels.get(('validation', frame), [])
        for i, d in enumerate(r.get('raw_detections') or []):
            if d['label'] not in SMALL:
                continue
            b = d['box']
            sb = [region[0] + b[0] * f, region[1] + b[1] * f, region[0] + b[2] * f, region[1] + b[3] * f]
            cx, cy = (sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2
            best = max(((iou(sb, lb), lc) for lc, lb, _ in rows), default=(0.0, None))
            in_zone = [lc for lc, lb, src in rows if src == 'hidden2' and lb[0] - 20 <= cx <= lb[2] + 20 and lb[1] - 20 <= cy <= lb[3] + 20]
            z1 = hidden1.get(str(frame))
            in_plane_row = bool(z1) and z1[0] <= cx <= z1[2] and z1[1] <= cy <= z1[3]
            near = any(abs((lb[0] + lb[2]) / 2 - cx) < 64 and abs((lb[1] + lb[3]) / 2 - cy) < 64 for _, lb, _ in rows)
            if best[0] >= 0.3 and best[1] == d['label'] or d['label'] in in_zone:
                truth = 'object'
            elif best[0] >= 0.3 or in_zone or in_plane_row:
                truth = 'other_object' if (best[1] not in (None, 'unknown') or in_plane_row or any(c != 'unknown' for c in in_zone)) else 'unknown'
            elif near:
                truth = 'unknown'
            else:
                truth = 'terrain'
            truth_count[(d['label'], truth)] += 1
            tests.append(mark('validation', frame, d['label'], 'test', cx, cy, f'test-{frame}-{i}-{d["label"]}',
                              det_conf=float(d['confidence']), det_label=d['label'], det_level=level, truth=truth,
                              split_frame=split_of('validation', frame), det_box_source=[round(v, 1) for v in sb]))
    test_marks = {'version': 1, 'tool': 'build_marks.py', 'crop_spec': {'window_source_px': 128}, 'scenes': SCENES, 'marks': tests}
    (out / 'test-marks.json').write_text(json.dumps(test_marks))
    print('test:', len(tests))
    for c in SMALL:
        print(f'  {c:16s}', {t: n for (cc, t), n in sorted(truth_count.items()) if cc == c})


if __name__ == '__main__':
    main()
