"""Where does each class lose its AP? One served run (diagnostics JSONL) against the scene's label files.

    python elias/miss_analysis.py --log elias/out/harness/W_combo_ref/local.jsonl [--scene validation]

Every labelled box of every answered frame lands in exactly one bin:
  hit          an answer of the right class with IoU >= 0.5
  box          right class, best IoU 0.1 to 0.5 (a box size or position problem, the launcher case)
  confused     an answer of ANOTHER class with IoU >= 0.3 (the class it was called is reported)
  held_back    no answer, the object is in the delivered view and a raw detection overlaps it (IoU >= 0.3): the tracker
               did not emit it (birth threshold, ambiguity rule, conflicting class)
  unseen       no answer, in the delivered view, no raw detection on it: the detector missed it at this size
  never_born   no answer, outside the delivered view, and the class was never answered on this object before
  lost         no answer, outside the delivered view, answered before: the track died or its forecast drifted off
Plus per class: answers with no label of any class under them (false positives or unlabelled objects), with their
mean confidence. Labels are the team's pseudo-labels on validation: measurement only, never training data.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import frame_numbers, load_annotations  # noqa: E402

W, H = 3840, 2160
BINS = ['hit', 'box', 'confused', 'held_back', 'unseen', 'never_born', 'lost']


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2-x1)*max(0., y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union > 0 else 0.


def gt_box(a):
    b = a.get('bbox') or a.get('bbox_source_xyxy') or a.get('box')
    return [float(v) for v in b]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log', required=True); ap.add_argument('--scene', default='validation')
    ap.add_argument('--classes', default='', help='comma list: print the per-frame bins of these classes')
    a = ap.parse_args()
    rows = {}
    for line in open(a.log, encoding='utf-8'):
        if line.strip():
            r = json.loads(line)
            if 'frame_index' in r:
                rows[r['frame_index']] = r
    frames = frame_numbers(a.scene)
    bins = collections.defaultdict(collections.Counter); called = collections.defaultdict(collections.Counter)
    held = collections.defaultdict(list); fp = collections.defaultdict(list); unanswered = 0
    seen_before = collections.defaultdict(list)          # class -> boxes answered (right class, IoU >= 0.3) so far
    trace = collections.defaultdict(list); want = {c for c in a.classes.split(',') if c}
    for k, frame in enumerate(frames):
        gts = [(g.get('object_id') or g.get('label'), gt_box(g)) for g in load_annotations(frame, a.scene)]
        r = rows.get(k)
        if r is None or not r.get('emitted'):
            unanswered += 1; continue
        answers = [(x['object_id'], [x['bbox'][0]*W, x['bbox'][1]*H, x['bbox'][2]*W, x['bbox'][3]*H], x.get('confidence', 0.)) for x in r.get('response', [])]
        rx1, ry1, rx2, ry2 = r['region']; s = (rx2-rx1)/960.
        raws = [(d['label'], [rx1+d['box'][0]*s, ry1+d['box'][1]*s, rx1+d['box'][2]*s, ry1+d['box'][3]*s], d.get('confidence', 0.)) for d in r.get('raw_detections', [])]
        for cls, g in gts:
            same = max((iou(g, b) for c, b, _ in answers if c == cls), default=0.)
            other = max(((iou(g, b), c) for c, b, _ in answers if c != cls), default=(0., None))
            cx, cy = (g[0]+g[2])/2, (g[1]+g[3])/2
            if same >= .5:
                kind = 'hit'
            elif same >= .1:
                kind = 'box'
            elif other[0] >= .3:
                kind = 'confused'; called[cls][other[1]] += 1
            elif rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                over = [(c, conf) for c, b, conf in raws if iou(g, b) >= .3]
                kind = 'held_back' if over else 'unseen'
                if over:
                    held[cls].append(max(over, key=lambda t: t[1]))
            else:
                kind = 'lost' if any(iou(g, b) >= .05 or abs((b[0]+b[2])/2-cx) < 150 for b in seen_before[cls]) else 'never_born'
            if same >= .3:
                seen_before[cls].append(g)
            bins[cls][kind] += 1
            if cls in want:
                trace[cls].append((k, kind, round(same, 2)))
        for c, b, conf in answers:
            if all(iou(b, g) < .3 for _, g in gts):
                fp[c].append((conf, k, round((b[0]+b[2])/2), round((b[1]+b[3])/2)))
    print(f'{len(rows)} logged frames, {unanswered} of {len(frames)} label frames unanswered')
    print(f"{'class':16}{'labels':>7}" + ''.join(f'{b:>11}' for b in BINS) + f"{'no-label answers':>18}")
    for cls in sorted(bins, key=lambda c: -sum(bins[c].values())):
        n = sum(bins[cls].values())
        f = fp.get(cls, [])
        print(f'{cls:16}{n:7d}' + ''.join(f'{100*bins[cls][b]/n:10.0f}%' for b in BINS) + f"{len(f):10d} @ {sum(x[0] for x in f)/max(1, len(f)):.2f}")
    for cls in sorted(called):
        print(f'  {cls} called:', dict(called[cls].most_common(3)))
    for cls in sorted(held):
        h = held[cls]; lab = collections.Counter(c for c, _ in h)
        print(f'  {cls} held back {len(h)}: raw labels {dict(lab.most_common(3))}, raw confidence mean {sum(c for _, c in h)/len(h):.2f}')
    for cls in sorted(fp):
        if cls not in bins:
            f = fp[cls]; print(f'  answers of unlabelled class {cls}: {len(f)} @ {sum(x[0] for x in f)/len(f):.2f}')
    for cls in want:
        print(cls, ' '.join(f'{k}:{kind[:2]}{s}' for k, kind, s in trace[cls]))


if __name__ == '__main__':
    main()
