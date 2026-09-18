"""Analyze an endpoint diagnostics log against a scene's labels.

Reports, per recognition family and zoom level, how many raw detector boxes
matched a label (same class, IoU >= 0.50 in source pixels) and how many did
not, with their confidences; per-class raw recall on the views the camera
actually delivered; and the same for the final responses. Labels may be
incomplete, so unmatched boxes are an upper bound on false alarms; the
--top-band option restricts the false-alarm count to the region the label
review covered (the top 540 source rows after frame 5).

    python analyze_run_log.py logs/val-v7 --scene validation --top-band
"""
import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def iou(a, b):
    ix = max(0, min(a[2], b[2])-max(a[0], b[0])); iy = max(0, min(a[3], b[3])-max(a[1], b[1]))
    inter = ix*iy; union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union > 0 else 0.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log_dir')
    p.add_argument('--scene', default='validation')
    p.add_argument('--top-band', action='store_true', help='Count false alarms only inside the reviewed top 540 rows (frame 5: whole frame)')
    p.add_argument('--birth', type=float, default=.6)
    a = p.parse_args()
    rows = []
    for path in sorted(glob.glob(str(Path(a.log_dir)/'*.jsonl'))):
        rows += [json.loads(l) for l in open(path)]
    labels = {}
    for r in rows:
        f = r['frame']
        if f not in labels:
            path = HERE/'src'/a.scene/'annotations'/f'frame_{f:06d}.json'
            labels[f] = [(x['object_id'], x['bbox']) for x in json.loads(path.read_text())['annotations']] if path.exists() else []
    def reviewed(frame, box):
        if not a.top_band or frame == 5:
            return True
        return box[1] < 540
    fam = defaultdict(lambda: {'tp': [], 'fp': []}); recall = Counter(); visible = Counter(); resp_tp = 0; resp_fp = 0; resp_n = 0
    resp_class = defaultdict(lambda: [0, 0])
    for r in rows:
        f = r['frame']; x1, y1, x2, y2 = r['region']; sx = (x2-x1)/960; sy = (y2-y1)/540; level = r['level']
        vis = []
        for c, b in labels[f]:
            v = [max(b[0], x1), max(b[1], y1), min(b[2], x2), min(b[3], y2)]
            if v[2]-v[0] > 1 and v[3]-v[1] > 1:
                vis.append((c, v)); visible[c] += 1
        used = set()
        for d in r.get('raw_detections') or []:
            b = d['box']; src = [x1+b[0]*sx, y1+b[1]*sy, x1+b[2]*sx, y1+b[3]*sy]
            key = (d.get('family') or 'detector', level)
            best = max(((iou(src, v), i) for i, (c, v) in enumerate(vis) if c == d['label'] and i not in used), default=(0, None))
            if best[0] >= .5:
                fam[key]['tp'].append(d['confidence']); used.add(best[1]); recall[d['label']] += 1
            elif reviewed(f, src):
                fam[key]['fp'].append(d['confidence'])
        # Final response boxes against the full-frame labels.
        full = labels[f]; used = set()
        for ann in r.get('response') or []:
            src = [ann['bbox'][0]*3840, ann['bbox'][1]*2160, ann['bbox'][2]*3840, ann['bbox'][3]*2160]
            resp_n += 1
            best = max(((iou(src, b), i) for i, (c, b) in enumerate(full) if c == ann['object_id'] and i not in used), default=(0, None))
            if best[0] >= .5:
                resp_tp += 1; used.add(best[1]); resp_class[ann['object_id']][0] += 1
            else:
                resp_fp += 1; resp_class[ann['object_id']][1] += 1
    print('%-22s %2s %5s %5s  %-12s %-12s'%('family', 'L', 'TP', 'FP', 'TPconf med/min', 'FP>=birth'))
    for (family, level), v in sorted(fam.items()):
        tp, fp = v['tp'], v['fp']
        print('%-22s %2d %5d %5d  %-12s %-12s'%(family, level, len(tp), len(fp),
              ('%.2f/%.2f'%(np.median(tp), min(tp))) if tp else '-', str(sum(c >= a.birth for c in fp))))
    print('\nraw detector recall on delivered views (labelled boxes visible in the view):')
    for c in sorted(visible):
        print('  %-16s %4d / %4d'%(c, recall[c], visible[c]))
    total_labels = sum(len(v) for v in labels.values())
    print('\nfinal responses: %d boxes, %d matched a label (of %d labels over %d frames), %d unmatched'%(resp_n, resp_tp, total_labels, len(labels), resp_fp))
    for c, (tp, fp) in sorted(resp_class.items()):
        print('  %-16s matched %4d  unmatched %4d'%(c, tp, fp))
    times = [r['total_ms'] for r in rows]
    print('\nframes %d, total ms median %.0f max %.0f, detector ms median %.0f'%(len(rows), np.median(times), max(times), np.median([r['detector_ms'] for r in rows])))
    ev = Counter(e['event'] for r in rows for e in (r['events'] or []))
    print('events', dict(ev))


if __name__ == '__main__':
    main()
