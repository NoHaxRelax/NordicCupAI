"""Score one expert on grid crops: held-out targets, proposals, and false alarms on empty tiles.

Each branch is scored separately (the expert's own output and the SIFT comparison
branch). Targets are the organiser-style boxes of the requested class; fully
contained targets and edge-cut targets are reported apart. Unmatched proposals
on positive crops are listed, not counted as false alarms, because validation
labels are partial. Empty tiles are verified-empty training tiles.
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .common import iou, sha
from .registry import make_expert, families as class_families, CLASSES, load_gates

ROOT = Path(__file__).resolve().parents[2]


def match(proposals, targets, threshold=.5):
    """One-to-one greedy matching by IoU; returns matched target indices and unmatched proposals."""
    used, matched, unmatched = set(), {}, []
    for p in sorted(proposals, key=lambda r: -r['score']):
        best, best_iou = None, threshold
        for i, t in enumerate(targets):
            if i in used:
                continue
            v = iou(p['bbox'], t['bbox_xyxy'])
            if v >= best_iou:
                best, best_iou = i, v
        if best is None:
            unmatched.append(p)
        else:
            used.add(best)
            matched[best] = dict(proposal=p, iou=best_iou)
    return matched, unmatched


def run(expert, image, zoom, families):
    """Normalise the two detect() return shapes to (rows_by_family, candidates)."""
    out = expert.detect(image, 1., zoom, explain=True)
    if isinstance(out[0], dict):
        by_family, candidates = out
    else:
        accepted, sift_rows, candidates = out
        by_family = {families[0]: accepted, 'sift': sift_rows}
    return {f: by_family.get(f, []) for f in families}, candidates


def draw(image, targets, rows_by_family, path):
    canvas = image.copy()
    for t in targets:
        x1, y1, x2, y2 = map(int, t['bbox_xyxy'])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
    colours = {'hangar_expert': (0, 0, 255), 'condor_expert': (0, 0, 255), 'expert': (0, 0, 255), 'pixel': (0, 200, 255), 'sift': (255, 128, 0)}
    for family, rows in rows_by_family.items():
        for r in rows:
            x1, y1, x2, y2 = map(int, r['bbox'])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colours.get(family, (255, 255, 255)), 1)
            cv2.putText(canvas, f"{family[:5]} {r['score']:.2f}", (x1, max(10, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, .35, colours.get(family, (255, 255, 255)), 1)
    cv2.imwrite(str(path), canvas)


def main():
    import os
    cv2.setNumThreads(int(os.environ.get('CV_THREADS', '2')))  # six processes x all cores thrashed the pod
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--class', dest='class_name', required=True, choices=CLASSES)
    p.add_argument('--bank', type=Path, default=ROOT / 'data/drone/expert-bank-20260918-v1')
    p.add_argument('--grid', type=Path, default=ROOT / 'data/drone/grid384-20260918-v1')
    p.add_argument('--split', default='dev', help='split whose positive crops are scored')
    p.add_argument('--tracks', nargs='*', help='restrict positives to these track ids')
    p.add_argument('--max-empty', type=int, help='cap on empty tiles for a quick pass')
    p.add_argument('--gates', type=Path, help='fitted gates json; candidates below the gate are recorded as rejected_by=gate')
    p.add_argument('--zooms', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--empty-split', default='train', help='split whose verified-empty tiles measure false alarms')
    p.add_argument('--no-empty', action='store_true')
    p.add_argument('--limit', type=int)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.grid / 'manifest.json').read_text())
    expert = make_expert(a.class_name, a.bank, load_gates(a.gates) if a.gates else None)
    families = class_families(a.class_name)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'overlays').mkdir()
    positives = [r for r in manifest['records'] if r['split'] == a.split and r['zoom'] in a.zooms
                 and any(x['class_name'] == a.class_name and (not a.tracks or x['track_id'] in a.tracks) for x in r['annotations'])]
    if a.limit:
        positives = positives[:a.limit]
    rows, totals = [], {f: Counter() for f in families}
    started = time.time()
    for r in positives:
        image = cv2.imread(str(a.grid / r['file']))
        if sha(a.grid / r['file']) != r['sha256']:
            raise ValueError('Crop hash mismatch: ' + r['file'])
        targets = [x for x in r['annotations'] if x['class_name'] == a.class_name and (not a.tracks or x['track_id'] in a.tracks)]
        by_family, candidates = run(expert, image, r['zoom'], families)
        row = dict(id=r['id'], zoom=r['zoom'], frame=r['frame'], targets=[dict(bbox=t['bbox_xyxy'], complete=t['fully_contained'], track=t['track_id']) for t in targets],
                   candidates=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()} for c in candidates], results={})
        for family, proposals in by_family.items():
            matched, unmatched = match(proposals, targets)
            for i, t in enumerate(targets):
                kind = 'complete' if t['fully_contained'] else 'partial'
                totals[family][f'{kind}_targets'] += 1
                totals[family][f'{kind}_targets_zoom{r["zoom"]}'] += 1
                if i in matched:
                    totals[family][f'{kind}_matched'] += 1
                    totals[family][f'{kind}_matched_zoom{r["zoom"]}'] += 1
            totals[family]['proposals'] += len(proposals)
            totals[family]['unmatched_proposals'] += len(unmatched)
            row['results'][family] = dict(proposals=[dict(bbox=q['bbox'], score=q['score']) for q in proposals],
                                          matched={str(i): dict(iou=m['iou'], score=m['proposal']['score']) for i, m in matched.items()},
                                          unmatched=len(unmatched))
        draw(image, targets, by_family, a.output / 'overlays' / f"{r['id']}.png")
        rows.append(row)
        if len(rows) % 20 == 0:
            (a.output / 'partial.json').write_text(json.dumps(dict(class_name=a.class_name, crops=rows, done=len(rows), of=len(positives)), default=str))
    empty_rows, empty_totals = [], {f: Counter() for f in families}
    if not a.no_empty:
        empties = [r for r in manifest['records'] if r['split'] == a.empty_split and r['kind'] != 'positive']
        if a.max_empty:
            empties = empties[:a.max_empty]
        for r in empties:
            image = cv2.imread(str(a.grid / r['file']))
            if (a.grid / r['label']).read_text().strip() if 'label' in r else False:
                raise ValueError('Empty tile has labels: ' + r['file'])
            by_family, candidates = run(expert, image, r['zoom'], families)
            for family, proposals in by_family.items():
                empty_totals[family]['tiles'] += 1
                empty_totals[family]['false_alarms'] += len(proposals)
                empty_totals[family]['tiles_with_alarm'] += bool(proposals)
            reasons = Counter(c.get('rejected_by') or 'accepted' for c in candidates)
            empty_rows.append(dict(id=r['id'], zoom=r['zoom'], alarms={f: len(v) for f, v in by_family.items()}, candidate_reasons=dict(reasons),
                                   candidates=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()} for c in candidates]))
            if any(by_family.values()):
                draw(image, [], by_family, a.output / 'overlays' / f"empty-{r['id']}.png")
    summary = {}
    for family in families:
        t, e = totals[family], empty_totals[family]
        summary[family] = dict(complete=f"{t['complete_matched']}/{t['complete_targets']}", partial=f"{t['partial_matched']}/{t['partial_targets']}",
                               per_zoom={z: f"{t[f'complete_matched_zoom{z}']}/{t[f'complete_targets_zoom{z}']}" for z in a.zooms},
                               proposals=t['proposals'], unmatched_proposals=t['unmatched_proposals'],
                               empty_tiles=e['tiles'], false_alarms=e['false_alarms'], tiles_with_alarm=e['tiles_with_alarm'])
    report = dict(class_name=a.class_name, split=a.split, tracks=a.tracks, zooms=a.zooms, empty_split=None if a.no_empty else a.empty_split,
                  bank_sha256=sha(a.bank / 'manifest.json'), grid_manifest_sha256=sha(a.grid / 'manifest.json'),
                  settings={k: str(v) for k, v in vars(getattr(expert, 'settings', None) or getattr(expert, 'spec')).items()}, positives=len(positives), seconds=time.time() - started,
                  summary=summary, crops=rows, empty=empty_rows,
                  note='Targets are participant/organiser boxes; unmatched proposals on positive crops are not false alarms because labels are partial.')
    (a.output / 'report.json').write_text(json.dumps(report, indent=1, default=lambda v: float(v) if isinstance(v, (np.floating, np.integer)) else str(v)))
    print(json.dumps(dict(positives=len(positives), seconds=round(report['seconds'], 1), summary=summary), indent=1))


if __name__ == '__main__':
    main()
