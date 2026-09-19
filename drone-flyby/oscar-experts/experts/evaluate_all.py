"""Evaluate every class on the same tiles in one process: shared GPU proposer, threaded fine pose, cached proposals.

Writes one evaluate.py-compatible report per class under OUTPUT/<class>/, so report_all.py, fit_gates.py
and crops.py work unchanged. Proposals are cached on disk per (tile, zoom, kernel signature), so a rerun
that changes only signatures, gates or the fine pose skips the proposer entirely.
"""
import argparse
import hashlib
import json
import sys
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .common import iou, sha
from .evaluate import match, draw
from .exclusions import Exclusions
from .registry import make_expert, families as class_families, CLASSES, load_gates


DELIVERED_SCALE = {0: .25, 1: .5, 2: 1.}  # delivered view pixels per native pixel: L0 views are 1/4 scale, L1 1/2, L2 native


def rescale(rows, k):
    """Bring boxes and centres found on a downscaled image back to native tile coordinates."""
    for q in rows:
        for f in ('bbox', 'mask_bbox'):
            if q.get(f):
                q[f] = [float(v) * k for v in q[f]]
        for f in ('cx', 'cy'):
            if isinstance(q.get(f), (int, float)):
                q[f] = float(q[f]) * k
    return rows


def run_expert(expert, family, image, zoom, proposals, scale=1.):
    try:
        try:
            out = expert.detect(image, scale, zoom, explain=True, proposals=proposals) if proposals is not None else expert.detect(image, scale, zoom, explain=True)
        except TypeError:
            out = expert.detect(image, scale, zoom, explain=True)
    except Exception:  # one broken candidate must never take a whole shard down: record it, keep going
        import traceback
        print(f'EXPERT_ERROR {family} zoom={zoom}\n{traceback.format_exc()}', file=sys.stderr, flush=True)
        return {family: [], 'sift': []}, [dict(error=traceback.format_exc()[-400:])]
    if isinstance(out[0], dict):
        by_family, candidates = out
    else:
        accepted, sift_rows, candidates = out
        by_family = {family: accepted, 'sift': sift_rows}
    return by_family, candidates


def main():
    cv2.setNumThreads(int(os.environ.get('CV_THREADS', '2')))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--classes', nargs='+', default=CLASSES)
    p.add_argument('--bank', type=Path, default=Path('data/drone/expert-bank-20260918-v1'))
    p.add_argument('--grid', type=Path, default=Path('data/drone/grid384-20260918-v1'))
    p.add_argument('--split', default='train')
    p.add_argument('--zooms', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--max-empty', type=int, default=60)
    p.add_argument('--gates', type=Path)
    p.add_argument('--scales', default=None, help='comma-separated run scale per zoom L0,L1,L2 (e.g. 0.5,1,1); overrides --delivered')
    p.add_argument('--delivered', action='store_true', help='run the experts on delivered-resolution pixels (L0 x1/4, L1 x1/2) with templates scaled to match; results are rescaled to native for matching')
    p.add_argument('--verifier', type=Path, help='trained verifier; candidates it calls background are recorded as rejected_by=verifier')
    p.add_argument('--verifier-threshold', type=float, default=.5)
    p.add_argument('--gpu', default='cuda:0')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--cache', type=Path, default=Path('/workspace/experts/cache/proposals'))
    p.add_argument('--limit', type=int)
    p.add_argument('--sample', type=int, help='tiles per class per zoom, stratified by frame; use with --seed that rotates per iteration')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.grid / 'manifest.json').read_text())
    exclusions = Exclusions()
    gates = load_gates(a.gates) if a.gates else None
    experts = {c: make_expert(c, a.bank, gates) for c in a.classes}
    from .gpu_proposer import SharedProposer
    shared = SharedProposer({n: e for n, e in experts.items()}, device=a.gpu)
    verifier = None
    if a.verifier:
        from .verifier import Verifier
        verifier = Verifier(a.verifier, a.gpu)
    from .gpu_proposer import PROPOSER_VERSION
    signature = hashlib.sha256(json.dumps(dict(version=PROPOSER_VERSION, kernels={n: [(t.id, t.zoom) for t in e.proposer_templates_for(z)] + [list(e.proposer.headings), e.proposer.downscale, e.proposer.blur, e.proposer.threshold]
                                            for n, e in shared.experts.items() for z in (0, 1, 2)}), sort_keys=True, default=str).encode()).hexdigest()[:12]
    a.cache.mkdir(parents=True, exist_ok=True)
    a.output.mkdir(parents=True, exist_ok=False)
    for c in a.classes:
        (a.output / c / 'overlays').mkdir(parents=True)
    records = [r for r in manifest['records'] if r['split'] == a.split and r['zoom'] in a.zooms and not exclusions.tile_has_excluded(r)]
    positives = [r for r in records if any(x['class_name'] in a.classes for x in r['annotations'])]
    empties = [r for r in manifest['records'] if r['split'] == 'train' and r['kind'] != 'positive'][:a.max_empty]
    if a.sample:
        # Rotating stratified sample: per class and zoom, spread over frames, different tiles for each seed,
        # so quick iterations never tune to one fixed set of frames (Oscar, 19 Sep 03:00).
        rng = np.random.default_rng(a.seed)
        chosen = {}
        for c in a.classes:
            for z in a.zooms:
                pool = [r for r in positives if r['zoom'] == z and any(x['class_name'] == c for x in r['annotations'])]
                by_frame = defaultdict(list)
                for r in pool:
                    by_frame[r['frame']].append(r)
                frames = list(by_frame)
                rng.shuffle(frames)
                picked = []
                while len(picked) < min(a.sample, len(pool)):
                    for f in frames:
                        if by_frame[f]:
                            picked.append(by_frame[f].pop(int(rng.integers(len(by_frame[f])))))
                            if len(picked) >= min(a.sample, len(pool)):
                                break
                for r in picked:
                    chosen[r['id']] = r
        positives = list(chosen.values())
        rng.shuffle(empties)
    if a.limit:
        positives = positives[:a.limit]
    if a.shards > 1:
        positives = positives[a.shard::a.shards]
        empties = empties[a.shard::a.shards]
    per_class = {c: dict(rows=[], empty=[], totals=defaultdict(Counter), empty_totals=defaultdict(Counter)) for c in a.classes}
    pool = ThreadPoolExecutor(max_workers=a.workers)
    started = time.time(); proposer_seconds = 0.; cached = 0
    for kind, tiles in (('positive', positives), ('empty', empties)):
        for r in tiles:
            image = cv2.imread(str(a.grid / r['file']))
            factor = float(a.scales.split(',')[r['zoom']]) if a.scales else (DELIVERED_SCALE[r['zoom']] if a.delivered else 1.)
            run_image = image if factor == 1. else cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            key = a.cache / f"{signature}-{r['id']}{'' if factor == 1. else f'-d{factor}'}.json"
            if key.exists():
                shared_props = json.loads(key.read_text()); cached += 1
            else:
                t0 = time.time(); shared_props = shared.propose_all(run_image, factor, r['zoom']); proposer_seconds += time.time() - t0
                key.write_text(json.dumps(shared_props))
            futures = {c: pool.submit(run_expert, experts[c], class_families(c)[0], run_image, r['zoom'], shared_props.get(c), factor) for c in a.classes}
            for c in a.classes:
                by_family, candidates = futures[c].result()
                if factor != 1.:
                    # accepted rows and the candidate list share dict objects: scale each distinct one exactly once
                    seen = {}
                    for rows_ in list(by_family.values()) + [candidates]:
                        for q in rows_:
                            seen.setdefault(id(q), q)
                    rescale(list(seen.values()), 1. / factor)
                fams = class_families(c)
                by_family = {f: by_family.get(f, []) for f in fams}
                if verifier is not None and by_family.get(fams[0]):
                    annotated = verifier.annotate(image, by_family[fams[0]])
                    kept = []
                    for row in annotated:
                        row['verifier_reject'] = bool(row['verifier_background'] >= a.verifier_threshold)
                        if row['verifier_reject']:
                            row['rejected_by'] = 'verifier'
                        else:
                            kept.append(row)
                    by_family[fams[0]] = kept
                state = per_class[c]
                targets = [x for x in r['annotations'] if x['class_name'] == c] if kind == 'positive' else []
                if kind == 'positive' and not targets:
                    continue
                clean = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cand.items()} for cand in candidates]
                if kind == 'positive':
                    row = dict(id=r['id'], zoom=r['zoom'], frame=r['frame'], targets=[dict(bbox=t['bbox_xyxy'], complete=t['fully_contained'], track=t['track_id']) for t in targets], candidates=clean, results={})
                    for family, proposals in by_family.items():
                        matched, unmatched = match(proposals, targets)
                        for i, t in enumerate(targets):
                            k = 'complete' if t['fully_contained'] else 'partial'
                            state['totals'][family][f'{k}_targets'] += 1; state['totals'][family][f"{k}_targets_zoom{r['zoom']}"] += 1
                            if i in matched:
                                state['totals'][family][f'{k}_matched'] += 1; state['totals'][family][f"{k}_matched_zoom{r['zoom']}"] += 1
                        state['totals'][family]['proposals'] += len(proposals); state['totals'][family]['unmatched_proposals'] += len(unmatched)
                        row['results'][family] = dict(proposals=[dict(bbox=q['bbox'], score=q['score']) for q in proposals], matched={str(i): dict(iou=m['iou'], score=m['proposal']['score']) for i, m in matched.items()}, unmatched=len(unmatched))
                    draw(image, targets, by_family, a.output / c / 'overlays' / f"{r['id']}.png")
                    state['rows'].append(row)
                else:
                    for family, proposals in by_family.items():
                        state['empty_totals'][family]['tiles'] += 1; state['empty_totals'][family]['false_alarms'] += len(proposals); state['empty_totals'][family]['tiles_with_alarm'] += bool(proposals)
                    state['empty'].append(dict(id=r['id'], zoom=r['zoom'], alarms={f: len(v) for f, v in by_family.items()}, candidate_reasons=dict(Counter(x.get('rejected_by') or 'accepted' for x in candidates)), candidates=clean))
    seconds = time.time() - started
    for c in a.classes:
        state = per_class[c]; fams = class_families(c); summary = {}
        for family in fams:
            t, e = state['totals'][family], state['empty_totals'][family]
            summary[family] = dict(complete=f"{t['complete_matched']}/{t['complete_targets']}", partial=f"{t['partial_matched']}/{t['partial_targets']}", per_zoom={z: f"{t[f'complete_matched_zoom{z}']}/{t[f'complete_targets_zoom{z}']}" for z in a.zooms}, proposals=t['proposals'], unmatched_proposals=t['unmatched_proposals'], empty_tiles=e['tiles'], false_alarms=e['false_alarms'], tiles_with_alarm=e['tiles_with_alarm'])
        report = dict(class_name=c, split=a.split, zooms=a.zooms, empty_split='train', bank_sha256=sha(a.bank / 'manifest.json'), grid_manifest_sha256=sha(a.grid / 'manifest.json'), settings={k: str(v) for k, v in vars(getattr(experts[c], 'settings')).items()}, positives=len(state['rows']), seconds=seconds / len(a.classes), summary=summary, crops=state['rows'], empty=state['empty'], excluded_rules=exclusions.rules, note='evaluate_all: shared GPU proposer; seconds is the wall time divided by the number of classes.')
        (a.output / c / 'report.json').write_text(json.dumps(report, indent=1, default=lambda v: float(v) if isinstance(v, (np.floating, np.integer)) else str(v)))
    (a.output / 'timing.json').write_text(json.dumps(dict(tiles=len(positives) + len(empties), seconds=seconds, proposer_seconds=proposer_seconds, cached_tiles=cached, per_tile=seconds / max(1, len(positives) + len(empties)))))
    print(json.dumps(dict(tiles=len(positives) + len(empties), seconds=round(seconds), proposer_seconds=round(proposer_seconds), cached=cached, per_tile=round(seconds / max(1, len(positives) + len(empties)), 2))))


if __name__ == '__main__':
    main()
