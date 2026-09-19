"""Per-object failure analysis of the experts on the Higgsfield synthetic composites (Oscar, 20 Sep 2026).

Every composite pastes reviewed sprites (rotated by k quarter turns, optionally flipped, rescaled, then blended)
onto a background tile. For each pasted object the deployed pipeline (shared proposer -> expert -> gate -> verifier)
is run and the outcome is classified by the stage that lost it:
  found      accepted candidate with IoU >= .5
  verifier   a candidate with IoU >= .5 existed but the verifier called it background
  gate       ... rejected by the gate
  box        best candidate overlaps (IoU .2-.5): pose right, box wrong
  pose       a proposal lies within half the object size but no candidate reached IoU .2
  proposal   no proposal near the object at all
The report breaks the outcomes down per class, zoom, rotation (k), flip and scale bucket.

  python -m drone.experts.eval_synthetic --synthetic data/drone/synthetic-validation-sprites-20260918-v1 \
      --gates runs/pass9/gates.json --verifier runs/verifier-0052/model/best.pt --output runs/synth-val-experts
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .common import iou
from .registry import make_expert, load_gates, CLASSES
from .evaluate_all import class_families
from .gpu_proposer import SharedProposer


def outcome_for(target, accepted, candidates, proposals, long_side):
    """Classify one pasted object against the expert's accepted rows, all candidates and the proposals."""
    tb = target['bbox_xyxy']; tc = ((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2)
    if any(iou(r['bbox'], tb) >= .5 for r in accepted):
        return 'found', None
    with_box = [c for c in candidates if c.get('bbox')]
    good = [c for c in with_box if iou(c['bbox'], tb) >= .5]
    if good:
        reasons = Counter((next(iter(c['rejected_by'])) if isinstance(c.get('rejected_by'), dict) else c.get('rejected_by')) or 'accepted-other-class' for c in good)
        r = reasons.most_common(1)[0][0]
        return ('verifier' if r == 'verifier' else 'gate' if r == 'gate' else 'other:' + str(r)), reasons
    if any(iou(c['bbox'], tb) >= .2 for c in with_box):
        best = max(with_box, key=lambda c: iou(c['bbox'], tb))
        return 'box', dict(iou=round(iou(best['bbox'], tb), 2), pred=[round(v) for v in best['bbox']], target=[round(v) for v in tb], scale=best.get('fitted_scale'))
    near = [p for p in proposals if np.hypot(p['cx'] - tc[0], p['cy'] - tc[1]) <= .5 * long_side]
    if near:
        return 'pose', dict(nearest_score=round(max(p['proposer_score'] for p in near), 2), n_near=len(near))
    return 'proposal', dict(nearest_px=round(min((np.hypot(p['cx'] - tc[0], p['cy'] - tc[1]) for p in proposals), default=9999)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic', type=Path, required=True)
    p.add_argument('--bank', type=Path, default=Path('data/drone/expert-bank-20260918-v1'))
    p.add_argument('--gates', type=Path)
    p.add_argument('--verifier', type=Path)
    p.add_argument('--gpu', default='cuda:0')
    p.add_argument('--classes', default='')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.synthetic / 'manifest.json').read_text())
    records = manifest['records'][:a.limit] if a.limit else manifest['records']
    wanted = set(a.classes.split(',')) if a.classes else set(CLASSES)
    gates = load_gates(a.gates) if a.gates else None
    experts = {c: make_expert(c, a.bank, gates) for c in CLASSES if c in wanted}
    shared = SharedProposer(experts, device=a.gpu)
    verifier = None
    if a.verifier:
        from .verifier import Verifier
        verifier = Verifier(a.verifier, a.gpu)
    rows = []
    a.output.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(records):
        classes_here = sorted({t['class_name'] for t in r['annotations']} & set(experts))
        if not classes_here:
            continue
        image = cv2.imread(str(a.synthetic / r['file']))
        props = shared.propose_all(image, 1., r['zoom'])
        for c in classes_here:
            expert = experts[c]; family = class_families(c)[0]
            try:
                out = expert.detect(image, 1., r['zoom'], explain=True, proposals=props.get(c))
            except TypeError:
                out = expert.detect(image, 1., r['zoom'], explain=True)
            by_family, candidates = out if isinstance(out[0], dict) else ({family: out[0]}, out[2])
            accepted = list(by_family.get(family, []))
            if verifier is not None and accepted:
                annotated = verifier.annotate(image, accepted)
                kept = []
                for row in annotated:
                    if row['verifier_background'] >= .5:
                        row['rejected_by'] = 'verifier'
                    else:
                        kept.append(row)
                accepted = kept
            long_side = getattr(expert, 'long_side', 40.)
            for t in [t for t in r['annotations'] if t['class_name'] == c]:
                outcome, detail = outcome_for(t, accepted, candidates, props.get(c, []), long_side)
                paste = t.get('paste', {})
                rows.append(dict(record=r['id'], zoom=r['zoom'], class_name=c, outcome=outcome, detail=detail, k=paste.get('k'), flip=paste.get('flip'),
                                 scale=round(paste.get('scale', 1.), 2), sprite=t.get('sprite'), background_scene=r.get('background_scene'), size=[round(t['bbox_xyxy'][2] - t['bbox_xyxy'][0]), round(t['bbox_xyxy'][3] - t['bbox_xyxy'][1])]))
        if (i + 1) % 25 == 0:
            (a.output / 'rows.json').write_text(json.dumps(rows))
            print(f'{i + 1}/{len(records)} records, {len(rows)} objects', flush=True)
    (a.output / 'rows.json').write_text(json.dumps(rows))
    # report
    lines = ['| class | objects | found | verifier | gate | box | pose | proposal | other |', '|---|---|---|---|---|---|---|---|---|']
    by_class = defaultdict(list)
    for x in rows:
        by_class[x['class_name']].append(x)
    def share(xs, key):
        n = len(xs); return f"{sum(1 for x in xs if x['outcome'] == key)} ({100 * sum(1 for x in xs if x['outcome'] == key) / max(1, n):.0f}%)"
    for c in sorted(by_class):
        xs = by_class[c]
        other = sum(1 for x in xs if x['outcome'].startswith('other'))
        lines.append(f"| {c} | {len(xs)} | {share(xs, 'found')} | {share(xs, 'verifier')} | {share(xs, 'gate')} | {share(xs, 'box')} | {share(xs, 'pose')} | {share(xs, 'proposal')} | {other} |")
    lines.append('')
    for factor in ('zoom', 'k', 'flip'):
        lines.append(f'Found rate by {factor}: ' + ', '.join(f"{v}: {100 * sum(1 for x in xs if x['outcome'] == 'found') / len(xs):.0f}% ({len(xs)})" for v, xs in sorted(((v, [x for x in rows if x[factor] == v]) for v in {x[factor] for x in rows}), key=lambda kv: str(kv[0])) if xs))
    buckets = {'<0.9': lambda s: s < .9, '0.9-1.1': lambda s: .9 <= s <= 1.1, '>1.1': lambda s: s > 1.1}
    lines.append('Found rate by paste scale: ' + ', '.join(f"{name}: {100 * sum(1 for x in xs if x['outcome'] == 'found') / len(xs):.0f}% ({len(xs)})" for name, xs in ((n, [x for x in rows if f(x['scale'])]) for n, f in buckets.items()) if xs))
    lines.append('')
    lines.append('Per class, found rate by rotation k / flip: ' + '; '.join(f"{c}: " + ', '.join(f"k{k}{'f' if fl else ''} {100 * sum(1 for x in xs if x['outcome'] == 'found') / len(xs):.0f}%" for (k, fl), xs in sorted(((kf, [x for x in by_class[c] if (x['k'], x['flip']) == kf]) for kf in {(x['k'], x['flip']) for x in by_class[c]}), key=lambda kv: str(kv[0])) if xs) for c in sorted(by_class)))
    report = '\n'.join(lines)
    (a.output / 'report.md').write_text(report + '\n')
    print(report)


if __name__ == '__main__':
    main()
