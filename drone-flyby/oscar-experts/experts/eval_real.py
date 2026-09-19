"""Stage-by-stage failure analysis of the experts on REAL validation-scene frames (Oscar, 19 Sep 2026).

For every labelled object of the requested classes, an L1 view is centred on it the way the camera would serve
it (1920x1080 source crop, delivered at 960x540, upsampled x2 as the live detector does), the deployed pipeline
(shared proposer -> expert -> gate -> verifier) is run on that view, and the outcome is classified by the stage
that lost the object (same classes of outcome as eval_synthetic: found / verifier / gate / box / pose / proposal).

  python -m drone.experts.eval_real --scene /workspace/live-expert/drone-flyby/src/validation \
      --classes large_tower,small_tower,mine_roller,medium_launcher --gates runs/pass9/gates.json \
      --verifier runs/verifier-0052/model/best.pt --output runs/real-val-zero
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .registry import make_expert, load_gates, CLASSES
from .evaluate_all import class_families
from .gpu_proposer import SharedProposer
from .eval_synthetic import outcome_for

W, H = 3840, 2160
VIEW_W, VIEW_H = 1920, 1080   # L1 source footprint; delivered at 960x540


def render_view(frame, cx, cy, down, up):
    x1 = int(np.clip(cx - VIEW_W / 2, 0, W - VIEW_W)); y1 = int(np.clip(cy - VIEW_H / 2, 0, H - VIEW_H))
    crop = frame[y1:y1 + VIEW_H, x1:x1 + VIEW_W]
    delivered = cv2.resize(crop, (VIEW_W // 2, VIEW_H // 2), interpolation=down)
    native = cv2.resize(delivered, (VIEW_W, VIEW_H), interpolation=up)
    return native, x1, y1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--bank', type=Path, default=Path('data/drone/expert-bank-20260918-v1'))
    p.add_argument('--gates', type=Path)
    p.add_argument('--verifier', type=Path)
    p.add_argument('--gpu', default='cuda:0')
    p.add_argument('--classes', required=True)
    p.add_argument('--every', type=int, default=10, help='use every n-th frame')
    p.add_argument('--limit', type=int, default=20, help='objects per class')
    p.add_argument('--down', default='area', choices=['area', 'linear'])
    p.add_argument('--tight-box', type=float, default=None, help='replace every candidate box by the posed MASK extent grown by this fraction per side (tests the box convention)')
    p.add_argument('--max-candidates', type=int, default=None, help='override ClassSpec.max_candidates (tests the candidate cut)')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    wanted = [c for c in a.classes.split(',') if c]
    gates = load_gates(a.gates) if a.gates else None
    experts = {c: make_expert(c, a.bank, gates) for c in CLASSES if c in wanted}
    if a.max_candidates:
        import dataclasses
        for name, e in experts.items():
            try:
                e.spec.max_candidates = a.max_candidates
            except Exception:
                try:
                    e.spec = dataclasses.replace(e.spec, max_candidates=a.max_candidates)
                except Exception:
                    print('no spec on', name, '(cap unchanged)')
    shared = SharedProposer(experts, device=a.gpu)
    verifier = None
    if a.verifier:
        from .verifier import Verifier
        verifier = Verifier(a.verifier, a.gpu)
    down = cv2.INTER_AREA if a.down == 'area' else cv2.INTER_LINEAR
    # targets: (frame, class, bbox) sampled every n-th frame, up to --limit per class
    targets = defaultdict(list)
    for f in sorted((a.scene / 'annotations').glob('frame_*.json')):
        n = int(f.stem.split('_')[1])
        if n % a.every:
            continue
        for t in json.loads(f.read_text())['annotations']:
            if t['object_id'] in experts and len(targets[t['object_id']]) < a.limit:
                targets[t['object_id']].append((n, t['bbox']))
    rows = []
    a.output.mkdir(parents=True, exist_ok=True)
    frames = {}
    for c, items in targets.items():
        expert = experts[c]; family = class_families(c)[0]; long_side = getattr(expert, 'long_side', 40.)
        for n, bb in items:
            if n not in frames:
                frames = {n: cv2.imread(str(a.scene / 'images' / f'frame_{n:06d}.png'))}
            view, x1, y1 = render_view(frames[n], (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2, down, cv2.INTER_LINEAR)
            tb = [bb[0] - x1, bb[1] - y1, bb[2] - x1, bb[3] - y1]
            props = shared.propose_all(view, 1., 1, classes=[c])
            try:
                out = expert.detect(view, 1., 1, explain=True, proposals=props.get(c))
            except TypeError:
                out = expert.detect(view, 1., 1, explain=True)
            by_family, candidates = out if isinstance(out[0], dict) else ({family: out[0]}, out[2])
            accepted = list(by_family.get(family, []))
            if a.tight_box is not None:
                seen = set()
                for cand in list(candidates) + accepted:  # do not shadow the class name c
                    mb = cand.get('mask_bbox')
                    if mb and id(cand) not in seen:
                        seen.add(id(cand)); w, h = mb[2] - mb[0], mb[3] - mb[1]; m = a.tight_box
                        cand['bbox'] = [mb[0] - m * w, mb[1] - m * h, mb[2] + m * w, mb[3] + m * h]
            if verifier is not None and accepted:
                kept = []
                for row in verifier.annotate(view, accepted):
                    if row['verifier_background'] >= .5:
                        row['rejected_by'] = 'verifier'
                    else:
                        kept.append(row)
                accepted = kept
            outcome, detail = outcome_for({'bbox_xyxy': tb}, accepted, candidates, props.get(c, []), long_side)
            best = max(candidates, key=lambda r: r.get('score', 0.), default=None)
            rows.append(dict(frame=n, class_name=c, outcome=outcome, detail=detail, size=[bb[2] - bb[0], bb[3] - bb[1]],
                             n_proposals=len(props.get(c, [])), n_candidates=len(candidates), n_accepted=len(accepted),
                             best_score=round(float(best.get('score', 0.)), 3) if best else None))
            print(c, n, outcome, detail, 'props', len(props.get(c, [])), 'cands', len(candidates), 'acc', len(accepted), flush=True)
    (a.output / 'rows.json').write_text(json.dumps(rows))
    lines = ['| class | objects | found | verifier | gate | box | pose | proposal | other |', '|---|---|---|---|---|---|---|---|---|']
    by_class = defaultdict(list)
    for x in rows:
        by_class[x['class_name']].append(x)
    def share(xs, key):
        k = sum(1 for x in xs if x['outcome'] == key); return f'{k} ({100 * k / max(1, len(xs)):.0f}%)'
    for c in sorted(by_class):
        xs = by_class[c]
        lines.append(f"| {c} | {len(xs)} | {share(xs, 'found')} | {share(xs, 'verifier')} | {share(xs, 'gate')} | {share(xs, 'box')} | {share(xs, 'pose')} | {share(xs, 'proposal')} | {sum(1 for x in xs if x['outcome'].startswith('other'))} |")
    report = '\n'.join(lines)
    (a.output / 'report.md').write_text(report + '\n')
    print(report)


if __name__ == '__main__':
    main()
