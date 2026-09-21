#!/usr/bin/env python3
"""Final-member tile datasets (Phase 2: validation sprites allowed, Oscar 20 Sep 08:45) with the data agent's HOLD-OUT rule applied.

    build_ds_final.py OUT_NAME --sets flyover-train-v1 flyover-val-v1 [flypaste-adj-train-v1 ...] [--syn]
                      [--neg DIR:MAXFRAC ...] [--heldout HELDOUT.json]

- A tile is dropped if ANY of its objects uses a held-out sprite view: validation bank = (track, frame) in held_out,
  train bank = (class, reference frame) in held_out_train_bank.views. synthetic-train objects carry the reference frame in
  'sprite' ("inst:reference-f000006-..."); flyover / flypaste / adjacency objects carry sprite_track + sprite_view_frame.
- File names are prefixed with the set name, so sets that share tile names cannot overwrite each other's labels.
- --neg DIR:MAXFRAC adds background-only tiles (DIR/images/*.png|jpg, empty label) capped at MAXFRAC of the positive pool; all
  negatives together are capped at 15 % (the m-fly-neg run at 20 % background collapsed small_tower).
Val split = 5 % of the pool, monitoring only.
"""
import argparse, json, os, random, re
from pathlib import Path
CLASSES = ['condor','hangar','helicopter','jammer','jet_plane','large_launcher','large_tower','medium_launcher','medium_plane',
           'mine_roller','small_launcher','small_plane','small_tower','spacecraft','ta-ta','tank']
ROOT = Path('/root/data/flyover-synth-20260919'); SYN = Path('/root/data/synthetic-train-realistic-20260919-v1')
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--sets', nargs='+', required=True); ap.add_argument('--syn', action='store_true')
ap.add_argument('--neg', nargs='*', default=[]); ap.add_argument('--sub', nargs='*', default=[], help='SET:MAXFRAC keep at most MAXFRAC x (size of the other positives) tiles of that set'); ap.add_argument('--heldout', default='/root/HELDOUT-VIEWS.json'); a = ap.parse_args()
h = json.load(open(a.heldout))
HV = {(v['track'], int(v['frame'])) for v in h['held_out']}
HT = {(v['class_name'], int(v['frame'])) for v in h['held_out_train_bank']['views']}
HCLS = {c for c, _ in HT} | {v['class_name'] for v in h['held_out']}
def held(an):
    if an.get('sprite_track') is None and an.get('sprite_view_frame') is None and not an.get('sprite') and an.get('class_name') in HCLS and 'sprite_track' in an:
        return True      # strict: unknown sprite view of a held-out class counts as held-out
    t, f = an.get('sprite_track'), an.get('sprite_view_frame')
    if t is not None and f is not None:
        if (t, int(f)) in HV: return True
        if str(t).startswith('reference') and (an['class_name'], int(f)) in HT: return True
    m = re.search(r'reference-f0*(\d+)-', str(an.get('sprite', '')))
    return bool(m and (an['class_name'], int(m.group(1))) in HT)
random.seed(7); items = []; dropped = {}; SUB = {x.rsplit(':', 1)[0]: float(x.rsplit(':', 1)[1]) for x in a.sub}; late = {}
for name in a.sets:
    fly = ROOT/name; m = json.load(open(fly/'manifest.json')); assert m['classes'] == CLASSES, name
    for r in m['records']:
        if any(held(an) for an in r.get('annotations', [])):
            dropped[name] = dropped.get(name, 0) + 1; continue
        lab = fly/'labels'/(Path(r['file']).stem + '.txt')
        (late.setdefault(name, []) if name in SUB else items).append((fly/r['file'], lab.read_text() if lab.exists() else '', f"{name}__{Path(r['file']).name}"))
if a.syn:
    s = json.load(open(SYN/'manifest.json')); assert s['classes'] == CLASSES
    for r in s['records']:
        if any(held(an) for an in r['annotations']):
            dropped['synthetic-train'] = dropped.get('synthetic-train', 0) + 1; continue
        side = r.get('input_size', 256); lines = []
        for an in r['annotations']:
            x1, y1, x2, y2 = an['bbox_xyxy']; x1, y1, x2, y2 = max(0., x1), max(0., y1), min(side, x2), min(side, y2)
            if x2 - x1 >= 1 and y2 - y1 >= 1:
                lines.append(f"{an['class_id']} {(x1+x2)/2/side:.6f} {(y1+y2)/2/side:.6f} {(x2-x1)/side:.6f} {(y2-y1)/side:.6f}")
        items.append((SYN/r['file'], '\n'.join(lines), f"syn__{Path(r['file']).name}"))
for name, rows in late.items():      # subsampled positive sets: whole triplets are not needed, tiles are independent samples
    random.shuffle(rows); keep = rows[:int(SUB[name] * len(items))]; items += keep; print('subsampled', name, len(keep), 'of', len(rows))
pool = len(items); negs = []
for spec in a.neg:
    d, frac = spec.rsplit(':', 1); d = Path(d); files = sorted(list((d/'images').glob('*.png')) + list((d/'images').glob('*.jpg')))
    random.shuffle(files); files = files[:int(float(frac) * pool)]
    negs += [(f, '', f"{d.name}__{f.name}") for f in files]; print('negatives', d.name, len(files), f'(cap {frac})')
random.shuffle(negs); negs = negs[:int(0.15 * pool)]; items += negs
random.shuffle(items); nval = max(200, len(items)//20); ds = Path('/root/datasets')/a.out
for split, rows in (('val', items[:nval]), ('train', items[nval:])):
    (ds/split/'images').mkdir(parents=True, exist_ok=True); (ds/split/'labels').mkdir(parents=True, exist_ok=True)
    for src, txt, name in rows:
        dst = ds/split/'images'/name
        if not dst.exists():
            try: os.link(src, dst)
            except OSError: os.symlink(src, dst)
        (ds/split/'labels'/(Path(name).stem + '.txt')).write_text(txt)
(ds/'data.yaml').write_text(f"path: {ds}\ntrain: train/images\nval: val/images\nnames:\n" + ''.join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
cnt = {}
for _, txt, _ in items:
    for l in txt.splitlines():
        if l.strip(): cnt[CLASSES[int(l.split()[0])]] = cnt.get(CLASSES[int(l.split()[0])], 0) + 1
print('dataset', ds, 'train', len(items)-nval, 'val', nval, '| positives pool', pool, 'negatives', len(negs), '| dropped for hold-out', dropped)
print('labels per class', {c: cnt.get(c, 0) for c in CLASSES})
