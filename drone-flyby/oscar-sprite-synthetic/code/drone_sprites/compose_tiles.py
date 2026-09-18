"""Copy-paste composites from approved training tiles only.

Objects are fully contained known training annotations cut from training tiles
(complete or partial: only the known box is used, never the unknown remainder).
Backgrounds are complete training tiles: verified-empty tiles and, optionally,
positive complete tiles whose existing labels are kept. Every composite is
therefore completely labelled. Development and reserved data are never read.

Mask modes: ``box`` pastes the rectangle with feathered edges; ``grabcut``
estimates a foreground mask inside the box (falling back to the box when the
estimate is degenerate) so the object's original background is dropped;
``sprite`` transfers a per-class library mask to each instance (see
``sprite_library``), pastes that instance's own pixels and swaps the old
background out of the blurred edge pixels. Instances whose transfer scores below
``--sprite-min-score`` and classes without a library entry fall back to grabcut.
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from . import sprite_library as sl
from .sprite_library import sha, write


def feather(h, w, margin=3):
    ys = np.minimum(np.arange(h)+1, np.arange(h)[::-1]+1)
    xs = np.minimum(np.arange(w)+1, np.arange(w)[::-1]+1)
    return np.clip(np.minimum(ys[:, None], xs[None, :])/margin, 0, 1).astype(np.float32)


def grabcut_mask(image, box, pad=8, iterations=5):
    x, y, u, v = box
    h, w = image.shape[:2]
    X, Y, U, V = max(0, x-pad), max(0, y-pad), min(w, u+pad), min(h, v+pad)
    patch = np.ascontiguousarray(image[Y:V, X:U])
    mask = np.zeros(patch.shape[:2], np.uint8)
    rect = (x-X, y-Y, u-x, v-y)
    if rect[2] < 4 or rect[3] < 4 or patch.shape[0] < rect[3]+2 or patch.shape[1] < rect[2]+2:
        return None
    try:
        cv2.grabCut(patch, mask, rect, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), iterations, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)[y-Y:v-Y, x-X:u-X]
    frac = fg.mean()
    if frac < .2 or frac > .97:
        return None
    fg = cv2.dilate(fg, np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(fg.astype(np.float32), (5, 5), 1.0)


def overlaps(box, others):
    return any(box[0] < o[2] and o[0] < box[2] and box[1] < o[3] and o[1] < box[3] for o in others)


def build_pool(data, m, mask_mode, min_side=8, library=None, min_score=.6, transfer_log=None):
    pool = {}
    for r in m['records']:
        if r['split'] != 'train' or r['kind'] != 'positive':
            continue
        image = None
        for a in r['annotations']:
            if not a['fully_contained']:
                continue
            x, y, u, v = [int(round(c)) for c in a['bbox_xyxy']]
            if u-x < min_side or v-y < min_side:
                continue
            if image is None:
                image = cv2.imread(str(data/r['file']))
            patch = image[y:v, x:u].copy()
            alpha = sprite = None
            if mask_mode == 'sprite' and library and a['class_name'] in library:
                t = sl.transfer_best(image, a['bbox_xyxy'], r['zoom'], library[a['class_name']])
                ok = t is not None and t[1] >= min_score
                if transfer_log is not None:
                    transfer_log.append(dict(source=r['id'], class_name=a['class_name'], score=None if t is None else round(t[1], 3),
                                             angle=None if t is None else t[2], scale=None if t is None else t[3],
                                             variant=None if t is None else t[4], accepted=ok))
                if ok:
                    sprite = sl.sprite_patch(image, a['bbox_xyxy'], t[0])
            if sprite is None and mask_mode in ('grabcut', 'sprite'):
                alpha = grabcut_mask(image, (x, y, u, v))
            pool.setdefault(a['class_name'], []).append(dict(class_id=a['class_id'], class_name=a['class_name'], group=a['group'],
                                                            source=r['id'], zoom=r['zoom'], patch=patch, alpha=alpha, sprite=sprite))
    return pool


def paste(canvas, patch, alpha, x, y):
    h, w = patch.shape[:2]
    region = canvas[y:y+h, x:x+w].astype(np.float32)
    a = (feather(h, w) if alpha is None else np.minimum(alpha, feather(h, w)))[..., None]
    canvas[y:y+h, x:x+w] = (a*patch.astype(np.float32)+(1-a)*region).round().clip(0, 255).astype(np.uint8)


def distractor_patch(data, empties, zoom, rng, mask_mode):
    """An unlabelled patch cut from a verified-empty training tile: pasted with the same blending, it teaches that paste edges are not objects."""
    src = rng.choice([r for r in empties if r['zoom'] == zoom] or empties)
    im = cv2.imread(str(data/src['file']))
    side = im.shape[0]
    w, h = rng.randint(16, 120), rng.randint(16, 120)
    x, y = rng.randrange(0, side-w), rng.randrange(0, side-h)
    patch = im[y:y+h, x:x+w].copy()
    alpha = None
    if mask_mode == 'grabcut':
        # Irregular soft blob so the distractor also mimics GrabCut-shaped pastes.
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = h/2, w/2
        alpha = np.clip(1.2-((yy-cy)/(h/2))**2-((xx-cx)/(w/2))**2, 0, 1).astype(np.float32)
    return patch, alpha, src['id']


def compose(data, m, output, count, seed, mask_mode, max_objects, scale_range, rot90, hflip, noise, onto_positive, distractors=0, min_side=8,
            sprite_library=None, sprite_min_score=.6, sprite_only=False, zooms=(0, 1, 2)):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    library, library_info, transfer_log = None, None, []
    if mask_mode == 'sprite':
        lib_meta, library = sl.load_library(sprite_library)
        assert lib_meta['source_manifest_sha256'] == sha(data/'manifest.json'), 'sprite library was built from a different manifest'
        library_info = dict(path=str(sprite_library), library_sha256=sha(Path(sprite_library)/'library.json'), classes=sorted(library),
                            variants={k: [e.get('variant', k) for e in v] for k, v in sorted(library.items())}, min_score=sprite_min_score)
    pool = build_pool(data, m, mask_mode, min_side, library, sprite_min_score, transfer_log)
    if sprite_only:  # keep only instances with an accepted sprite mask (no GrabCut or box fallback)
        pool = {k: [o for o in v if o['sprite'] is not None] for k, v in pool.items()}
    classes = [c for c in m['classes'] if pool.get(c)]
    train = [r for r in m['records'] if r['split'] == 'train' and r['annotation_complete']]
    empties = [r for r in train if r['kind'] == 'background']
    positives = [r for r in train if r['kind'] == 'positive']
    side = train[0]['input_size']
    (output/'images').mkdir(parents=True)
    (output/'labels').mkdir()
    records = []
    grabcut_used = 0
    sprite_used = 0
    distractors_used = 0
    for i in range(count):
        zoom = rng.randrange(3) if tuple(zooms) == (0, 1, 2) else rng.choice(list(zooms))  # keeps old seeds reproducible
        candidates = positives if (onto_positive and rng.random() < onto_positive) else empties
        bg = rng.choice([r for r in candidates if r['zoom'] == zoom])
        canvas = cv2.imread(str(data/bg['file']))
        assert canvas.shape[:2] == (side, side)
        boxes = [[int(round(c)) for c in a['bbox_xyxy']] for a in bg['annotations']]
        annotations = [dict(class_id=a['class_id'], class_name=a['class_name'], group=a['group'], bbox_xyxy=a['bbox_xyxy'], origin='background') for a in bg['annotations']]
        for _ in range(rng.randint(0, distractors)):
            patch, alpha, source = distractor_patch(data, empties, zoom, rng, mask_mode)
            h, w = patch.shape[:2]
            for _ in range(25):
                x, y = rng.randrange(0, side-w+1), rng.randrange(0, side-h+1)
                box = [x, y, x+w, y+h]
                if not overlaps(box, boxes):
                    paste(canvas, patch, alpha, x, y)
                    boxes.append(box)
                    distractors_used += 1
                    annotations.append(dict(class_id=None, class_name=None, group=None, bbox_xyxy=box, origin='distractor', source=source))
                    break
        for _ in range(rng.randint(1, max_objects)):
            name = rng.choice(classes)
            same_zoom = [o for o in pool[name] if o['zoom'] == zoom]
            obj = rng.choice(same_zoom or pool[name])
            patch, alpha = obj['patch'], obj['alpha']
            s = rng.uniform(*scale_range)
            if obj['sprite'] is not None:
                flip = hflip and rng.random() < .5
                k = rng.randrange(4) if rot90 else 0
                spatch, shard, sbsrc, lbox = sl.transform(obj['sprite'], s, flip, k)
                h, w = spatch.shape[:2]
                if w >= side or h >= side:
                    continue
                for _ in range(25):
                    x, y = rng.randrange(0, side-w+1), rng.randrange(0, side-h+1)
                    window = [x, y, x+w, y+h]
                    if not overlaps(window, boxes):
                        sl.paste_sprite(canvas, spatch, shard, sbsrc, x, y)
                        boxes.append(window)
                        sprite_used += 1
                        box = [max(0, x+lbox[0]), max(0, y+lbox[1]), min(side, x+lbox[2]), min(side, y+lbox[3])]
                        annotations.append(dict(class_id=obj['class_id'], class_name=name, group=obj['group'], bbox_xyxy=box, origin='pasted',
                                                source=obj['source'], source_zoom=obj['zoom'], scale=round(s, 3), mask='sprite'))
                        break
                continue
            w, h = max(8, int(round(patch.shape[1]*s))), max(8, int(round(patch.shape[0]*s)))
            patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA)
            if alpha is not None:
                alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
            if hflip and rng.random() < .5:
                patch = patch[:, ::-1].copy()
                alpha = None if alpha is None else alpha[:, ::-1].copy()
            if rot90:
                k = rng.randrange(4)
                patch = np.rot90(patch, k).copy()
                alpha = None if alpha is None else np.rot90(alpha, k).copy()
                h, w = patch.shape[:2]
            if w >= side or h >= side:
                continue
            for _ in range(25):
                x, y = rng.randrange(0, side-w+1), rng.randrange(0, side-h+1)
                box = [x, y, x+w, y+h]
                if not overlaps(box, boxes):
                    paste(canvas, patch, alpha, x, y)
                    boxes.append(box)
                    grabcut_used += alpha is not None
                    annotations.append(dict(class_id=obj['class_id'], class_name=name, group=obj['group'], bbox_xyxy=box, origin='pasted',
                                            source=obj['source'], source_zoom=obj['zoom'], scale=round(s, 3)))
                    break
        if noise:
            canvas = (canvas.astype(np.float32)+np_rng.normal(0, noise, canvas.shape)).round().clip(0, 255).astype(np.uint8)
        if not any(a['origin'] == 'pasted' for a in annotations):
            continue
        rid = f'composite-{seed}-{i:05d}'
        path = output/'images'/(rid+'.png')
        cv2.imwrite(str(path), canvas)
        lines = [f"{a['class_id']} {(a['bbox_xyxy'][0]+a['bbox_xyxy'][2])/2/side:.6f} {(a['bbox_xyxy'][1]+a['bbox_xyxy'][3])/2/side:.6f} {(a['bbox_xyxy'][2]-a['bbox_xyxy'][0])/side:.6f} {(a['bbox_xyxy'][3]-a['bbox_xyxy'][1])/side:.6f}" for a in annotations if a['origin'] != 'distractor']
        (output/'labels'/(rid+'.txt')).write_text('\n'.join(lines))
        records.append(dict(id=rid, file=f'images/{rid}.png', sha256=sha(path), zoom=zoom, background=bg['id'], background_kind=bg['kind'],
                            annotations=[a for a in annotations if a['origin'] != 'distractor'], distractors=[a for a in annotations if a['origin'] == 'distractor'], annotation_complete=True, split='train', kind='positive', supervision='composite_from_known_training_objects'))
    settings = dict(count=count, seed=seed, mask=mask_mode, max_objects=max_objects, scale_range=list(scale_range), rot90=rot90, hflip=hflip,
                    noise_sigma=noise, onto_positive_fraction=onto_positive, grabcut_pastes=grabcut_used, sprite_pastes=sprite_used, distractors=distractors, distractor_pastes=distractors_used, min_side=min_side, pasted_classes=classes,
                    pool={k: len(v) for k, v in pool.items()}, pool_with_mask={k: sum(o['alpha'] is not None for o in v) for k, v in pool.items()},
                    pool_with_sprite={k: sum(o['sprite'] is not None for o in v) for k, v in pool.items()}, sprite_library=library_info, sprite_only=sprite_only, zooms=list(zooms))
    if transfer_log:
        write(output/'sprite_transfer.json', transfer_log)
    write(output/'manifest.json', dict(classes=m['classes'], input_size=side, source_manifest_sha256=sha(data/'manifest.json'), settings=settings, records=records,
                                       provenance='Objects: fully contained known training annotations. Backgrounds: complete training tiles. No development or reserved data.'))
    print(json.dumps(dict(records=len(records), pasted=sum(sum(a['origin'] == 'pasted' for a in r['annotations']) for r in records), **{k: v for k, v in settings.items() if k not in ('pool', 'pool_with_mask', 'pool_with_sprite', 'sprite_library')})))
    return records


def contact_sheet(output, records, n=24, cell=256):
    cols = 6
    rows = (min(n, len(records))+cols-1)//cols
    sheet = np.zeros((rows*cell, cols*cell, 3), np.uint8)
    for k, r in enumerate(records[:n]):
        im = cv2.imread(str(output/r['file']))
        im = cv2.resize(im, (cell, cell))
        f = cell/im.shape[0]
        for a in r['annotations']:
            x, y, u, v = [int(c*f) for c in a['bbox_xyxy']]
            cv2.rectangle(im, (x, y), (u, v), (0, 255, 0) if a['origin'] == 'pasted' else (0, 200, 255), 1)
            cv2.putText(im, a['class_name'][:8], (x, max(10, y-2)), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)
        sheet[(k//cols)*cell:(k//cols+1)*cell, (k % cols)*cell:(k % cols+1)*cell] = im
    cv2.imwrite(str(output/'contact-sheet.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=1500)
    p.add_argument('--seed', type=int, default=1731)
    p.add_argument('--mask', choices=['box', 'grabcut', 'sprite'], default='box')
    p.add_argument('--sprite-library', type=Path, default=None, help='library directory for --mask sprite (see sprite_library.py)')
    p.add_argument('--sprite-min-score', type=float, default=.6, help='minimum masked-NCC transfer score; lower-scoring instances fall back to grabcut')
    p.add_argument('--sprite-only', action='store_true', help='paste only instances with an accepted sprite mask')
    p.add_argument('--zooms', type=int, nargs='+', default=[0, 1, 2], choices=[0, 1, 2], help='zoom levels to generate (background and preferred object zoom)')
    p.add_argument('--max-objects', type=int, default=3)
    p.add_argument('--scale', type=float, nargs=2, default=[.85, 1.2])
    p.add_argument('--rot90', action='store_true')
    p.add_argument('--no-hflip', action='store_true')
    p.add_argument('--noise', type=float, default=0)
    p.add_argument('--onto-positive', type=float, default=.25)
    p.add_argument('--distractors', type=int, default=0, help='up to this many unlabelled background patches pasted per composite')
    p.add_argument('--min-side', type=int, default=8, help='skip source objects smaller than this on either side (tiny objects give poor masks)')
    a = p.parse_args()
    m = json.loads((a.data/'manifest.json').read_text())
    if a.mask == 'sprite' and not a.sprite_library:
        p.error('--mask sprite needs --sprite-library')
    records = compose(a.data, m, a.output, a.count, a.seed, a.mask, a.max_objects, a.scale, a.rot90, not a.no_hflip, a.noise, a.onto_positive, a.distractors, a.min_side,
                      a.sprite_library, a.sprite_min_score, a.sprite_only, tuple(a.zooms))
    contact_sheet(a.output, records)


if __name__ == '__main__':
    main()
