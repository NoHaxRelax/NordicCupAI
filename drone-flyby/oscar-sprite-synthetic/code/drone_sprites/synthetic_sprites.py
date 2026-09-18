"""Sprite bank from the reviewed masks, and a synthetic dataset built from it.

    python3 -m drone.grid_training.synthetic_sprites bank BANK_DIR
    python3 -m drone.grid_training.synthetic_sprites prepare BANK_DIR SET_DIR --count N
    (blend every SET_DIR/prepared/*.png with an image model, masked to SET_DIR/band/*.png; save to SET_DIR/model/)
    python3 -m drone.grid_training.synthetic_sprites finalize SET_DIR

Split contract: sprites come only from reviewed training-split instances, backgrounds only from
training-split tiles verified empty (organiser reference tiles with complete labels, and tiles Oscar
reviewed as empty). Development (held-out validation) tiles are never read, so the set can be
evaluated on them.

Finalize keeps every sprite pixel exactly: where the sprite alpha is 1 the prepared pixel is used; the
model's output is used only inside a thin seam band around each sprite, never elsewhere, so it can
neither change a sprite nor add an unlabelled object.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from . import mask_editor as me
from . import sprite_library as sl
from .sprite_library import sha, write

DATA = me.DATA


def export_bank(out, review=me.REVIEW, library=me.LIBRARY):
    """Transparent BGRA sprite for every approved or redrawn review item, with its organiser box."""
    store = me.Store(review=review, library=library)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    m = store.m
    names = m['classes']
    sprites = []
    for iid, d in sorted(store.decisions.items()):
        if d['status'] not in ('approved', 'redrawn') or iid not in store.items:
            continue
        it = store.items[iid]
        r = store.rec[it['tile']]
        assert r['split'] == 'train', f'{iid} is not a training tile'
        tile = store.tile(it['tile'])
        mask, *_ = store.current_mask(it, tile)
        if mask.sum() < 4:
            continue
        X, Y, U, V = it['crop']
        hard = np.zeros(tile.shape[:2], np.uint8)
        hard[Y:V, X:U] = mask
        if it.get('box') is not None:
            box = it['box']
        else:
            anns = [a for a in r['annotations'] if a['class_name'] == it['class_name'] and a['fully_contained']]
            box = [int(round(t)) for t in anns[0]['bbox_xyxy']]
        sp = sl.sprite_patch(tile, box, hard)
        rgba, (ox, oy) = sl.cutout_rgba(sp, return_offset=True)
        bx = [int(sp["box"][0]-ox), int(sp["box"][1]-oy), int(sp["box"][2]-ox), int(sp["box"][3]-oy)]  # organiser box relative to the sprite image
        d_out = out/it['class_name']
        d_out.mkdir(exist_ok=True)
        f = d_out/f'{me.safe(iid)}.png'
        cv2.imwrite(str(f), rgba)
        sprites.append(dict(file=str(f.relative_to(out)), id=iid, class_name=it['class_name'], class_id=names.index(it['class_name']),
                            zoom=it['zoom'], source=r['source'], frame=r['frame'], tile=it['tile'], tile_sha256=r['sha256'], split=r['split'],
                            review_status=d['status'], size=[rgba.shape[1], rgba.shape[0]], box_in_sprite=bx, sha256=sha(f)))
    write(out/'bank.json', dict(classes=names, source_manifest_sha256=sha(DATA/'manifest.json'), review=str(review),
                               note='Reviewed training-split sprites. Original pixels; edge pixels un-mixed from the old background. '
                                    'box_in_sprite is the organiser annotation box in sprite coordinates (it can extend past the sprite).',
                               sprites=sprites))
    return sprites


def transform(rgba, box, k, flip, scale):
    h, w = rgba.shape[:2]
    x, y, u, v = box
    if scale != 1.0:
        W2, H2 = max(2, round(w*scale)), max(2, round(h*scale))
        rgba = cv2.resize(rgba, (W2, H2), interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)
        x, u, y, v = x*W2/w, u*W2/w, y*H2/h, v*H2/h
        w, h = W2, H2
    if flip:
        rgba = rgba[:, ::-1]
        x, u = w-u, w-x
    for _ in range(k % 4):
        rgba = np.rot90(rgba)
        x, y, u, v = y, w-u, v, w-x
        w, h = h, w
    return np.ascontiguousarray(rgba), [x, y, u, v]


def prepare(bank, out, count, seed, zoom_weights=(.25, .5, .25), max_objects=3, scale=(.9, 1.1), band_px=4):
    bank, out = Path(bank), Path(out)
    b = json.loads((bank/'bank.json').read_text())
    m = json.loads((DATA/'manifest.json').read_text())
    assert b['source_manifest_sha256'] == sha(DATA/'manifest.json')
    empties = [r for r in m['records'] if r['split'] == 'train' and r['kind'] == 'background' and r['annotation_complete'] and not r['annotations']]
    by_zoom = {z: [s for s in b['sprites'] if s['zoom'] == z] for z in (0, 1, 2)}
    rng = random.Random(seed)
    for sub in ('prepared', 'band', 'alpha', 'labels'):
        (out/sub).mkdir(parents=True, exist_ok=False)
    images = []
    for i in range(count):
        zoom = rng.choices((0, 1, 2), weights=zoom_weights)[0]
        bg = rng.choice([r for r in empties if r['zoom'] == zoom])
        canvas = cv2.imread(str(DATA/bg['file'])).astype(np.float32)
        side = canvas.shape[0]
        alpha_all = np.zeros(canvas.shape[:2], np.float32)
        taken, anns = [], []
        classes = sorted({s['class_name'] for s in by_zoom[zoom]})
        for _ in range(rng.randint(1, max_objects)):
            cls = rng.choice(classes)
            s = rng.choice([q for q in by_zoom[zoom] if q['class_name'] == cls])
            rgba = cv2.imread(str(bank/s['file']), cv2.IMREAD_UNCHANGED)
            rgba, box = transform(rgba, s['box_in_sprite'], rng.randrange(4), rng.random() < .5, rng.uniform(*scale))
            h, w = rgba.shape[:2]
            # extent = sprite plus its organiser box; both must fit and must not overlap earlier objects
            ex = [min(0, box[0]), min(0, box[1]), max(w, box[2]), max(h, box[3])]
            ew, eh = ex[2]-ex[0], ex[3]-ex[1]
            if ew >= side or eh >= side:
                continue
            for _ in range(40):
                X0, Y0 = rng.randrange(0, int(side-ew)), rng.randrange(0, int(side-eh))
                x0, y0 = int(X0-ex[0]), int(Y0-ex[1])
                region = [X0, Y0, X0+ew, Y0+eh]
                if not any(region[0] < o[2] and o[0] < region[2] and region[1] < o[3] and o[1] < region[3] for o in taken):
                    a = rgba[..., 3:].astype(np.float32)/255
                    canvas[y0:y0+h, x0:x0+w] = a*rgba[..., :3]+(1-a)*canvas[y0:y0+h, x0:x0+w]
                    alpha_all[y0:y0+h, x0:x0+w] = np.maximum(alpha_all[y0:y0+h, x0:x0+w], a[..., 0])
                    taken.append(region)
                    lb = [max(0, x0+box[0]), max(0, y0+box[1]), min(side, x0+box[2]), min(side, y0+box[3])]
                    anns.append(dict(class_id=s['class_id'], class_name=cls, bbox_xyxy=[round(t, 1) for t in lb], sprite=s['id'], sprite_file=s['file']))
                    break
        if not anns:
            continue
        rid = f'synth-{seed}-{i:05d}'
        cv2.imwrite(str(out/'prepared'/f'{rid}.png'), canvas.round().clip(0, 255).astype(np.uint8))
        np.save(out/'alpha'/f'{rid}.npy', alpha_all.astype(np.float16))
        present = (alpha_all > .02).astype(np.uint8)
        core = (alpha_all > .99).astype(np.uint8)
        band = cv2.dilate(present, np.ones((2*band_px+1, 2*band_px+1), np.uint8)) & (1-cv2.erode(core, np.ones((3, 3), np.uint8)))
        cv2.imwrite(str(out/'band'/f'{rid}.png'), band*255)
        images.append(dict(id=rid, zoom=zoom, background=bg['id'], background_sha256=bg['sha256'], background_supervision=bg['supervision'],
                           annotations=anns))
    write(out/'prepared.json', dict(bank=str(bank), bank_sha256=sha(bank/'bank.json'), seed=seed, count=count, zoom_weights=list(zoom_weights),
                                    max_objects=max_objects, scale=list(scale), band_px=band_px, images=images))
    return images


def finalize(out, feather=1.5, use_model=True):
    """Model output only inside the seam band; exact sprite pixels in the core; prepared pixels everywhere else."""
    out = Path(out)
    p = json.loads((out/'prepared.json').read_text())
    (out/'images').mkdir(exist_ok=True)
    done, missing, stats = [], [], []
    for im in p['images']:
        mf = out/'model'/f"{im['id']}.png"
        prep = cv2.imread(str(out/'prepared'/f"{im['id']}.png")).astype(np.float32)
        side = prep.shape[0]
        if not use_model:
            model = prep
        elif not mf.exists():
            missing.append(im['id'])
            continue
        else:
            model = cv2.resize(cv2.imread(str(mf)), (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
        band = cv2.imread(str(out/'band'/f"{im['id']}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
        alpha = np.load(out/'alpha'/f"{im['id']}.npy").astype(np.float32)
        w = cv2.GaussianBlur(band, (0, 0), feather)*(alpha < .99)
        w = w[..., None]
        final = w*model+(1-w)*prep
        cv2.imwrite(str(out/'images'/f"{im['id']}.png"), final.round().clip(0, 255).astype(np.uint8))
        diff = np.abs(model-prep).mean(2)
        stats.append(dict(id=im['id'], band_change=round(float(diff[band > 0].mean()), 2) if (band > 0).any() else 0,
                          model_change_outside_band=round(float(diff[band == 0].mean()), 2)))
        lines = []
        for a in im['annotations']:
            x, y, u, v = a['bbox_xyxy']
            lines.append(f"{a['class_id']} {(x+u)/2/side:.6f} {(y+v)/2/side:.6f} {(u-x)/side:.6f} {(v-y)/side:.6f}")
        (out/'labels'/f"{im['id']}.txt").write_text('\n'.join(lines))
        done.append(dict(im, file=f"images/{im['id']}.png", sha256=sha(out/'images'/f"{im['id']}.png")))
    m = json.loads((DATA/'manifest.json').read_text())
    write(out/'manifest.json', dict(classes=m['classes'], input_size=256, source_manifest_sha256=sha(DATA/'manifest.json'), records=done,
                                    blend=('image model in the seam band' if use_model else 'none: local sprite paste with un-mixed edges'),
                                    missing_model_output=missing, blend_stats=stats if use_model else [],
                                    provenance='Sprites: reviewed training-split instances. Backgrounds: training-split verified-empty tiles. '
                                               'Model output used only in the seam band around sprites. No development data.'))
    return done, missing


def contact_sheet(out, n=24, boxes=True):
    out = Path(out)
    man = json.loads((out/'manifest.json').read_text())
    cells = []
    for r in man['records'][:n]:
        im = cv2.imread(str(out/r['file']))
        if boxes:
            for a in r['annotations']:
                x, y, u, v = [int(t) for t in a['bbox_xyxy']]
                cv2.rectangle(im, (x, y), (u, v), (0, 255, 0), 1)
                cv2.putText(im, a['class_name'][:12], (x, max(9, y-2)), cv2.FONT_HERSHEY_SIMPLEX, .33, (255, 255, 255), 1)
        cells.append(im)
    while len(cells) % 6:
        cells.append(np.zeros_like(cells[0]))
    sheet = np.vstack([np.hstack(cells[k:k+6]) for k in range(0, len(cells), 6)])
    cv2.imwrite(str(out/('contact-sheet.jpg' if boxes else 'contact-sheet-clean.jpg')), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('bank')
    b.add_argument('out', type=Path)
    pr = sub.add_parser('prepare')
    pr.add_argument('bank', type=Path)
    pr.add_argument('out', type=Path)
    pr.add_argument('--count', type=int, default=180)
    pr.add_argument('--seed', type=int, default=1918)
    f = sub.add_parser('finalize')
    f.add_argument('out', type=Path)
    f.add_argument('--without-model', action='store_true', help='use the local composites as final images')
    a = ap.parse_args()
    if a.cmd == 'bank':
        s = export_bank(a.out)
        print(f'{len(s)} sprites', dict(Counter((q['class_name']) for q in s)))
    elif a.cmd == 'prepare':
        ims = prepare(a.bank, a.out, a.count, a.seed)
        print(f'{len(ims)} images, {sum(len(i["annotations"]) for i in ims)} objects', dict(Counter(q['class_name'] for i in ims for q in i['annotations'])))
    else:
        done, missing = finalize(a.out, use_model=not a.without_model)
        contact_sheet(a.out)
        contact_sheet(a.out, boxes=False)
        print(f'{len(done)} final images, {len(missing)} without model output')


if __name__ == '__main__':
    main()
