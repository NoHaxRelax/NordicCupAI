"""Per-class sprite masks, transferred to every training instance, pasted without the old background.

A library holds one hand-checked hard mask per class, cut from one training tile.
For every other training instance of that class the mask is aligned by masked
template matching over heading and scale, constrained to sit centred inside
the annotation box. The pasted pixels are
always that instance's own original pixels: the mask decides only which pixels
belong to the sprite. Edge pixels, which the source blur mixes with the old
background, get the old background's share swapped for the new one:

    out = original + (1 - alpha) * (new_background - old_background)

where old_background is the source crop inpainted under the sprite. Pixels with
alpha = 1 are unchanged. Only training tiles are read.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from .train_v2 import sha, write
except ImportError:  # torch-free environments (laptop composite builds)
    import hashlib

    def write(p, d):
        p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps(d, indent=2, allow_nan=False)); tmp.replace(p)

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_library(data, m, spec_path, out):
    """spec: [{class_name, tile, crop_xyxy, mask, method, variant?}], masks relative to the spec file.

    A class may have several entries (one per appearance, e.g. reference vs validation frames);
    `variant` names each one and defaults to the class name. A mask image of a different size
    than the crop (for example a hand-drawn mask on an upscaled crop) is resized to the crop."""
    spec = json.loads(Path(spec_path).read_text())
    records = {r['id']: r for r in m['records']}
    out = Path(out)
    if out.exists():
        raise SystemExit(f'Refusing to overwrite {out}')
    entries = []
    for s in spec:
        r = records[s['tile']]
        assert r['split'] == 'train' and r['kind'] == 'positive', 'library sprites must come from training positives'
        X, Y, U, V = s['crop_xyxy']
        tile = cv2.imread(str(data/r['file']))
        crop = tile[Y:V, X:U]
        raw = cv2.imread(str(Path(spec_path).parent/s['mask']), cv2.IMREAD_GRAYSCALE)
        if raw.shape != crop.shape[:2]:
            raw = cv2.resize(raw, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA)
        hard = (raw > 127).astype(np.uint8)
        anns = [a for a in r['annotations'] if a['class_name'] == s['class_name'] and a['fully_contained']]
        assert len(anns) == 1, (s['class_name'], s['tile'])
        x, y, u, v = anns[0]['bbox_xyxy']
        variant = s.get('variant', s['class_name'])
        d = out/variant
        d.mkdir(parents=True)
        cv2.imwrite(str(d/'sprite.png'), crop)
        cv2.imwrite(str(d/'mask.png'), hard*255)
        entry = dict(class_name=s['class_name'], variant=variant, source_tile=s['tile'], source_sha256=r['sha256'], zoom=r['zoom'], crop_xyxy=[X, Y, U, V],
                     box_in_crop=[x-X, y-Y, u-X, v-Y], mask_pixels=int(hard.sum()), method=s['method'],
                     sprite_sha256=sha(d/'sprite.png'), mask_sha256=sha(d/'mask.png'))
        write(d/'meta.json', entry)
        entries.append(entry)
    write(out/'library.json', dict(source_manifest_sha256=sha(data/'manifest.json'), entries=entries,
                                   note='Masks cut from single training tiles; pixels always come from the instance being pasted.'))
    return entries


def load_library(path):
    """Returns (library.json contents, {class_name: [entries]})."""
    path = Path(path)
    lib = json.loads((path/'library.json').read_text())
    out = {}
    for e in lib['entries']:
        d = path/e.get('variant', e['class_name'])
        out.setdefault(e['class_name'], []).append(
            dict(e, sprite=cv2.imread(str(d/'sprite.png')), mask=(cv2.imread(str(d/'mask.png'), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)))
    return lib, out


def transfer_best(tile, box, zoom, entries, **kw):
    """Try every library entry of a class; returns (hard mask, score, angle, scale, variant) of the best, or None."""
    best = None
    for e in entries:
        t = transfer_mask(tile, box, zoom, e, **kw)
        if t is not None and (best is None or t[1] > best[1]):
            best = (*t, e.get('variant', e['class_name']))
    return best


def _rotated(img, mask, angle, k):
    """Rotate and scale sprite+mask about the mask centre; crop to the rotated mask's bounding box."""
    ys, xs = np.where(mask > 0)
    cy, cx = (ys.min()+ys.max())/2, (xs.min()+xs.max())/2
    R = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, k)
    h, w = mask.shape
    side = int(np.ceil(np.hypot(h, w)*k))+4
    R[0, 2] += side/2-cx
    R[1, 2] += side/2-cy
    t = cv2.warpAffine(img, R, (side, side), flags=cv2.INTER_LINEAR)
    m = cv2.warpAffine(mask.astype(np.float32), R, (side, side), flags=cv2.INTER_LINEAR) > .5
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None, None
    return t[ys.min():ys.max()+1, xs.min():xs.max()+1], m[ys.min():ys.max()+1, xs.min():xs.max()+1].astype(np.float32)


def box_iou(a, b):
    ix, iy = max(0, min(a[2], b[2])-max(a[0], b[0])), max(0, min(a[3], b[3])-max(a[1], b[1]))
    inter = ix*iy
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-9)


def transfer_mask(tile, box, zoom, entry, angles=range(0, 360, 10), scales=(.8, .87, .94, 1.0, 1.07, 1.15, 1.25),
                  centre_slack=.15, grow=.12):
    """Align the library mask to one instance by masked template matching over heading and scale.

    Annotation boxes are looser than the visible sprite (for some classes three times wider), so the
    placement is constrained, not the size: the mask's centre must lie within centre_slack of the box
    centre and the mask must stay inside the box grown by `grow` on each side. Object size is the same
    at L0, L1 and L2 in the grid inputs, so scale is searched around 1.
    Returns (hard mask in tile coords, score, angle, scale) or None."""
    x, y, u, v = [int(round(c)) for c in box]
    H, W = tile.shape[:2]
    bw, bh = u-x, v-y
    gx, gy = grow*bw, grow*bh
    lim = [x-gx, y-gy, u+gx, v+gy]
    cx, cy = (x+u)/2, (y+v)/2
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY).astype(np.float32)
    spr = cv2.cvtColor(entry['sprite'], cv2.COLOR_BGR2GRAY).astype(np.float32)
    best = None
    for s in scales:
        for ang in angles:
            t, tm = _rotated(spr, entry['mask'], ang, s)
            if t is None or tm.sum() < 8 or t[tm > 0].std() < 1:
                continue
            th, tw = tm.shape
            if tw > lim[2]-lim[0] or th > lim[3]-lim[1]:
                continue
            # top-left positions keeping the mask centre near the box centre and the mask inside the grown box
            X0 = int(np.ceil(max(lim[0], cx-centre_slack*bw-tw/2, 0)))
            Y0 = int(np.ceil(max(lim[1], cy-centre_slack*bh-th/2, 0)))
            X1 = int(np.floor(min(lim[2]-tw, cx+centre_slack*bw-tw/2, W-tw)))
            Y1 = int(np.floor(min(lim[3]-th, cy+centre_slack*bh-th/2, H-th)))
            if X1 < X0 or Y1 < Y0:
                continue
            res = cv2.matchTemplate(gray[Y0:Y1+th, X0:X1+tw], t, cv2.TM_CCOEFF_NORMED, mask=tm)
            res[~np.isfinite(res)] = -1
            _, score, _, (px, py) = cv2.minMaxLoc(res)
            if best is None or score > best[0]:
                best = (score, ang, s, X0+px, Y0+py, tm)
    if best is None:
        return None
    score, ang, k, px, py, tm = best
    hard = np.zeros((H, W), np.uint8)
    hard[py:py+tm.shape[0], px:px+tm.shape[1]] = tm.astype(np.uint8)
    return hard, float(score), int(ang), float(k)


def sprite_patch(tile, box, hard, margin=6):
    """Window around the sprite with its own pixels, old-background estimate, and the label box inside it."""
    H, W = tile.shape[:2]
    x, y, u, v = [int(round(c)) for c in box]
    ys, xs = np.where(cv2.dilate(hard, np.ones((9, 9), np.uint8)) > 0)
    X, Y = max(0, min(xs.min(), x)-margin), max(0, min(ys.min(), y)-margin)
    U, V = min(W, max(xs.max()+1, u)+margin), min(H, max(ys.max()+1, v)+margin)
    patch = tile[Y:V, X:U].copy()
    h = hard[Y:V, X:U].copy()
    bsrc = cv2.inpaint(patch, cv2.dilate(h, np.ones((5, 5), np.uint8)), 5, cv2.INPAINT_TELEA)
    return dict(patch=patch, hard=h, bsrc=bsrc, box=[x-X, y-Y, u-X, v-Y])


def cutout_rgba(sp, sigma=1.0, min_alpha=.1, return_offset=False):
    """Transparent BGRA cutout of a sprite patch, cropped to the sprite.

    alpha is the blurred hard mask. Edge pixels are un-mixed from the old background,
    F = (I - (1 - alpha) * old_background) / alpha, so no light or dark rim remains;
    pixels with alpha = 1 keep their original values exactly."""
    patch, hard, bsrc = sp['patch'].astype(np.float32), sp['hard'], sp['bsrc'].astype(np.float32)
    a = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), sigma)
    a[a < min_alpha] = 0
    fg = patch.copy()
    edge = (a > 0) & (a < 1)
    fg[edge] = (patch[edge]-(1-a[edge])[:, None]*bsrc[edge])/a[edge][:, None]
    ys, xs = np.where(a > 0)
    y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
    out = np.dstack([fg.round().clip(0, 255).astype(np.uint8), (a*255).round().astype(np.uint8)])
    return (out[y0:y1, x0:x1], (int(x0), int(y0))) if return_offset else out[y0:y1, x0:x1]


def transform(sp, scale=1.0, flip=False, k=0):
    """Scale, flip and quarter-turn a sprite patch; returns arrays plus the transformed label box."""
    patch, hard, bsrc = sp['patch'], sp['hard'].astype(np.float32), sp['bsrc']
    x, y, u, v = sp['box']
    h, w = patch.shape[:2]
    if scale != 1.0:
        W2, H2 = max(4, int(round(w*scale))), max(4, int(round(h*scale)))
        interp = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
        patch, bsrc = cv2.resize(patch, (W2, H2), interpolation=interp), cv2.resize(bsrc, (W2, H2), interpolation=interp)
        hard = cv2.resize(hard, (W2, H2), interpolation=cv2.INTER_LINEAR)
        fx, fy = W2/w, H2/h
        x, u, y, v = x*fx, u*fx, y*fy, v*fy
        h, w = H2, W2
    if flip:
        patch, bsrc, hard = patch[:, ::-1], bsrc[:, ::-1], hard[:, ::-1]
        x, u = w-u, w-x
    for _ in range(k % 4):  # np.rot90 is counter-clockwise: (x, y) -> (y, w - x)
        patch, bsrc, hard = np.rot90(patch), np.rot90(bsrc), np.rot90(hard)
        x, y, u, v = y, w-u, v, w-x
        h, w = w, h
    hard = (hard > .5).astype(np.uint8)
    box = [int(round(x)), int(round(y)), int(round(u)), int(round(v))]
    return np.ascontiguousarray(patch), hard, np.ascontiguousarray(bsrc), box


def paste_sprite(canvas, patch, hard, bsrc, x0, y0, sigma=1.0):
    """Paste at (x0, y0): sprite pixels untouched, edge pixels get the old background swapped for the canvas."""
    h, w = patch.shape[:2]
    B = canvas[y0:y0+h, x0:x0+w].astype(np.float32)
    a = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), sigma)[..., None]
    # correction band: exactly 1 on and next to the sprite, fading to 0 a few pixels out
    sup = np.maximum(cv2.GaussianBlur(cv2.dilate(hard, np.ones((9, 9), np.uint8)).astype(np.float32), (0, 0), 2),
                     cv2.dilate(hard, np.ones((5, 5), np.uint8)).astype(np.float32))[..., None]
    swap = patch.astype(np.float32)+(1-a)*(B-bsrc.astype(np.float32))
    canvas[y0:y0+h, x0:x0+w] = (sup*swap+(1-sup)*B).round().clip(0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description='Build a sprite library from a spec of training-tile crops and hard masks.')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    m = json.loads((a.data/'manifest.json').read_text())
    for e in build_library(a.data, m, a.spec, a.output):
        print(e['class_name'], e['source_tile'], e['mask_pixels'])


if __name__ == '__main__':
    main()
