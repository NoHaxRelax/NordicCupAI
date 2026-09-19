"""Synthetic FULL VIEWS in Ultralytics YOLO format, built from the same sprites and frames as synth.py.

A window classifier is not a detector: a 960x540 view holds about two thousand background windows, and
on unseen-scene views the window model gave 45 false positives per view (mAP50 0.03). A standard
detector trained on whole views sees mostly background in every image and regresses its own boxes, and
Oscar's endpoint loads an Ultralytics checkpoint natively (DRONE_DETECTOR=ultralytics). What his earlier
YOLO lacked (0/71 on the unseen scene) is what this generator supplies: every heading, sprites from
more than one instance, and object and terrain exposure varied independently and widely.

One image = one delivered view rendered exactly like the evaluator does it:
  1. pick a frame of a background scene, a level L (factor f = 4, 2, 1) and a legal camera centre
     (half of the L1 views are the deployed top-band centres);
  2. take the source region (960 f x 540 f native pixels), apply the background exposure LUT;
  3. paste sprites at native resolution with SynthWindows._render (rotation with per-class limits,
     sprite exposure, sharp warp), never on top of a labelled real object or of each other;
  4. reduce to 960x540 with cv2.INTER_AREA;
  5. labels = the pasted organiser-style boxes plus the real labelled objects in the region, clipped
     to the view, kept when at least 35 % visible and at least 3 px on a side.
Views that contain a box of a hand-excluded pseudo-label track (validation_exclude.json, plus the
track lists below) are redrawn, so known label errors never become training targets.

    python elias/data/synth_yolo.py --out elias/out/yolo_both --n 16000 \
        --sprite-scenes helsinki validation --background-scenes helsinki validation \
        --real-val-scenes helsinki validation --workers 8

Exposure ranges come from the environment like synth.py (SYNTH_SPRITE_GAIN, SYNTH_BG_GAIN); this script
sets the wide defaults measured on 2026-09-19 when they are absent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault('SYNTH_SPRITE_GAIN', '0.4,1.9')
os.environ.setdefault('SYNTH_BG_GAIN', '0.5,1.8')

from synth import CLASSES, SynthWindows, _env_range, make_lut  # noqa: E402

FACTOR = {0: 4, 1: 2, 2: 1}
BOUNDS = {0: (1920, 1920, 1080, 1080), 1: (960, 2880, 540, 1620), 2: (480, 3360, 270, 1890)}
MEAN_SPRITES = {0: 5.0, 1: 3.0, 2: 1.5}
# Tall classes lean away from the nadir point, which sits at the bottom edge: their lean always
# points into the upper half of the image (elias/out/tilt_report.md). Sprites were cut from real
# frames, so a limited turn keeps the lean physical; low classes turn freely.
LEAN_LIMITS = {'large_tower': 50.0, 'small_tower': 60.0, 'ta-ta': 60.0, 'medium_launcher': 60.0, 'large_launcher': 60.0}
# Pseudo-label tracks that are the wrong class altogether (Oscar's README, sprites review).
BAD_TRACKS = {'mine-roller-a-005-009', 'large-launcher-c2-080-105', 'large-launcher-c-047-078'}
_G = {}


def _excluded_boxes(store, scene, frame):
    """Boxes of this frame that must never appear in a training view (label known to be wrong)."""
    if scene != 'validation':
        return np.zeros((0, 4))
    data = store.annotation_file(scene, frame); out = []
    ranges = _G.setdefault('ranges', json.loads((HERE/'validation_exclude.json').read_text())['exclude'])
    for a, p in zip(data['annotations'], data.get('provenance') or []):
        t = p.get('track_id', '')
        if t in BAD_TRACKS or any(r['track'] == t and r['first'] <= frame <= r['last'] for r in ranges):
            out.append(a['bbox'])
    hidden = _G.setdefault('hidden', json.loads((HERE/'validation_hidden.json').read_text())['zones'] if (HERE/'validation_hidden.json').exists() else {})
    if str(frame) in hidden:
        out.append(hidden[str(frame)])
    return np.array(out, float).reshape(-1, 4)


def _overlaps(box, others, pad=6.0):
    return any(box[0] < o[2]+pad and box[2] > o[0]-pad and box[1] < o[3]+pad and box[3] > o[1]-pad for o in others)


def _extra_canvas(rng, files, width, height, metres_per_px=0.3):
    """A native-resolution background made of public aerial tiles (UC Merced, 0.3 m/px, 256 px).
    Our native frames are about 0.19 m/px, so the mosaic is built at the tiles' own scale and enlarged
    by 0.3/0.19; it is only used for L0 and L1 views, where the later INTER_AREA reduction by 4 or 2
    removes the softness of that enlargement. Most mosaics use one land-use folder for coherence."""
    k = metres_per_px/0.19; tw, th = int(np.ceil(width/k)), int(np.ceil(height/k))
    folders = sorted({str(Path(f).parent) for f in files})
    pool = [f for f in files if str(Path(f).parent) == folders[int(rng.integers(len(folders)))]] if rng.random() < 0.7 else files
    canvas = np.zeros((th, tw, 3), np.uint8); y = 0
    while y < th:
        x = 0; rowh = 0
        while x < tw:
            t = cv2.imread(pool[int(rng.integers(len(pool)))], cv2.IMREAD_COLOR)
            if t is None:
                continue
            t = np.rot90(t, int(rng.integers(4))); t = t[:, ::-1] if rng.random() < .5 else t
            hh, ww = min(t.shape[0], th-y), min(t.shape[1], tw-x)
            canvas[y:y+hh, x:x+ww] = t[:hh, :ww]; x += t.shape[1]; rowh = max(rowh, t.shape[0])
        y += rowh
    return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_LINEAR)


def make_view(win, rng, real_only=False):
    """(image 540x960x3 uint8, [(class index, x1, y1, x2, y2) in delivered px])"""
    store = win.store
    for _ in range(20):
        scene, frame = store.keys[int(rng.integers(len(store.keys)))]
        level = int(rng.choice(3, p=[0.30, 0.45, 0.25])); f = FACTOR[level]
        x0, x1, y0, y1 = BOUNDS[level]
        if level == 1 and rng.random() < 0.5:
            cx, cy = int(rng.choice([960, 1920, 2880])), 540
        else:
            cx, cy = int(rng.integers(x0, x1+1)), int(rng.integers(y0, y1+1))
        rx, ry = cx-480*f, cy-270*f
        region = np.array([rx, ry, rx+960*f, ry+540*f], float)
        if not _overlaps(region, _excluded_boxes(store, scene, frame), pad=0):
            break
    extra = _G.get('extra') if (not real_only and level < 2 and rng.random() < _G.get('extra_prob', 0.)) else None
    patch = (_extra_canvas(rng, extra, 960*f, 540*f) if extra else
             np.ascontiguousarray(store.frame(scene, frame)[ry:ry+540*f, rx:rx+960*f]))
    blo, bhi = _env_range('SYNTH_BG_GAIN', 1.0, 1.0)
    if bhi > blo and not real_only:
        patch = cv2.LUT(patch, make_lut(float(np.exp(rng.uniform(np.log(blo), np.log(bhi)))),
                                        rng.uniform(0.85, 1.18), rng.uniform(0.85, 1.18), 0.45))
    labels, taken = [], []
    for a in ([] if extra else store.annotations(scene, frame)):    # real labelled objects in the region
        b = np.array(a['bbox'], float)-np.array([rx, ry, rx, ry])
        if b[2] > 0 and b[3] > 0 and b[0] < 960*f and b[1] < 540*f:
            taken.append(b); labels.append((CLASSES.index(a['object_id']), b))
    if not real_only:
        for _ in range(int(rng.poisson(MEAN_SPRITES[level]))):
            c = win.available[int(rng.integers(len(win.available)))]
            rendered = win._render(rng, win._choose_sprite(rng, c, level))
            for _try in range(12):
                ref = np.array([rng.uniform(-10, 960*f+10), rng.uniform(-10, 540*f+10)])
                box = rendered.label+np.tile(ref, 2)
                if not _overlaps(box, taken):
                    placed = win._paste(patch, rendered, ref)
                    box = rendered.label+np.tile(placed, 2); taken.append(box); labels.append((c, box)); break
    image = patch if f == 1 else cv2.resize(patch, (960, 540), interpolation=cv2.INTER_AREA)
    rows = []
    for c, b in labels:
        d = b/f; area = max(1e-6, (d[2]-d[0])*(d[3]-d[1]))
        k = np.array([max(0., d[0]), max(0., d[1]), min(960., d[2]), min(540., d[3])])
        if k[2]-k[0] >= 3 and k[3]-k[1] >= 3 and (k[2]-k[0])*(k[3]-k[1])/area >= 0.35:
            rows.append((c, *k))
    return image, rows


def _init(kw, extra_dir=None, extra_prob=0.):
    cv2.setNumThreads(1)
    win = SynthWindows(1, **kw); win._setup(); _G['win'] = win
    if extra_dir:
        files = sorted(str(f) for f in Path(extra_dir).rglob('*') if f.suffix.lower() in ('.tif', '.tiff', '.jpg', '.jpeg', '.png'))
        _G['extra'], _G['extra_prob'] = files, float(extra_prob)


def _work(job):
    index, seed, split, out, real_only = job
    rng = np.random.default_rng([seed, index, 11 if real_only else 3])
    image, rows = make_view(_G['win'], rng, real_only)
    stem = f'{split}_{index:06d}'
    cv2.imwrite(str(Path(out)/'images'/split/f'{stem}.png'), image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    with open(Path(out)/'labels'/split/f'{stem}.txt', 'w') as h:
        for c, x1, y1, x2, y2 in rows:
            h.write(f'{c} {(x1+x2)/1920:.6f} {(y1+y2)/1080:.6f} {(x2-x1)/960:.6f} {(y2-y1)/540:.6f}\n')
    return len(rows)


def build(out, n, split, kw, workers, seed, real_only=False, extra_dir=None, extra_prob=0.):
    for d in ('images', 'labels'):
        (Path(out)/d/split).mkdir(parents=True, exist_ok=True)
    jobs = [(i, seed, split, str(out), real_only) for i in range(n)]
    warm = SynthWindows(1, **kw); warm._setup()   # decode every frame into the npy cache ONCE, in this process:
    for key in warm.store.keys:                    # two workers decoding the same frame collide on os.replace (Windows)
        warm.store.frame(*key)
    del warm
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(kw, extra_dir, extra_prob)) as pool:
        counts = list(pool.imap_unordered(_work, jobs, chunksize=16))
    print(f'[{split}] {n} views, {sum(counts)} boxes ({np.mean(counts):.1f} per view, {sum(c == 0 for c in counts)} empty), '
          f'{n/(time.time()-t0):.1f} views/s')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=16000)
    ap.add_argument('--sprite-scenes', nargs='+', default=['helsinki', 'validation'])
    ap.add_argument('--background-scenes', nargs='+', default=['helsinki', 'validation'])
    ap.add_argument('--real-val-scenes', nargs='+', default=None, help='scenes whose REAL views form the val split')
    ap.add_argument('--n-val', type=int, default=600); ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--max-bg-frames', type=int, default=60)
    ap.add_argument('--free-rotation', action='store_true', help='ignore the lean limits of tall classes')
    ap.add_argument('--extra-bg', default=None, help='folder of public aerial tiles used as extra backgrounds for L0/L1 views')
    ap.add_argument('--extra-bg-prob', type=float, default=0.4)
    ap.add_argument('--sprite-blur-max', type=float, default=0.0, help='Gaussian sigma upper bound on the warped sprite')
    a = ap.parse_args()
    kw = dict(sprite_scenes=tuple(a.sprite_scenes), background_scenes=tuple(a.background_scenes), rot_max=180.0,
              rot_max_per_class={} if a.free_rotation else LEAN_LIMITS, seed=a.seed, lossless_prob=0.2,
              sprite_blur_max=a.sprite_blur_max,
              max_bg_frames=a.max_bg_frames, exclude_tracks=tuple(BAD_TRACKS))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    build(out, a.n, 'train', kw, a.workers, a.seed, extra_dir=a.extra_bg, extra_prob=a.extra_bg_prob)
    val_scenes = a.real_val_scenes or a.background_scenes
    build(out, a.n_val, 'val', dict(kw, background_scenes=tuple(val_scenes), max_bg_frames=200), a.workers, a.seed+1, real_only=True)
    (out/'data.yaml').write_text(f'path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n'
                                 + ''.join(f'  {i}: {c}\n' for i, c in enumerate(CLASSES)))
    (out/'params.json').write_text(json.dumps({**vars(a), 'sprite_gain': os.environ['SYNTH_SPRITE_GAIN'],
                                               'bg_gain': os.environ['SYNTH_BG_GAIN'], 'alpha_mode': os.environ.get('SYNTH_ALPHA_MODE', 'feather'), 'lean_limits': LEAN_LIMITS}, indent=1))
    print('wrote', out/'data.yaml')


if __name__ == '__main__':
    main()
