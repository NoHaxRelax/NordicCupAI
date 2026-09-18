"""Build the REAL labelled 64x64 window datasets described in elias/SPEC.md.

Usage (run from drone-flyby/):

    python elias/data/real_crops.py --scene both
    python elias/data/real_crops.py --scene helsinki --views 6 --seed 0
    python elias/data/real_crops.py --scene validation --views 3
    python elias/data/real_crops.py --scene validation --labels-only     (label audit, no windows)

Outputs, per scene, in elias/out/:

    real_<scene>.npz          the SPEC.md container (x, label, level, scale, size_px, box_w, box_h,
                              off, scene, frame, track, source) plus additive extras:
                              vis (visible fraction of the object, 1.0 = whole), vis_w, vis_h
                              (visible part of the box in source px), vis_off (centre of the
                              visible part relative to the window centre, window px), view (camera
                              centre cx, cy that the window was cut from), status (validation
                              review_status, empty on helsinki) and fill (share of the window that
                              lies outside the delivered view and is therefore zero).
    real_<scene>.json         manifest: parameters, counts per class x level, size_px and scale
                              distributions, visible fraction statistics, label filter report,
                              exactness check.
    real_<scene>.png          contact sheet: one row per class, 5 columns per level.
    real_<scene>_edges.png    contact sheet of the windows that have zero fill or a cut object.
    real_<scene>_dropped.png  (pseudo-labelled scenes) every track that lost boxes to the label
                              filters: source crops with the dropped boxes in red and the
                              neighbouring kept boxes in green.

Which box the metadata describes
    box_w, box_h, size_px, off and the choice of the scale s always describe the ANNOTATED box,
    also when a view edge cuts the object. A cut object is therefore shown at the same scale as
    the whole object and falls into the same size_px bin. The window of a cut object is centred
    on the visible part, which vis_w, vis_h and vis_off describe. For whole objects the two
    descriptions are identical. A box clipped by the source frame is annotated clipped, so for
    those samples the annotated box is the visible part.

What one sample is
    For every annotated object, every level L in {0, 1, 2} and K legal camera views that contain the
    whole box, one centred window and one jittered window (object centre moved by up to +-6 window
    px). Level 0 has a single legal view, (1920, 1080), so it gives one centred window and K
    jittered windows. The window is cut in DELIVERED pixel coordinates (integers) of that view and
    is never rendered from a full view: the delivered pixels are recovered from the source patch
    aligned to the view's f-grid with cv2.INTER_AREA (exact block averaging), then reduced by s / f
    with cv2.INTER_AREA, exactly as SPEC.md prescribes. Whatever part of the window falls outside
    the delivered view is zero, which is what a detector sees at a view edge.

Partly visible objects
    1. Frame edge: a box clipped by the source frame (entering or leaving). Its visible fraction is
       estimated against the median unclipped box of the same track. It is a positive when at
       least --min-visible (0.6) is visible. The window is centred on the clipped box.
    2. View edge (levels 1 and 2): --partial-views extra views per (object, level) whose edge cuts
       the box so that 60 to 95 % of its area is inside the view. The window is centred on the
       visible part of the box, because that is what a detector can localise.

Label filters (scenes with pseudo-labels, that is validation)
    The validation boxes come from score probing and some show no object. Four checks run before
    any window is cut, in this order, and the manifest lists every box they remove:
    1. --exclude: a JSON list of hand-checked (track, frame range) entries
       (default elias/data/validation_exclude.json).
    2. area: a box that the frame does not clip and whose area is below --min-area-ratio (0.5) of
       its track's median.
    3. motion: objects are static, so every box must move with the terrain. The step between
       consecutive frames must be 40..110 source px down and at most 25 px sideways (measured on
       the box centre, or on the unclipped edge when the frame clips a box). A step of at most
       2 px counts as a repeated frame when most tracks of that frame pair agree, and the next
       step may then be twice as long. Steps link boxes into chains. The longest chain of a track
       is kept, other chains are dropped unless they have at least --min-chain (8) boxes.
    4. contained (report only unless --drop-contained): a box that lies at least 90 % inside a box
       of another class. On validation this flags two boxes, one empty box and one real jet
       parked in front of a hangar, so it does not drop by default.
    Dropped boxes stay keep-out zones for the negatives, inflated to the median size of their
    track, because the object may still be near a misplaced box.

Negatives (label 16)
    One per positive, with the same (level, scale), drawn in the same frame where possible. The
    window centre keeps at least max(0.75 x longer side, --neg-guard window px) from every object
    centre. On validation the centre is also inside the band of rows spanned by that frame's
    labelled boxes and at least 96 source px from every labelled box; those labels are incomplete,
    so validation negatives are noisy.

Unobserved regions
    The validation frames are reconstructed from recorded views, and a frame can hold rectangles
    that were never observed (exact zeros, validation frame 4 has three 960x540 ones). Any window,
    positive or negative, whose source footprint touches such a hole is dropped and counted.

Exactness check
    --verify-frames frames are checked against local_evaluator.render_view (the organisers' own
    rendering, PNG round trip included). The manifest stores the max abs pixel difference.
"""

import argparse
import base64
import collections
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # drone-flyby/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH, OBJECT_CLASSES  # noqa: E402
from utils import center_bounds_for_level  # noqa: E402

WINDOW = 64                      # window side in window px
VIEW_W, VIEW_H = 960, 540        # delivered view size
FACTOR = {0: 4, 1: 2, 2: 1}      # source px per delivered px
BACKGROUND = len(OBJECT_CLASSES)  # label 16
CLASS_INDEX = {name: i for i, name in enumerate(OBJECT_CLASSES)}
ROW_NAMES = list(OBJECT_CLASSES) + ['background']
MAX_OBJECT_PX = 48               # SPEC: max(box_w, box_h) / s <= 48
MIN_IMAGE_BYTES = 1_000_000      # smaller images are placeholders
PLACEHOLDER_FRAMES = {'validation': {1, 2, 3}}
VALIDATION_CLEARANCE = 96        # source px between a negative centre and any labelled box
HOLE_MIN_SIDE = 256              # exact zero regions this large are unobserved, not content
DEFAULT_EXCLUDE = ROOT / 'elias' / 'data' / 'validation_exclude.json'
MOTION_DY = (40.0, 110.0)        # legal terrain step per frame, source px down the image
MOTION_DX = 25.0                 # legal sideways step per frame, source px
FROZEN_PX = 2.0                  # a step this small means the frame was repeated
CONTAINED_FRACTION = 0.9         # share of a box inside a box of another class that is flagged


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def pick_scale(longer_side, level):
    """Return (s, fallback). s is the smallest of {f, 2f, 4f} (capped at 8) with longer/s <= 48."""
    f = FACTOR[level]
    candidates = [c for c in (f, 2 * f, 4 * f) if c <= 8]
    for s in candidates:
        if longer_side / s <= MAX_OBJECT_PX:
            return s, False
    return candidates[-1], True


def view_origin(level, view_center):
    """Top-left corner of the source region of a view."""
    f = FACTOR[level]
    return view_center[0] - VIEW_W * f // 2, view_center[1] - VIEW_H * f // 2


def is_legal_view(level, view_center):
    min_x, max_x, min_y, max_y = center_bounds_for_level(level)
    return min_x <= view_center[0] <= max_x and min_y <= view_center[1] <= max_y


def origin_range(lo, hi, view_len, frame_len):
    """Integer view origins v with v <= lo, v + view_len >= hi and the view inside the frame."""
    v_min = max(0, int(math.ceil(hi - view_len)))
    v_max = min(frame_len - view_len, int(math.floor(lo)))
    return v_min, v_max


def draw_origin(v_min, v_max, phase, f, rng):
    """Uniform integer in [v_min, v_max], with v % f == phase when such a value exists."""
    if phase is not None and f > 1:
        first = v_min + ((phase - v_min) % f)
        if first <= v_max:
            return int(first + f * rng.integers(0, (v_max - first) // f + 1))
    return int(rng.integers(v_min, v_max + 1))


def sample_whole_view(box, level, rng, phase=None):
    """A legal camera centre whose view contains the whole box, or None."""
    f = FACTOR[level]
    span_w, span_h = VIEW_W * f, VIEW_H * f
    x_min, x_max = origin_range(box[0], box[2], span_w, IMAGE_WIDTH)
    y_min, y_max = origin_range(box[1], box[3], span_h, IMAGE_HEIGHT)
    if x_min > x_max or y_min > y_max:
        return None
    px, py = phase if phase is not None else (None, None)
    vx = draw_origin(x_min, x_max, px, f, rng)
    vy = draw_origin(y_min, y_max, py, f, rng)
    return vx + span_w // 2, vy + span_h // 2


def sample_cutting_view(box, level, rng, vis_lo, vis_hi):
    """A legal view whose edge cuts the box: (centre, visible_box, visible_fraction) or None."""
    f = FACTOR[level]
    span_w, span_h = VIEW_W * f, VIEW_H * f
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    for side in rng.permutation(4):
        hidden = 1.0 - rng.uniform(vis_lo, vis_hi)
        if side == 0:      # the view's left edge cuts the box
            vx = int(round(x1 + hidden * w))
        elif side == 1:    # right edge
            vx = int(round(x2 - hidden * w)) - span_w
        elif side == 2:    # top edge
            vy = int(round(y1 + hidden * h))
        else:              # bottom edge
            vy = int(round(y2 - hidden * h)) - span_h
        if side < 2:
            if not 0 <= vx <= IMAGE_WIDTH - span_w:
                continue
            y_min, y_max = origin_range(y1, y2, span_h, IMAGE_HEIGHT)
            if y_min > y_max:
                continue
            vy = int(rng.integers(y_min, y_max + 1))
        else:
            if not 0 <= vy <= IMAGE_HEIGHT - span_h:
                continue
            x_min, x_max = origin_range(x1, x2, span_w, IMAGE_WIDTH)
            if x_min > x_max:
                continue
            vx = int(rng.integers(x_min, x_max + 1))
        visible = (max(x1, vx), max(y1, vy), min(x2, vx + span_w), min(y2, vy + span_h))
        fraction = (visible[2] - visible[0]) * (visible[3] - visible[1]) / float(w * h)
        if vis_lo <= fraction < 1.0:
            return (vx + span_w // 2, vy + span_h // 2), visible, fraction
    return None


def window_origin(level, view_center, center_src, s, jitter_d=(0, 0)):
    """Top-left of the window in delivered px. jitter_d shifts the OBJECT, in delivered px."""
    f = FACTOR[level]
    side_d = WINDOW * s // f
    vx, vy = view_origin(level, view_center)
    ud = (center_src[0] - vx) / f
    vd = (center_src[1] - vy) / f
    ox = int(math.floor(ud - side_d / 2 + 0.5)) - jitter_d[0]
    oy = int(math.floor(vd - side_d / 2 + 0.5)) - jitter_d[1]
    return ox, oy


def clip_span(start, length, limit):
    return max(start, 0), min(start + length, limit)


def reduce_area(image, factor):
    """Integer INTER_AREA reduction (block averaging anchored at the image origin)."""
    if factor == 1:
        return image
    size = (image.shape[1] // factor, image.shape[0] // factor)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def window_from_source(frame_img, level, view_center, ox, oy, s):
    """The production path: source patch on the view's f-grid -> delivered px -> reduce by s / f.

    Returns (window uint8[64,64,3], zero_fill_fraction).
    """
    f = FACTOR[level]
    m = s // f
    side_d = WINDOW * m
    vx, vy = view_origin(level, view_center)
    ax, bx = clip_span(ox, side_d, VIEW_W)
    ay, by = clip_span(oy, side_d, VIEW_H)
    canvas = np.zeros((side_d, side_d, 3), np.uint8)
    if ax < bx and ay < by:
        patch = frame_img[vy + ay * f: vy + by * f, vx + ax * f: vx + bx * f]
        canvas[ay - oy: by - oy, ax - ox: bx - ox] = reduce_area(patch, f)
    inside = max(bx - ax, 0) * max(by - ay, 0)
    return reduce_area(canvas, m), 1.0 - inside / float(side_d * side_d)


def window_from_source_direct(frame_img, level, view_center, ox, oy, s):
    """Single stage variant: zero padded source patch reduced by s in one INTER_AREA call."""
    f = FACTOR[level]
    side_d = WINDOW * s // f
    vx, vy = view_origin(level, view_center)
    ax, bx = clip_span(ox, side_d, VIEW_W)
    ay, by = clip_span(oy, side_d, VIEW_H)
    canvas = np.zeros((side_d * f, side_d * f, 3), np.uint8)
    if ax < bx and ay < by:
        patch = frame_img[vy + ay * f: vy + by * f, vx + ax * f: vx + bx * f]
        canvas[(ay - oy) * f: (by - oy) * f, (ax - ox) * f: (bx - ox) * f] = patch
    return reduce_area(canvas, s)


def window_from_view(view_img, ox, oy, s, f):
    """Reference path: cut the window from a fully rendered 960x540 view."""
    m = s // f
    side_d = WINDOW * m
    ax, bx = clip_span(ox, side_d, VIEW_W)
    ay, by = clip_span(oy, side_d, VIEW_H)
    canvas = np.zeros((side_d, side_d, 3), np.uint8)
    if ax < bx and ay < by:
        canvas[ay - oy: by - oy, ax - ox: bx - ox] = view_img[ay:by, ax:bx]
    return reduce_area(canvas, m)


def render_view_like_evaluator(frame_img, level, view_center):
    """Render a view with the organisers' own code (PNG round trip included)."""
    from local_evaluator import Camera, render_view
    camera = Camera(resolution_level=level, center_x=view_center[0], center_y=view_center[1])
    encoded = render_view(frame_img, camera)
    data = np.frombuffer(base64.b64decode(encoded), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# --------------------------------------------------------------------------- #
# Scene loading
# --------------------------------------------------------------------------- #

def touches_frame(box):
    """Sides of the source frame that clip this box (organiser boxes stop at 3839 / 2159)."""
    sides = []
    if box[0] <= 0:
        sides.append('left')
    if box[1] <= 0:
        sides.append('top')
    if box[2] >= IMAGE_WIDTH - 1:
        sides.append('right')
    if box[3] >= IMAGE_HEIGHT - 1:
        sides.append('bottom')
    return sides


def load_scene(scene, max_frames=None, label_options=None):
    """Return (frames, skipped, label_report). Each frame: dict(frame, path, objects, ignored).

    label_options: keyword arguments of filter_labels, or None to keep every box. The filters only
    run on scenes whose annotations carry provenance, that is pseudo-labels.
    """
    base = ROOT / 'src' / scene
    frames, skipped = [], []
    for ann_path in sorted((base / 'annotations').glob('frame_*.json')):
        record = json.loads(ann_path.read_text())
        number = int(record['frame'])
        image_path = base / 'images' / f'frame_{number:06d}.png'
        if number in PLACEHOLDER_FRAMES.get(scene, set()):
            skipped.append((number, 'placeholder frame'))
            continue
        if not image_path.exists() or image_path.stat().st_size < MIN_IMAGE_BYTES:
            skipped.append((number, 'image missing or smaller than 1 MB'))
            continue
        provenance = record.get('provenance')
        objects = []
        for i, ann in enumerate(record['annotations']):
            box = tuple(int(v) for v in ann['bbox'])
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            name = ann['object_id']
            objects.append({
                'name': name,
                'label': CLASS_INDEX[name],
                'box': box,
                'track': provenance[i]['track_id'] if provenance else name,
                'status': provenance[i]['review_status'] if provenance else '',
                'frame_sides': touches_frame(box),
            })
        frames.append({'frame': number, 'path': image_path, 'objects': objects, 'ignored': [],
                       'pseudo_labels': provenance is not None})
    label_report = None
    if label_options is not None and any(fr['pseudo_labels'] for fr in frames):
        label_report = filter_labels(scene, frames, **label_options)
    if max_frames:
        step = max(1, len(frames) // max_frames)
        frames = frames[::step][:max_frames]
    return frames, skipped, label_report


# --------------------------------------------------------------------------- #
# Label sanity filters for pseudo-labelled scenes
# --------------------------------------------------------------------------- #

def load_exclusions(path, scene, key='exclude'):
    """Hand-checked (track, first, last, reason) entries of one scene from the --exclude JSON.

    key 'exclude' lists the boxes to drop, key 'flagged_but_kept' lists known loose boxes that
    only go into the manifest.
    """
    if not path or not Path(path).exists():
        return []
    entries = json.loads(Path(path).read_text(encoding='utf-8')).get(key, [])
    return [e for e in entries if e.get('scene', scene) == scene]


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def edge_aware_step(obj_a, obj_b):
    """(dx, dy) of a static object between two frames, or None when it cannot be measured.

    The box centre is used unless the source frame clips one of the two boxes. Then the edge that
    is free in both frames is used, because a clipped box grows instead of moving.
    """
    sides = set(obj_a['frame_sides']) | set(obj_b['frame_sides'])
    if {'top', 'bottom'} <= sides or {'left', 'right'} <= sides:
        return None
    a, b = obj_a['box'], obj_b['box']
    if 'top' in sides:
        dy = b[3] - a[3]
    elif 'bottom' in sides:
        dy = b[1] - a[1]
    else:
        dy = (b[1] + b[3] - a[1] - a[3]) / 2.0
    if 'left' in sides:
        dx = b[2] - a[2]
    elif 'right' in sides:
        dx = b[0] - a[0]
    else:
        dx = (b[0] + b[2] - a[0] - a[2]) / 2.0
    return float(dx), float(dy)


def motion_chains(rows, frozen_pairs):
    """Split one track into chains of boxes that move with the terrain.

    rows: list of (frame, obj) sorted by frame. frozen_pairs: set of (frame_a, frame_b) that most
    tracks agree are a repeated frame. Returns a list of chains, each a list of row indices.
    """
    chains, carry = [[0]], 0
    for i in range(1, len(rows)):
        (frame_a, obj_a), (frame_b, obj_b) = rows[i - 1], rows[i]
        step = edge_aware_step(obj_a, obj_b)
        gap = frame_b - frame_a
        if step is None:
            linked, carry = True, 0
        elif (frame_a, frame_b) in frozen_pairs and max(abs(step[0]), abs(step[1])) <= FROZEN_PX:
            linked, carry = True, carry + 1
        else:
            n = gap + carry          # a repeated frame makes the next step that much longer
            linked = (MOTION_DY[0] * n <= step[1] <= MOTION_DY[1] * n
                      and abs(step[0]) <= MOTION_DX * n)
            carry = 0
        if linked:
            chains[-1].append(i)
        else:
            chains.append([i])
    return chains


def filter_labels(scene, frames, exclude_path, min_area_ratio, min_chain, drop_contained):
    """Remove boxes that fail the label checks. Returns the report that goes into the manifest.

    A removed object moves from fr['objects'] to fr['ignored'] with obj['drop_reason'] set and
    obj['keepout_box'] holding the box inflated to the median size of its track.
    """
    by_track = collections.defaultdict(list)
    for fr in frames:
        for obj in fr['objects']:
            obj['drop_reason'] = None
            by_track[obj['track']].append((fr['frame'], obj))
    for rows in by_track.values():
        rows.sort(key=lambda row: row[0])

    # 1. hand-checked exclusions
    exclusions = load_exclusions(exclude_path, scene)
    for entry in exclusions:
        for number, obj in by_track.get(entry['track'], []):
            if entry['first'] <= number <= entry['last']:
                obj['drop_reason'] = 'exclude_list'

    # The automatic checks also run on a copy that ignores the exclusion list, so the manifest can
    # say how much of the hand-checked list the automatic checks would have found on their own.
    def automatic_checks(respect_exclusions):
        verdict, kept_by_track = {}, {}
        for track, rows in by_track.items():
            live = [(n, o) for n, o in rows
                    if not (respect_exclusions and o['drop_reason'] == 'exclude_list')]
            # 2. area against the track median of boxes the frame does not clip
            free = [box_area(o['box']) for _, o in live if not o['frame_sides']]
            median_area = float(np.median(free)) if free else 0.0
            kept = []
            for n, o in live:
                if (not o['frame_sides'] and median_area
                        and box_area(o['box']) < min_area_ratio * median_area):
                    verdict[(track, n)] = 'area'
                else:
                    kept.append((n, o))
            kept_by_track[track] = kept
        # 3. motion: first find the frame pairs that most tracks see as a repeated frame
        pair_votes = collections.defaultdict(list)
        for track, kept in kept_by_track.items():
            for (na, oa), (nb, ob) in zip(kept[:-1], kept[1:]):
                step = edge_aware_step(oa, ob)
                if step is not None and nb - na == 1:
                    pair_votes[(na, nb)].append(max(abs(step[0]), abs(step[1])) <= FROZEN_PX)
        frozen_pairs = {pair for pair, votes in pair_votes.items()
                        if sum(votes) * 2 > len(votes)}
        secondary = []
        for track, kept in kept_by_track.items():
            if len(kept) < 2:
                continue
            chains = motion_chains(kept, frozen_pairs)
            longest = max(chains, key=len)
            for chain in chains:
                if chain is longest:
                    continue
                if len(chain) >= min_chain:
                    secondary.append({'track': track, 'first': kept[chain[0]][0],
                                      'last': kept[chain[-1]][0]})
                    continue
                for i in chain:
                    verdict[(track, kept[i][0])] = 'motion'
        return verdict, sorted(frozen_pairs), secondary

    verdict, frozen_pairs, secondary = automatic_checks(respect_exclusions=True)
    verdict_alone, _, _ = automatic_checks(respect_exclusions=False)
    for (track, number), reason in verdict.items():
        for n, obj in by_track[track]:
            if n == number and obj['drop_reason'] is None:
                obj['drop_reason'] = reason

    # 4. containment in a box of another class (a flag, a drop only with --drop-contained)
    contained = []
    for fr in frames:
        for obj in fr['objects']:
            for other in fr['objects']:
                if other is obj or other['name'] == obj['name']:
                    continue
                a, b = obj['box'], other['box']
                ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
                iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
                share = ix * iy / float(box_area(a))
                if share >= CONTAINED_FRACTION:
                    contained.append({'track': obj['track'], 'frame': fr['frame'],
                                      'inside': other['track'], 'share': round(share, 3),
                                      'other_verdict': obj['drop_reason'] or 'kept'})
                    if drop_contained and obj['drop_reason'] is None:
                        obj['drop_reason'] = 'contained'

    # Move the dropped objects aside and describe them.
    median_dims = {}
    for track, rows in by_track.items():
        good = [(o['box'][2] - o['box'][0], o['box'][3] - o['box'][1]) for _, o in rows
                if o['drop_reason'] is None and not o['frame_sides']]
        if good:
            median_dims[track] = np.median(np.array(good, float), axis=0)
    dropped = collections.defaultdict(lambda: collections.defaultdict(list))
    by_reason, by_class, by_status = (collections.Counter() for _ in range(3))
    for fr in frames:
        keep = []
        for obj in fr['objects']:
            if obj['drop_reason'] is None:
                keep.append(obj)
                continue
            x1, y1, x2, y2 = obj['box']
            half_w, half_h = (x2 - x1) / 2.0, (y2 - y1) / 2.0
            if obj['track'] in median_dims:
                half_w = max(half_w, median_dims[obj['track']][0] / 2.0)
                half_h = max(half_h, median_dims[obj['track']][1] / 2.0)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            obj['keepout_box'] = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
            fr['ignored'].append(obj)
            dropped[obj['track']][obj['drop_reason']].append(fr['frame'])
            by_reason[obj['drop_reason']] += 1
            by_class[obj['name']] += 1
            by_status[obj['status']] += 1
        fr['objects'] = keep

    excluded_keys = {(e['track'], n) for e in exclusions for n, _ in by_track.get(e['track'], [])
                     if e['first'] <= n <= e['last']}
    found_alone = sorted(k for k in excluded_keys if k in verdict_alone)
    extra_alone = sorted(k for k in verdict_alone if k not in excluded_keys and k not in verdict)
    return {
        'parameters': {'exclude_file': str(exclude_path) if exclude_path else None,
                       'min_area_ratio': min_area_ratio, 'motion_dy_src_px': list(MOTION_DY),
                       'motion_dx_src_px': MOTION_DX, 'frozen_px': FROZEN_PX,
                       'min_chain': min_chain, 'contained_fraction': CONTAINED_FRACTION,
                       'drop_contained': bool(drop_contained)},
        'boxes_before': int(sum(len(rows) for rows in by_track.values())),
        'boxes_dropped': int(sum(by_reason.values())),
        'dropped_by_reason': dict(by_reason),
        'dropped_by_class': dict(by_class),
        'dropped_by_review_status': dict(by_status),
        'dropped_by_track': {t: {r: f for r, f in reasons.items()}
                             for t, reasons in sorted(dropped.items())},
        'exclusions': exclusions,
        'flagged_but_kept': load_exclusions(exclude_path, scene, 'flagged_but_kept'),
        'exclusion_boxes': len(excluded_keys),
        'exclusion_boxes_the_automatic_checks_find_alone': len(found_alone),
        'exclusion_boxes_the_automatic_checks_miss': sorted(
            f'{t}:{n}' for t, n in excluded_keys - set(found_alone)),
        'automatic_drops_that_appear_only_without_the_exclusion_list': [
            f'{t}:{n}' for t, n in extra_alone],
        'repeated_frame_pairs': [list(pair) for pair in frozen_pairs],
        'secondary_chains_kept': secondary,
        'contained_boxes_flagged': contained,
    }


def estimate_frame_visibility(frames):
    """Set obj['vis'] for every object: 1.0 unless the source frame clips the box.

    The full size of a clipped box is taken as the median width and height of the unclipped boxes
    of the same track (fallback: same class in the scene, then the box itself).
    """
    by_track = collections.defaultdict(list)
    by_class = collections.defaultdict(list)
    for fr in frames:
        for obj in fr['objects']:
            if not obj['frame_sides']:
                dims = (obj['box'][2] - obj['box'][0], obj['box'][3] - obj['box'][1])
                by_track[obj['track']].append(dims)
                by_class[obj['name']].append(dims)
    for fr in frames:
        for obj in fr['objects']:
            obj['vis'] = 1.0
            obj['full_dims_source'] = 'box'
            if not obj['frame_sides']:
                continue
            w, h = obj['box'][2] - obj['box'][0], obj['box'][3] - obj['box'][1]
            if by_track[obj['track']]:
                reference, obj['full_dims_source'] = by_track[obj['track']], 'track'
            elif by_class[obj['name']]:
                reference, obj['full_dims_source'] = by_class[obj['name']], 'class'
            else:
                continue
            full_w, full_h = np.median(np.array(reference, float), axis=0)
            obj['vis'] = float(min(1.0, w / full_w) * min(1.0, h / full_h))


def find_holes(frame_img):
    """Quarter resolution mask of unobserved regions, or None.

    Unobserved means exact zeros over an area at least 256 source px wide and high. A hangar is
    black too, but no organiser object reaches 256 px, so objects survive the erosion.
    """
    zero = (frame_img[::4, ::4].max(axis=2) == 0).astype(np.uint8)
    if not zero.any():
        return None
    kernel = np.ones((HOLE_MIN_SIDE // 4, HOLE_MIN_SIDE // 4), np.uint8)
    holes = cv2.dilate(cv2.erode(zero, kernel), kernel)
    return holes if holes.any() else None


def touches_hole(holes, level, view_center, ox, oy, s):
    """True when the source footprint of a window overlaps an unobserved region."""
    if holes is None:
        return False
    f = FACTOR[level]
    vx, vy = view_origin(level, view_center)
    x1, y1 = vx + ox * f, vy + oy * f
    x2, y2 = x1 + WINDOW * s, y1 + WINDOW * s
    x1, y1 = max(x1, 0) // 4, max(y1, 0) // 4
    x2, y2 = -(-min(x2, IMAGE_WIDTH) // 4), -(-min(y2, IMAGE_HEIGHT) // 4)
    return bool(holes[y1:y2, x1:x2].any())


# --------------------------------------------------------------------------- #
# Sample store
# --------------------------------------------------------------------------- #

class SampleStore:
    """Column lists for the npz container plus a few bookkeeping columns for statistics."""

    SAVED = ('x', 'label', 'level', 'scale', 'size_px', 'box_w', 'box_h', 'off', 'scene', 'frame',
             'track', 'source', 'vis', 'vis_w', 'vis_h', 'vis_off', 'view', 'status', 'fill')
    BOOKKEEPING = ('kind',)

    def __init__(self):
        self.columns = {key: [] for key in self.SAVED + self.BOOKKEEPING}

    def add(self, **row):
        for key in self.columns:
            self.columns[key].append(row[key])

    def __len__(self):
        return len(self.columns['label'])

    def arrays(self):
        c = self.columns
        return {
            'x': np.stack(c['x']).astype(np.uint8),
            'label': np.array(c['label'], np.int16),
            'level': np.array(c['level'], np.int8),
            'scale': np.array(c['scale'], np.int8),
            'size_px': np.array(c['size_px'], np.float32),
            'box_w': np.array(c['box_w'], np.float32),
            'box_h': np.array(c['box_h'], np.float32),
            'off': np.array(c['off'], np.float32).reshape(-1, 2),
            'scene': np.array(c['scene'], dtype='<U16'),
            'frame': np.array(c['frame'], np.int32),
            'track': np.array(c['track'], dtype='<U48'),
            'source': np.array(c['source'], dtype='<U10'),
            'vis': np.array(c['vis'], np.float32),
            'vis_w': np.array(c['vis_w'], np.float32),
            'vis_h': np.array(c['vis_h'], np.float32),
            'vis_off': np.array(c['vis_off'], np.float32).reshape(-1, 2),
            'view': np.array(c['view'], np.int16).reshape(-1, 2),
            'status': np.array(c['status'], dtype='<U40'),
            'fill': np.array(c['fill'], np.float32),
        }


# --------------------------------------------------------------------------- #
# Positives
# --------------------------------------------------------------------------- #

def draw_jitter(rng, jitter_px, m):
    """Object offset in delivered px, uniform within +-jitter_px window px, never (0, 0)."""
    limit = jitter_px * m
    while True:
        jx, jy = (int(v) for v in rng.integers(-limit, limit + 1, size=2))
        if (jx, jy) != (0, 0):
            return jx, jy


def emit_windows(store, frame_img, holes, scene, fr, obj, level, view_center, visible_box, vis,
                 kind, n_jittered, args, rng, negative_requests, verify_pool, stats):
    """One centred window and n_jittered jittered windows of one object seen from view_center.

    box_w, box_h, size_px, off and the scale s describe the annotated box obj['box']. The window
    is centred on visible_box, the part of that box inside the view (the same box when the view
    holds the whole object), which vis_w, vis_h and vis_off describe.
    """
    assert is_legal_view(level, view_center), (level, view_center)
    f = FACTOR[level]
    full_box = obj['box']
    w, h = full_box[2] - full_box[0], full_box[3] - full_box[1]
    vis_w, vis_h = visible_box[2] - visible_box[0], visible_box[3] - visible_box[1]
    s, fallback = pick_scale(max(w, h), level)
    differs = s != pick_scale(max(vis_w, vis_h), level)[0]
    m = s // f
    center = ((visible_box[0] + visible_box[2]) / 2.0, (visible_box[1] + visible_box[3]) / 2.0)
    # Where the centre of the annotated box sits relative to the centre of the visible part.
    shift = (((full_box[0] + full_box[2]) / 2.0 - center[0]) / s,
             ((full_box[1] + full_box[3]) / 2.0 - center[1]) / s)
    jitters = [(0, 0)]
    while len(jitters) < 1 + n_jittered:
        candidate = draw_jitter(rng, args.jitter, m)
        if candidate not in jitters:
            jitters.append(candidate)
    for jitter_d in jitters:
        ox, oy = window_origin(level, view_center, center, s, jitter_d)
        if touches_hole(holes, level, view_center, ox, oy, s):
            stats['positives_dropped_unobserved'] += 1
            continue
        window, fill = window_from_source(frame_img, level, view_center, ox, oy, s)
        vis_off = (jitter_d[0] / m, jitter_d[1] / m)
        store.add(x=window, label=obj['label'], level=level, scale=s, size_px=max(w, h) / f,
                  box_w=w, box_h=h, off=(vis_off[0] + shift[0], vis_off[1] + shift[1]),
                  scene=scene, frame=fr['frame'], track=obj['track'], source='real', vis=vis,
                  vis_w=vis_w, vis_h=vis_h, vis_off=vis_off, view=view_center,
                  status=obj['status'], kind=kind, fill=fill)
        stats['scale_fallback'] += int(fallback)                    # counted per window
        stats['scale_differs_from_visible_part'] += int(differs)
        verify_pool.append((level, view_center, ox, oy, s, window, fill))
        negative_requests.append({'level': level, 'scale': s, 'w': w, 'h': h, 'kind': kind,
                                  'frame': fr['frame'], 'frame_sides': obj['frame_sides']})


def build_positives(store, frame_img, holes, scene, fr, args, rng, negative_requests, verify_pool,
                    stats):
    for obj in fr['objects']:
        if obj['frame_sides']:
            stats['frame_edge_objects'] += 1
            stats['frame_edge_vis'].append(obj['vis'])
            if obj['vis'] < args.min_visible:
                stats['frame_edge_skipped'] += 1
                continue
        kind = 'frame_edge' if obj['frame_sides'] else 'whole'
        for level in (0, 1, 2):
            f = FACTOR[level]
            if level == 0:
                views, jittered = [(IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2)], args.k
            else:
                phases = [(a, b) for a in range(f) for b in range(f)]
                rng.shuffle(phases)
                views = [sample_whole_view(obj['box'], level, rng, phases[k % len(phases)])
                         for k in range(args.k)]
                views, jittered = [v for v in views if v is not None], 1
            for view_center in views:
                emit_windows(store, frame_img, holes, scene, fr, obj, level, view_center,
                             obj['box'], obj['vis'], kind, jittered, args, rng, negative_requests,
                             verify_pool, stats)
            if level == 0 or obj['frame_sides']:
                continue
            for _ in range(args.partial_views):
                cut = sample_cutting_view(obj['box'], level, rng, args.min_visible, 0.95)
                if cut is None:
                    stats['view_cut_failed'] += 1
                    continue
                view_center, visible_box, fraction = cut
                emit_windows(store, frame_img, holes, scene, fr, obj, level, view_center,
                             visible_box, fraction, 'view_cut', 1, args, rng, negative_requests,
                             verify_pool, stats)


# --------------------------------------------------------------------------- #
# Negatives
# --------------------------------------------------------------------------- #

def point_box_distance(px, py, box):
    dx = max(box[0] - px, 0.0, px - box[2])
    dy = max(box[1] - py, 0.0, py - box[3])
    return math.hypot(dx, dy)


def negative_centre_ok(cx, cy, s, objects, scene, band, args):
    """SPEC rule for a negative window centre, in source px.

    objects holds the labelled boxes and also the boxes that the label filters dropped (with
    their keep-out box), because a misplaced box still means an object is near.
    """
    for obj in objects:
        x1, y1, x2, y2 = obj.get('keepout_box', obj['box'])
        keep_out = max(0.75 * max(x2 - x1, y2 - y1), args.neg_guard * s)
        if math.hypot(cx - (x1 + x2) / 2.0, cy - (y1 + y2) / 2.0) < keep_out:
            return False
        if (scene == 'validation'
                and point_box_distance(cx, cy, (x1, y1, x2, y2)) < VALIDATION_CLEARANCE):
            return False
    if scene == 'validation' and not (band[0] <= cy <= band[1]):
        return False
    return True


def propose_negative_box(request, objects, scene, rng, args, attempt):
    """A pseudo box, sized like the matched positive, somewhere in the frame: (box, how)."""
    w, h = request['w'], request['h']          # the annotated box of the matched positive
    s = request['scale']
    sides = request['frame_sides']
    how = 'uniform'
    cx = rng.uniform(w / 2.0, IMAGE_WIDTH - w / 2.0)
    cy = rng.uniform(h / 2.0, IMAGE_HEIGHT - h / 2.0)
    if sides and attempt < args.neg_tries // 2:
        # Matched positive is clipped by the frame: put the pseudo box on the same frame side so
        # that zero fill is not a give-away for "object".
        how = 'frame_edge'
        side = sides[0]
        if side == 'left':
            cx = w / 2.0
        elif side == 'right':
            cx = IMAGE_WIDTH - 1 - w / 2.0
        elif side == 'top':
            cy = h / 2.0
        else:
            cy = IMAGE_HEIGHT - 1 - h / 2.0
    elif objects and rng.random() < args.near_frac:
        how = 'near_object'
        obj = objects[int(rng.integers(len(objects)))]
        x1, y1, x2, y2 = obj['box']
        keep_out = max(0.75 * max(x2 - x1, y2 - y1), args.neg_guard * s)
        if scene == 'validation':
            keep_out = max(keep_out, VALIDATION_CLEARANCE + 0.5 * max(x2 - x1, y2 - y1))
        radius = rng.uniform(keep_out, keep_out + 32.0 * s)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        cx = (x1 + x2) / 2.0 + radius * math.cos(angle)
        cy = (y1 + y2) / 2.0 + radius * math.sin(angle)
    box = (int(round(cx - w / 2.0)), int(round(cy - h / 2.0)))
    box = (box[0], box[1], box[0] + int(w), box[1] + int(h))
    if box[0] < 0 or box[1] < 0 or box[2] > IMAGE_WIDTH or box[3] > IMAGE_HEIGHT:
        return None, how
    return box, how


def build_negatives(store, frame_img, holes, scene, fr, requests, args, rng, stats):
    """Serve as many pending requests as this frame allows. Returns the requests left over."""
    objects = fr['objects']
    band = None
    if scene == 'validation':
        if not objects:
            return requests
        band = (min(o['box'][1] for o in objects), max(o['box'][3] for o in objects))
    left_over = []
    for request in requests:
        level, s = request['level'], request['scale']
        done = False
        for attempt in range(args.neg_tries):
            box, how = propose_negative_box(request, objects, scene, rng, args, attempt)
            if box is None:
                continue
            if request['kind'] == 'view_cut':
                cut = sample_cutting_view(box, level, rng, args.min_visible, 0.95)
                if cut is None:
                    continue
                view_center, target, _ = cut
                # keep the scale of the matched positive even if the pseudo cut differs a little
            else:
                view_center = sample_whole_view(box, level, rng)
                target = box
                if view_center is None:
                    continue
            cx, cy = (target[0] + target[2]) / 2.0, (target[1] + target[3]) / 2.0
            if not negative_centre_ok(cx, cy, s, objects + fr['ignored'], scene, band, args):
                continue
            ox, oy = window_origin(level, view_center, (cx, cy), s)
            if touches_hole(holes, level, view_center, ox, oy, s):
                continue
            window, fill = window_from_source(frame_img, level, view_center, ox, oy, s)
            store.add(x=window, label=BACKGROUND, level=level, scale=s,
                      size_px=max(request['w'], request['h']) / FACTOR[level],
                      box_w=request['w'], box_h=request['h'], off=(0.0, 0.0), scene=scene,
                      frame=fr['frame'], track='background', source='real', vis=1.0,
                      vis_w=request['w'], vis_h=request['h'], vis_off=(0.0, 0.0),
                      view=view_center, status='', kind='negative_' + how, fill=fill)
            stats['negative_how'][how] += 1
            stats['negatives_served_in_other_frame'] += int(request['frame'] != fr['frame'])
            done = True
            break
        if not done:
            left_over.append(request)
    return left_over


# --------------------------------------------------------------------------- #
# Exactness check
# --------------------------------------------------------------------------- #

def verify_frame(frame_img, verify_pool, rng, result):
    """Compare shortcut windows of this frame with windows cut from fully rendered views."""
    by_level = collections.defaultdict(list)
    for item in verify_pool:
        by_level[item[0]].append(item)
    for level, items in by_level.items():
        chosen = [items[int(rng.integers(len(items)))]]
        with_fill = [it for it in items if it[6] > 0]
        if with_fill:
            chosen.append(with_fill[int(rng.integers(len(with_fill)))])
        reduced = [it for it in items if it[4] > FACTOR[level]]
        if reduced:
            chosen.append(reduced[int(rng.integers(len(reduced)))])
        for _, view_center, ox, oy, s, window, fill in chosen:
            view_img = render_view_like_evaluator(frame_img, level, view_center)
            reference = window_from_view(view_img, ox, oy, s, FACTOR[level])
            direct = window_from_source_direct(frame_img, level, view_center, ox, oy, s)
            two_stage = int(np.abs(reference.astype(np.int16) - window.astype(np.int16)).max())
            one_stage = int(np.abs(reference.astype(np.int16) - direct.astype(np.int16)).max())
            entry = result['per_level'].setdefault(str(level), {'checked': 0, 'two_stage_max': 0,
                                                                'direct_max': 0, 'with_fill': 0})
            entry['checked'] += 1
            entry['with_fill'] += int(fill > 0)
            entry['two_stage_max'] = max(entry['two_stage_max'], two_stage)
            entry['direct_max'] = max(entry['direct_max'], one_stage)
            if one_stage:
                diff = np.abs(reference.astype(np.int16) - direct.astype(np.int16))
                result['direct_pixels_off_by_one'] += int((diff > 0).sum())
            result['pixels_compared'] += int(reference.size)


# --------------------------------------------------------------------------- #
# Statistics, manifest, contact sheets
# --------------------------------------------------------------------------- #

def quantiles(values):
    v = np.asarray(values, float)
    return {'n': int(v.size), 'min': round(float(v.min()), 2),
            'p25': round(float(np.percentile(v, 25)), 2),
            'median': round(float(np.median(v)), 2),
            'p75': round(float(np.percentile(v, 75)), 2), 'max': round(float(v.max()), 2)}


def summarise(arrays, kinds, fills):
    """Counts, size_px tables (all positives, and whole objects only) and scale mixes."""
    label, level, scale, size_px = (arrays[k] for k in ('label', 'level', 'scale', 'size_px'))
    whole = arrays['vis'] >= 1.0
    counts, sizes, sizes_whole, scales = {}, {}, {}, {}
    for idx, name in enumerate(ROW_NAMES):
        counts[name] = {f'L{L}': int(((label == idx) & (level == L)).sum()) for L in (0, 1, 2)}
        if idx == BACKGROUND:
            continue
        sizes[name], sizes_whole[name], scales[name] = {}, {}, {}
        for L in (0, 1, 2):
            mask = (label == idx) & (level == L)
            if mask.any():
                sizes[name][f'L{L}'] = quantiles(size_px[mask])
                values, n = np.unique(scale[mask], return_counts=True)
                scales[name][f'L{L}'] = {f's{int(v)}': int(c) for v, c in zip(values, n)}
            if (mask & whole).any():
                sizes_whole[name][f'L{L}'] = quantiles(size_px[mask & whole])
    positives = label != BACKGROUND
    per_level = {}
    for L in (0, 1, 2):
        mask = positives & (level == L)
        hashes = {hash(arrays['x'][i].tobytes()) for i in np.flatnonzero(mask)}
        values, n = np.unique(scale[mask], return_counts=True)
        neg_values, neg_n = np.unique(scale[~positives & (level == L)], return_counts=True)
        per_level[f'L{L}'] = {
            'positives': int(mask.sum()),
            'negatives': int((~positives & (level == L)).sum()),
            'unique_positive_windows': len(hashes),
            'positive_scale_mix': {f's{int(v)}': int(c) for v, c in zip(values, n)},
            'negative_scale_mix': {f's{int(v)}': int(c) for v, c in zip(neg_values, neg_n)},
            'positives_with_zero_fill': int((mask & (fills > 0)).sum()),
            'negatives_with_zero_fill': int((~positives & (level == L) & (fills > 0)).sum()),
        }
    kind_counts = {k: int(c) for k, c in zip(*np.unique(kinds, return_counts=True))}
    return counts, sizes, sizes_whole, scales, per_level, kind_counts


def print_report(scene, counts, sizes, sizes_whole, scales, per_level):
    print(f'\n=== {scene}: windows per class x level ===')
    print(f'{"class":18s} {"L0":>7s} {"L1":>7s} {"L2":>7s}')
    for name in ROW_NAMES:
        row = counts[name]
        print(f'{name:18s} {row["L0"]:7d} {row["L1"]:7d} {row["L2"]:7d}')
    print(f'\n=== {scene}: size_px of the annotated box (delivered px, min / median / max) and '
          f'chosen scale s, ALL positives ===')
    for name in OBJECT_CLASSES:
        if not sizes.get(name):
            continue
        parts = []
        for L in (0, 1, 2):
            key = f'L{L}'
            if key in sizes[name]:
                q = sizes[name][key]
                mix = ' '.join(f'{k}:{v}' for k, v in scales[name][key].items())
                parts.append(f'{key} {q["min"]:6.1f}/{q["median"]:6.1f}/{q["max"]:6.1f} [{mix}]')
        print(f'{name:18s} ' + ' | '.join(parts))
    print(f'\n=== {scene}: size_px, positives with vis == 1 only (n, min / median / max) ===')
    for name in OBJECT_CLASSES:
        if not sizes_whole.get(name):
            continue
        parts = []
        for L in (0, 1, 2):
            key = f'L{L}'
            if key in sizes_whole[name]:
                q = sizes_whole[name][key]
                parts.append(f'{key} n={q["n"]:5d} {q["min"]:6.1f}/{q["median"]:6.1f}/'
                             f'{q["max"]:6.1f}')
        print(f'{name:18s} ' + ' | '.join(parts))
    print(f'\n=== {scene}: per level ===')
    for key, row in per_level.items():
        print(key, json.dumps(row))


def caption_tile(window, text, zoom=3, strip=14):
    tile = cv2.resize(window, (WINDOW * zoom, WINDOW * zoom), interpolation=cv2.INTER_NEAREST)
    bar = np.full((strip, tile.shape[1], 3), 32, np.uint8)
    cv2.putText(bar, text, (2, strip - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (230, 230, 230), 1,
                cv2.LINE_AA)
    return np.vstack([tile, bar])


def spread(indices, n):
    """n entries spread evenly over a list of indices."""
    if len(indices) <= n:
        return list(indices)
    picks = np.linspace(0, len(indices) - 1, n).round().astype(int)
    return [indices[i] for i in picks]


def write_contact_sheet(path, arrays, fills, select, per_level=5, title=''):
    """One row per class. select(class_mask_indices, level) -> ordered sample indices to show."""
    zoom, strip, label_w, header_h = 3, 14, 170, 22
    tile_w, tile_h = WINDOW * zoom, WINDOW * zoom + strip
    columns = 3 * per_level
    sheet = np.full((header_h + len(ROW_NAMES) * tile_h, label_w + columns * tile_w, 3), 16,
                    np.uint8)
    cv2.putText(sheet, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                cv2.LINE_AA)
    for L in (0, 1, 2):
        x = label_w + L * per_level * tile_w
        cv2.putText(sheet, f'level {L} (f={FACTOR[L]})', (x + 4, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (120, 220, 255), 1, cv2.LINE_AA)
    for row, name in enumerate(ROW_NAMES):
        y = header_h + row * tile_h
        n = int((arrays['label'] == row).sum())
        cv2.putText(sheet, name, (4, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(sheet, f'n={n}', (4, y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170),
                    1, cv2.LINE_AA)
        for L in (0, 1, 2):
            indices = np.flatnonzero((arrays['label'] == row) & (arrays['level'] == L))
            for col, i in enumerate(select(indices, L)[:per_level]):
                off = arrays['off'][i]
                mark = 'c' if not off.any() else 'j'
                text = (f's{arrays["scale"][i]} {arrays["size_px"][i]:.0f}px f{arrays["frame"][i]}'
                        f' {mark} v{arrays["vis"][i]:.2f}')
                tile = caption_tile(arrays['x'][i], text, zoom, strip)
                x = label_w + (L * per_level + col) * tile_w
                sheet[y: y + tile_h, x: x + tile_w] = tile
    cv2.imwrite(str(path), sheet)


class DroppedLabelSheet:
    """Source crops of the boxes that the label filters dropped, next to their kept neighbours.

    One row per track that lost boxes. Red = dropped (caption says why), green = kept. The crops
    come straight from the source frame, so a human can see whether the object is there.
    """

    TILE, STRIP, PER_ROW = 176, 14, 12

    def __init__(self, frames):
        by_track = collections.defaultdict(list)
        for fr in frames:
            for obj in fr['objects'] + fr['ignored']:
                by_track[obj['track']].append((fr['frame'], obj))
        self.plan = collections.defaultdict(list)       # frame -> [(track, column, obj)]
        self.rows = {}                                  # track -> number of columns
        for track, rows in sorted(by_track.items(), key=lambda item: item[1][0][0]):
            rows.sort(key=lambda row: row[0])
            flags = [obj.get('drop_reason') is not None for _, obj in rows]
            if not any(flags):
                continue
            # dropped boxes, plus the two kept boxes on either side of every dropped run
            wanted = set()
            for i, flag in enumerate(flags):
                if flag:
                    wanted.update(j for j in range(i - 2, i + 3) if 0 <= j < len(rows))
            dropped = [i for i in sorted(wanted) if flags[i]]
            kept = [i for i in sorted(wanted) if not flags[i]]
            room = max(self.PER_ROW - len(kept), 2)
            chosen = sorted(set(spread(dropped, room)) | set(kept))[:self.PER_ROW]
            self.rows[track] = len(chosen)
            for column, i in enumerate(chosen):
                self.plan[rows[i][0]].append((track, column, rows[i][1]))
        self.tiles = {}

    def frames_needed(self):
        return set(self.plan)

    def collect(self, number, frame_img):
        for track, column, obj in self.plan.get(number, []):
            x1, y1, x2, y2 = obj['box']
            side = int(max(192, 2.5 * max(x2 - x1, y2 - y1)))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            ox, oy = cx - side // 2, cy - side // 2
            crop = np.zeros((side, side, 3), np.uint8)
            ax, bx = clip_span(ox, side, IMAGE_WIDTH)
            ay, by = clip_span(oy, side, IMAGE_HEIGHT)
            crop[ay - oy: by - oy, ax - ox: bx - ox] = frame_img[ay:by, ax:bx]
            zoom = self.TILE / float(side)
            tile = cv2.resize(crop, (self.TILE, self.TILE), interpolation=cv2.INTER_AREA)
            reason = obj.get('drop_reason')
            colour = (0, 0, 255) if reason else (0, 220, 0)
            cv2.rectangle(tile, (int((x1 - ox) * zoom), int((y1 - oy) * zoom)),
                          (int((x2 - ox) * zoom), int((y2 - oy) * zoom)), colour, 1)
            bar = np.full((self.STRIP, self.TILE, 3), 32, np.uint8)
            cv2.putText(bar, f'f{number} {reason or "kept"} {x2 - x1}x{y2 - y1}', (2, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, colour, 1, cv2.LINE_AA)
            self.tiles[(track, column)] = np.vstack([tile, bar])

    def write(self, path, title):
        if not self.rows:
            return False
        label_w, header_h = 250, 22
        tile_h = self.TILE + self.STRIP
        sheet = np.full((header_h + len(self.rows) * tile_h, label_w + self.PER_ROW * self.TILE, 3),
                        16, np.uint8)
        cv2.putText(sheet, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
        for r, (track, columns) in enumerate(self.rows.items()):
            y = header_h + r * tile_h
            cv2.putText(sheet, track, (4, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)
            for column in range(columns):
                tile = self.tiles.get((track, column))
                if tile is not None:
                    x = label_w + column * self.TILE
                    sheet[y: y + tile_h, x: x + self.TILE] = tile
        cv2.imwrite(str(path), sheet)
        return True


def main_sheet_selector(arrays, per_level):
    def select(indices, level):
        centred = [i for i in indices if not arrays['off'][i].any() and arrays['vis'][i] >= 1.0]
        jittered = [i for i in indices if arrays['off'][i].any() and arrays['vis'][i] >= 1.0]
        n_centred = per_level - 2 if jittered else per_level
        return spread(centred, n_centred) + spread(jittered, per_level - n_centred)
    return select


def edge_sheet_selector(arrays, fills, per_level):
    def select(indices, level):
        cut = [i for i in indices if arrays['vis'][i] < 1.0]
        filled = [i for i in indices if fills[i] > 0 and arrays['vis'][i] >= 1.0]
        filled.sort(key=lambda i: -fills[i])
        half = per_level // 2 + 1
        return spread(cut, half) + spread(filled[:40], per_level - min(half, len(cut)))
    return select


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def build_scene(scene, args):
    started = time.time()
    rng = np.random.default_rng(args.seed)
    args.k = args.views if args.views else (6 if scene == 'helsinki' else 3)
    label_options = None if args.no_label_filters else {
        'exclude_path': args.exclude, 'min_area_ratio': args.min_area_ratio,
        'min_chain': args.min_chain, 'drop_contained': args.drop_contained}
    frames, skipped, label_report = load_scene(scene, args.max_frames, label_options)
    estimate_frame_visibility(frames)
    verify_every = max(1, len(frames) // max(1, args.verify_frames))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dropped_sheet = DroppedLabelSheet(frames)
    dropped_path = out_dir / f'real_{scene}_dropped.png'
    if label_report is not None:
        print(f'\n=== {scene}: label filters ===')
        print(json.dumps({k: label_report[k] for k in (
            'boxes_before', 'boxes_dropped', 'dropped_by_reason', 'dropped_by_class',
            'dropped_by_review_status', 'dropped_by_track', 'exclusion_boxes',
            'exclusion_boxes_the_automatic_checks_find_alone',
            'exclusion_boxes_the_automatic_checks_miss',
            'automatic_drops_that_appear_only_without_the_exclusion_list',
            'repeated_frame_pairs', 'secondary_chains_kept', 'contained_boxes_flagged')},
            indent=1))
    if args.labels_only:
        for fr in frames:
            if fr['frame'] in dropped_sheet.frames_needed():
                dropped_sheet.collect(fr['frame'], cv2.imread(str(fr['path']), cv2.IMREAD_COLOR))
        if dropped_sheet.write(dropped_path, f'real_{scene}: boxes dropped by the label filters '
                                             f'(red) and their kept neighbours (green)'):
            print('wrote', dropped_path)
        return label_report

    store = SampleStore()
    stats = {'scale_fallback': 0, 'positives_dropped_unobserved': 0, 'frames_with_holes': [],
             'frame_edge_objects': 0, 'frame_edge_skipped': 0, 'frame_edge_vis': [],
             'view_cut_failed': 0, 'scale_differs_from_visible_part': 0,
             'negative_how': collections.Counter(), 'negatives_served_in_other_frame': 0}
    verification = {'per_level': {}, 'pixels_compared': 0, 'direct_pixels_off_by_one': 0,
                    'frames_checked': 0}
    pending = []
    for n, fr in enumerate(frames):
        frame_img = cv2.imread(str(fr['path']), cv2.IMREAD_COLOR)
        if frame_img is None or frame_img.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
            skipped.append((fr['frame'], 'unreadable or not 3840x2160'))
            continue
        holes = find_holes(frame_img)
        if holes is not None:
            stats['frames_with_holes'].append(fr['frame'])
        dropped_sheet.collect(fr['frame'], frame_img)
        requests, verify_pool = [], []
        build_positives(store, frame_img, holes, scene, fr, args, rng, requests, verify_pool,
                        stats)
        if args.verify_frames and n % verify_every == 0 and verify_pool:
            verify_frame(frame_img, verify_pool, rng, verification)
            verification['frames_checked'] += 1
        pending = build_negatives(store, frame_img, holes, scene, fr, pending + requests, args,
                                  rng, stats)
        if (n + 1) % 25 == 0 or n + 1 == len(frames):
            print(f'  {scene}: frame {n + 1}/{len(frames)}, {len(store)} windows, '
                  f'{len(pending)} negatives pending, {time.time() - started:.0f} s', flush=True)

    # Requests that the last frames could not serve go to earlier frames, newest first.
    for fr in reversed(frames):
        if not pending:
            break
        frame_img = cv2.imread(str(fr['path']), cv2.IMREAD_COLOR)
        pending = build_negatives(store, frame_img, find_holes(frame_img), scene, fr, pending,
                                  args, rng, stats)

    arrays = store.arrays()
    kinds = np.array(store.columns['kind'])
    fills = arrays['fill']
    counts, sizes, sizes_whole, scales, per_level, kind_counts = summarise(arrays, kinds, fills)
    print_report(scene, counts, sizes, sizes_whole, scales, per_level)

    npz_path = out_dir / f'real_{scene}.npz'
    np.savez_compressed(npz_path, **arrays)

    positives = arrays['label'] != BACKGROUND
    vis = arrays['vis']
    view_cut = kinds == 'view_cut'
    frame_edge = kinds == 'frame_edge'
    status_counts = {k: int(c) for k, c in zip(*np.unique(arrays['status'][positives],
                                                          return_counts=True))}
    manifest = {
        'file': npz_path.name,
        'scene': scene,
        'source': 'real',
        'trust': ('organiser ground truth' if scene == 'helsinki' else
                  'participant pseudo-labels: noisy and INCOMPLETE. Negatives are NOISY: an '
                  'unlabelled region is not proof of emptiness.'),
        'n_windows': int(len(arrays['label'])),
        'n_positives': int(positives.sum()),
        'n_negatives': int((~positives).sum()),
        'negatives_not_placed': len(pending),
        'frames_used': len(frames),
        'frames_skipped': [{'frame': f, 'reason': r} for f, r in skipped],
        'parameters': {
            'views_per_object_level': args.k, 'partial_views_per_object_level': args.partial_views,
            'jitter_window_px': args.jitter, 'min_visible': args.min_visible,
            'neg_guard_window_px': args.neg_guard, 'near_frac': args.near_frac,
            'neg_tries': args.neg_tries, 'seed': args.seed, 'window': WINDOW,
            'max_object_px': MAX_OBJECT_PX, 'validation_clearance_src_px': VALIDATION_CLEARANCE,
            'max_frames': args.max_frames,
        },
        'conventions': {
            'level0': 'single legal view (1920,1080): one centred window and K jittered windows '
                      'per object',
            'levels_1_2': 'K legal views containing the whole box (L1 cycles the four f-grid '
                          'phases), each gives one centred and one jittered window, plus '
                          'partial_views view-cut samples',
            'off': '(dx, dy) of the object centre from the window centre in window px, (0,0) when '
                   'centred',
            'annotated_box': 'box_w, box_h, size_px, off and the scale s always describe the '
                             'annotated box, also for view-cut samples',
            'partial_objects': 'vis < 1. View-cut objects: the window is centred on the visible '
                               'part of the box; vis_w, vis_h (source px) and vis_off (window px, '
                               'equal to the jitter) describe that part, off locates the centre of '
                               'the annotated box and is therefore not (0,0) for a centred cut '
                               'window. Frame-edge objects are annotated clipped, so their '
                               'box_w, box_h, size_px are those of the clipped box and '
                               'vis_w, vis_h repeat them',
            'negatives': 'label 16, track "background". size_px, box_w, box_h are those of the '
                         'matched positive (they fixed level and scale), vis is 1.0',
            'extra_arrays': ['vis', 'vis_w', 'vis_h', 'vis_off', 'view', 'status', 'fill'],
        },
        'label_filters': label_report,
        'counts_class_x_level': counts,
        'size_px_by_class_level': sizes,
        'size_px_by_class_level_vis_eq_1': sizes_whole,
        'size_px_note': 'size_px is the longer side of the annotated box in delivered px. The '
                        'first table holds all positives (whole, view_cut and frame_edge), the '
                        'second only vis == 1. View-cut samples carry the full box, so only '
                        'frame_edge samples (boxes annotated clipped) can lower the first table',
        'scale_by_class_level': scales,
        'per_level': per_level,
        'sample_kinds': kind_counts,
        'visible_fraction': {
            'frame_edge_objects': stats['frame_edge_objects'],
            'frame_edge_objects_skipped_below_min_visible': stats['frame_edge_skipped'],
            'frame_edge_object_vis': (quantiles(stats['frame_edge_vis'])
                                      if stats['frame_edge_vis'] else None),
            'frame_edge_windows': int(frame_edge.sum()),
            'frame_edge_window_vis': quantiles(vis[frame_edge]) if frame_edge.any() else None,
            'view_cut_windows': int(view_cut.sum()),
            'view_cut_window_vis': quantiles(vis[view_cut]) if view_cut.any() else None,
            'view_cut_failed': stats['view_cut_failed'],
            'histogram_vis_lt_1': {f'{lo:.1f}-{lo + 0.1:.1f}': int(((vis >= lo) & (vis < lo + 0.1)
                                                                   & positives).sum())
                                   for lo in (0.6, 0.7, 0.8, 0.9)},
        },
        'zero_fill': {
            'whole_object_windows': int((kinds == 'whole').sum()),
            'whole_object_windows_with_fill': int(((kinds == 'whole') & (fills > 0)).sum()),
            'whole_object_windows_fill_over_25pct': int(((kinds == 'whole')
                                                         & (fills > 0.25)).sum()),
            'fill_of_filled_whole_windows': (quantiles(fills[(kinds == 'whole') & (fills > 0)])
                                             if ((kinds == 'whole') & (fills > 0)).any() else None),
        },
        'scale_fallback_windows': stats['scale_fallback'],
        'windows_whose_scale_differs_from_the_scale_of_the_visible_part':
            stats['scale_differs_from_visible_part'],
        'unobserved_regions': {'frames_with_holes': stats['frames_with_holes'],
                               'positive_windows_dropped': stats['positives_dropped_unobserved']},
        'negative_placement': dict(stats['negative_how']),
        'negatives_served_in_other_frame': stats['negatives_served_in_other_frame'],
        'review_status_of_positives': status_counts,
        'exactness_check': verification,
        'seconds': round(time.time() - started, 1),
    }
    npz_path.with_suffix('.json').write_text(json.dumps(manifest, indent=1))

    per = 5
    write_contact_sheet(out_dir / f'real_{scene}.png', arrays, fills,
                        main_sheet_selector(arrays, per), per,
                        f'real_{scene}: whole objects')
    write_contact_sheet(out_dir / f'real_{scene}_edges.png', arrays, fills,
                        edge_sheet_selector(arrays, fills, per), per,
                        f'real_{scene}: cut / zero fill')
    if dropped_sheet.write(dropped_path, f'real_{scene}: boxes dropped by the label filters (red) '
                                         f'and their kept neighbours (green)'):
        print('wrote', dropped_path)
    print(f'\n{scene}: wrote {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB), '
          f'{manifest["n_positives"]} positives, {manifest["n_negatives"]} negatives, '
          f'{len(pending)} negatives not placed, {manifest["seconds"]} s')
    print('exactness check:', json.dumps(verification))
    print('visible fraction:', json.dumps(manifest['visible_fraction']))
    print('zero fill:', json.dumps(manifest['zero_fill']))
    print('scale fallback windows:', stats['scale_fallback'],
          '| negatives served in another frame:', stats['negatives_served_in_other_frame'],
          '| frames skipped:', len(skipped),
          '| frames with unobserved holes:', stats['frames_with_holes'],
          '| positives dropped there:', stats['positives_dropped_unobserved'])
    if status_counts and scene == 'validation':
        print('review_status of positives:', json.dumps(status_counts))
    print('negative placement:', json.dumps(manifest['negative_placement']),
          'kinds:', json.dumps(kind_counts))
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--scene', choices=['helsinki', 'validation', 'both'], default='both')
    parser.add_argument('--views', type=int, default=None,
                        help='K views per (object, level). Default: 6 on helsinki, 3 on validation')
    parser.add_argument('--partial-views', type=int, default=1,
                        help='extra view-cut samples per (object, level 1 or 2)')
    parser.add_argument('--jitter', type=int, default=6, help='jitter range in window px')
    parser.add_argument('--min-visible', type=float, default=0.6,
                        help='least visible box area fraction for a positive')
    parser.add_argument('--exclude', default=str(DEFAULT_EXCLUDE),
                        help='JSON list of hand-checked (track, first, last) label exclusions')
    parser.add_argument('--min-area-ratio', type=float, default=0.5,
                        help='drop a box the frame does not clip below this share of the track '
                             'median area')
    parser.add_argument('--min-chain', type=int, default=8,
                        help='a motion chain that is not the longest of its track survives with '
                             'at least this many boxes')
    parser.add_argument('--drop-contained', action='store_true',
                        help='also drop boxes that lie at least 90 %% inside a box of another '
                             'class (default: only report them)')
    parser.add_argument('--no-label-filters', action='store_true',
                        help='keep every pseudo-label box (the behaviour before the filters)')
    parser.add_argument('--labels-only', action='store_true',
                        help='run the label filters, write real_<scene>_dropped.png and stop')
    parser.add_argument('--neg-guard', type=float, default=10.0,
                        help='negatives stay at least this many window px from any object centre '
                             '(keeps them apart from jittered positives)')
    parser.add_argument('--near-frac', type=float, default=0.25,
                        help='share of negatives proposed next to an object instead of uniformly')
    parser.add_argument('--neg-tries', type=int, default=60)
    parser.add_argument('--verify-frames', type=int, default=25,
                        help='frames on which windows are checked against full renders, 0 = off')
    parser.add_argument('--max-frames', type=int, default=None, help='debug: subsample frames')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', default=str(ROOT / 'elias' / 'out'))
    return parser.parse_args()


def main():
    args = parse_args()
    scenes = ['helsinki', 'validation'] if args.scene == 'both' else [args.scene]
    for scene in scenes:
        build_scene(scene, args)


if __name__ == '__main__':
    main()
