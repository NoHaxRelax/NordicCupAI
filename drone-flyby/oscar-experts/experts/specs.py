"""Distinctive measurements per class, recorded on every candidate. None of them is a gate.

Each signature receives ctx (image, lab, L, chroma, gray, high, scale, zoom), the candidate
(with mask_bbox and cx, cy), the matched template and its posed mask, and returns a dict.
Thresholds inside come from the training sprites' own colours, measured once at import.
"""
import cv2
import numpy as np

from .generic import ClassSpec


def _window(ctx, candidate, pad=.15):
    x1, y1, x2, y2 = candidate['mask_bbox']
    w, h = x2 - x1, y2 - y1
    H, W = ctx['L'].shape
    X1, Y1 = int(max(0, x1 - pad * w)), int(max(0, y1 - pad * h))
    X2, Y2 = int(min(W, x2 + pad * w)), int(min(H, y2 + pad * h))
    return X1, Y1, X2, Y2


def _masked_region(ctx, candidate, tmask):
    """Pixels of the posed mask that fall inside the image, as a boolean image window."""
    x1, y1, x2, y2 = [int(round(v)) for v in candidate['mask_bbox']]
    H, W = ctx['L'].shape
    sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    m = tmask[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1]
    return (sx1, sy1, sx2, sy2), m


def red_markings(ctx, candidate, template, tmask):
    """Small plane: red paint on tail and nose. Red = Lab a well above neutral."""
    X1, Y1, X2, Y2 = _window(ctx, candidate, .2)
    lab = ctx['lab'][Y1:Y2, X1:X2]
    red = (lab[:, :, 1] > 150) & (lab[:, :, 0] > 60)
    count, _, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8))
    blobs = [int(a) for a in stats[1:, 4] if a >= 2 * ctx['scale'] ** 2]
    return dict(red_pixels=int(red.sum()), red_blobs=len(blobs), red_fraction=float(red.mean()))


def green_paint(ctx, candidate, template, tmask):
    """Green paint fraction inside the posed mask (jammer panel, launcher, small-tower roof, helicopter body)."""
    (sx1, sy1, sx2, sy2), m = _masked_region(ctx, candidate, tmask)
    lab = ctx['lab'][sy1:sy2, sx1:sx2]
    green = (lab[:, :, 1] < 122) & (lab[:, :, 2] > 136)
    inside = green[m] if m.sum() else np.zeros(0, bool)
    return dict(green_fraction=float(inside.mean()) if inside.size else 0., green_pixels=int(inside.sum()))


def dark_fraction(ctx, candidate, template, tmask):
    (sx1, sy1, sx2, sy2), m = _masked_region(ctx, candidate, tmask)
    L = ctx['L'][sy1:sy2, sx1:sx2][m] if m.sum() else np.zeros(0)
    ring = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~m
    ringL = ctx['L'][sy1:sy2, sx1:sx2][ring] if ring.any() else np.zeros(0)
    return dict(dark_fraction=float((L < 70).mean()) if L.size else 0., interior_L=float(np.median(L)) if L.size else 0.,
                rim_contrast=float(np.median(ringL) - np.median(L)) if L.size and ringL.size else 0.)


def bright_fraction(ctx, candidate, template, tmask):
    (sx1, sy1, sx2, sy2), m = _masked_region(ctx, candidate, tmask)
    L = ctx['L'][sy1:sy2, sx1:sx2][m] if m.sum() else np.zeros(0)
    chroma = ctx['chroma'][sy1:sy2, sx1:sx2][m] if m.sum() else np.zeros(0)
    return dict(bright_fraction=float((L > 190).mean()) if L.size else 0., low_chroma_fraction=float((chroma < 10).mean()) if chroma.size else 0.)


def square_roof(ctx, candidate, template, tmask):
    """Small tower: a green square roof on a pale slab. Squareness and fill of the green component."""
    X1, Y1, X2, Y2 = _window(ctx, candidate, .25)
    lab = ctx['lab'][Y1:Y2, X1:X2]
    green = ((lab[:, :, 1] < 122) & (lab[:, :, 2] > 136)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(green)
    if count < 2:
        return dict(roof_squareness=0., roof_fill=0., roof_side=0.)
    i = 1 + int(np.argmax(stats[1:, 4]))
    pts = np.column_stack(np.where(labels == i))[:, ::-1].astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    side = max(w, h)
    fill = float(stats[i, 4] / max(1., w * h))
    pale = ((lab[:, :, 0] > 170) & (np.hypot(lab[:, :, 1] - 128, lab[:, :, 2] - 128) < 14))
    ring = cv2.dilate((labels == i).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~(labels == i)
    return dict(roof_squareness=float(min(w, h) / max(1., side)), roof_fill=fill, roof_side=float(side / ctx['scale']),
                slab_fraction=float(pale[ring].mean()) if ring.any() else 0.)


def rectangle_panel(ctx, candidate, template, tmask):
    """Jammer: one saturated green rectangle. Aspect and fill of the largest green component."""
    X1, Y1, X2, Y2 = _window(ctx, candidate, .25)
    lab = ctx['lab'][Y1:Y2, X1:X2]
    green = ((lab[:, :, 1] < 118) & (lab[:, :, 2] > 140)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(green)
    if count < 2:
        return dict(panel_aspect=0., panel_fill=0., panel_area=0)
    i = 1 + int(np.argmax(stats[1:, 4]))
    pts = np.column_stack(np.where(labels == i))[:, ::-1].astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    return dict(panel_aspect=float(max(w, h) / max(1., min(w, h))), panel_fill=float(stats[i, 4] / max(1., w * h)), panel_area=int(stats[i, 4]))


def rotor_lines(ctx, candidate, template, tmask):
    """Helicopter: thin dark blades radiating from one hub. Count line segments whose extension passes near a common point."""
    X1, Y1, X2, Y2 = _window(ctx, candidate, .3)
    gray = (ctx['gray'][Y1:Y2, X1:X2] * 255).astype(np.uint8)
    if gray.size < 100:
        return dict(rotor_lines=0, rotor_hub_lines=0)
    dark = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    minimum = max(6, int(.25 * max(gray.shape) ))
    lines = cv2.HoughLinesP(dark, 1, np.pi / 180, threshold=max(8, minimum // 2), minLineLength=minimum, maxLineGap=3)
    if lines is None:
        return dict(rotor_lines=0, rotor_hub_lines=0)
    segs = lines[:, 0, :].astype(np.float32)
    # hub: point minimising distance to all lines (least squares), then count lines passing within r of it
    A, b = [], []
    for x1, y1, x2, y2 in segs:
        d = np.array([x2 - x1, y2 - y1]); n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-6)
        A.append(n); b.append(n @ np.array([x1, y1]))
    A, b = np.array(A), np.array(b)
    hub, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = .12 * max(gray.shape)
    through = int(sum(abs(n @ hub - bb) <= r for n, bb in zip(A, b)))
    angles = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    distinct = len({int(a // 20) for a in angles})
    return dict(rotor_lines=int(len(segs)), rotor_hub_lines=through, rotor_distinct_angles=distinct,
                hub_offset=float(np.hypot(hub[0] + X1 - candidate['cx'], hub[1] + Y1 - candidate['cy']) / ctx['scale']))


def camo_texture(ctx, candidate, template, tmask):
    """Tank, launchers, mine roller, large tower: camouflage = high local variance with green-brown chroma."""
    (sx1, sy1, sx2, sy2), m = _masked_region(ctx, candidate, tmask)
    if m.sum() < 8:
        return dict(camo_std=0., camo_green=0.)
    lab = ctx['lab'][sy1:sy2, sx1:sx2]
    L = lab[:, :, 0][m]; a = lab[:, :, 1][m]; b = lab[:, :, 2][m]
    return dict(camo_std=float(L.std()), camo_green=float(((a < 126) & (b > 132)).mean()), camo_mean_L=float(L.mean()))


def barrel_line(ctx, candidate, template, tmask):
    """Tank: a thin straight feature extending from the hull along the heading."""
    X1, Y1, X2, Y2 = _window(ctx, candidate, .35)
    gray = (ctx['gray'][Y1:Y2, X1:X2] * 255).astype(np.uint8)
    if gray.size < 100:
        return dict(barrel_lines=0)
    edges = cv2.Canny(gray, 40, 120)
    minimum = max(6, int(.3 * max(gray.shape)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(6, minimum // 2), minLineLength=minimum, maxLineGap=2)
    return dict(barrel_lines=0 if lines is None else int(len(lines)))


SPECS = {
    'helicopter': ClassSpec('helicopter', signatures=(rotor_lines, green_paint, camo_texture), competitors=('jammer', 'tank'), sift=True,
                            colour_blob=dict(probability=.3, min_fraction=.2, max_fraction=1.2)),
    'jammer': ClassSpec('jammer', signatures=(rectangle_panel, green_paint), competitors=('small_tower', 'tank'),
                        colour_blob=dict(probability=.4, min_fraction=.4, max_fraction=1.5)),
    'jet_plane': ClassSpec('jet_plane', signatures=(bright_fraction, dark_fraction), competitors=('small_plane', 'condor', 'medium_plane'), sift=True),
    # articulated (tube up/down) and seen from several sides: every reviewed sprite of the zoom takes part (5/12 -> 7/12 at L1)
    'large_launcher': ClassSpec('large_launcher', signatures=(camo_texture, barrel_line), competitors=('tank', 'large_tower', 'mine_roller'), sift=True,
                                fine_templates=6, proposer_templates=6),
    'large_tower': ClassSpec('large_tower', signatures=(camo_texture, bright_fraction), competitors=('large_launcher', 'tank'), sift=True),
    'medium_launcher': ClassSpec('medium_launcher', signatures=(dark_fraction, camo_texture), competitors=('small_launcher', 'ta-ta'),
                                 colour_blob=dict(probability=.35, min_fraction=.4, max_fraction=1.6), size_window=(.5, 1.8)),
    'medium_plane': ClassSpec('medium_plane', signatures=(dark_fraction, green_paint), competitors=('small_plane', 'jet_plane'),
                              colour_blob=dict(probability=.35, min_fraction=.4, max_fraction=1.6)),
    'mine_roller': ClassSpec('mine_roller', signatures=(camo_texture, dark_fraction), competitors=('tank', 'large_launcher')),
    'small_launcher': ClassSpec('small_launcher', signatures=(green_paint, dark_fraction), competitors=('medium_launcher',),
                                colour_blob=dict(probability=.3, min_fraction=.3, max_fraction=2.), size_window=(.4, 2.2), fine_offsets=(-20., -10., 0., 10., 20.)),
    'small_plane': ClassSpec('small_plane', signatures=(red_markings, bright_fraction), competitors=('medium_plane', 'jet_plane'),
                             colour_blob=dict(probability=.3, min_fraction=.05, max_fraction=.6, min_area=3), heading_step=15),
    'small_tower': ClassSpec('small_tower', signatures=(square_roof, green_paint, bright_fraction), competitors=('jammer', 'spacecraft'),
                             colour_blob=dict(probability=.35, min_fraction=.3, max_fraction=1.3)),
    'spacecraft': ClassSpec('spacecraft', signatures=(bright_fraction, dark_fraction), competitors=('small_tower', 'jet_plane'),
                            colour_blob=dict(probability=.3, min_fraction=.3, max_fraction=1.4)),
    'ta-ta': ClassSpec('ta-ta', signatures=(bright_fraction, dark_fraction), competitors=('medium_launcher', 'small_launcher'), size_window=(.5, 2.),
                       colour_blob=dict(probability=.4, min_fraction=.4, max_fraction=1.6, min_area=12), max_colour_candidates=30),
    'tank': ClassSpec('tank', signatures=(camo_texture, barrel_line), competitors=('mine_roller', 'small_tower', 'large_launcher')),
}
