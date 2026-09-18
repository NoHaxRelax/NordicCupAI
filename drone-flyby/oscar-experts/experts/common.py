"""Shared pieces for per-object experts: bank loading, flat-filled templates,
masked correlation with partial visibility, and a SIFT comparison matcher.

Templates come from the training-only expert bank. A sprite is composited
over its own mean foreground colour, so no source background survives in
any feature computed from it.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def features(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
    return gray, gray - cv2.GaussianBlur(gray, (0, 0), 2.)


def flat_fill(rgba, alpha_threshold=128):
    """Composite the sprite over its mean foreground colour; return (bgr, hard mask)."""
    alpha = rgba[:, :, 3].astype(np.float32) / 255.
    hard = rgba[:, :, 3] >= alpha_threshold
    fill = rgba[:, :, :3][hard].reshape(-1, 3).mean(0)
    bgr = rgba[:, :, :3].astype(np.float32) * alpha[:, :, None] + fill * (1 - alpha[:, :, None])
    return bgr.round().clip(0, 255).astype(np.uint8), hard


def long_axis_angle(mask):
    """Orientation of the mask's minimum-area rectangle long side, degrees in [0, 180)."""
    pts = np.column_stack(np.where(mask))[:, ::-1].astype(np.float32)
    (_, _), (w, h), angle = cv2.minAreaRect(pts)
    theta = angle if w >= h else angle + 90.
    return theta % 180., (max(w, h), min(w, h))


@dataclass
class Template:
    id: str
    zoom: int
    bgr: np.ndarray          # flat-filled, cropped to mask extent
    mask: np.ndarray         # bool
    organizer_offset: tuple  # (dx1, dy1, dx2, dy2) from mask extent to organiser box, native px
    angle: float             # long-axis orientation of the mask
    long_side: float
    short_side: float

    def posed(self, angle, scale):
        """Rotate by `angle` degrees and resize by `scale`; returns (bgr, mask) cropped to the mask."""
        return self.posed_with_box(angle, scale)[:2]

    def posed_with_box(self, angle, scale):
        """Like posed, plus the organiser box transformed with the same pose, as offsets
        (dx1, dy1, dx2, dy2) from the cropped mask extent. Asymmetric boxes (a hangar's wall,
        a mine roller's roller) must turn with the object."""
        bgr, mask = self.bgr, self.mask.astype(np.uint8) * 255
        if scale != 1.:
            w, h = max(4, round(bgr.shape[1] * scale)), max(4, round(bgr.shape[0] * scale))
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            bgr, mask = cv2.resize(bgr, (w, h), interpolation=interp), cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        h, w = mask.shape
        matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.)
        corners = cv2.transform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), matrix)[0]
        low, high = np.floor(corners.min(0)), np.ceil(corners.max(0))
        matrix[:, 2] -= low
        size = tuple((high - low).astype(int))
        bgr = cv2.warpAffine(bgr, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.warpAffine(mask, matrix, size, flags=cv2.INTER_NEAREST) > 127
        ys, xs = np.where(mask)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        ox1, oy1, ox2, oy2 = self.organizer_offset  # organiser box relative to the unposed mask extent (0, 0, w, h)
        box = np.float32([[[ox1, oy1], [w + ox2, oy1], [w + ox2, h + oy2], [ox1, h + oy2]]]) * (scale if scale != 1. else 1.)
        corners = cv2.transform(box, matrix)[0]
        low, high = corners.min(0), corners.max(0)
        offset = (float(low[0] - x1), float(low[1] - y1), float(high[0] - x2), float(high[1] - y2))
        return bgr[y1:y2, x1:x2], mask[y1:y2, x1:x2], offset


def load_templates(bank, class_name):
    bank = Path(bank)
    doc = json.loads((bank / 'manifest.json').read_text())
    if doc.get('format') != 'expert-sprite-bank-v1':
        raise ValueError('Unsupported expert bank')
    out = []
    for row in doc['sprites']:
        if row['class_name'] != class_name:
            continue
        path = bank / row['file']
        if sha(path) != row['sha256']:
            raise ValueError('Sprite checksum mismatch: ' + row['file'])
        rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        bgr, hard = flat_fill(rgba, row['alpha_threshold'])
        ys, xs = np.where(hard)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        ox1, oy1, ox2, oy2 = row['organizer_box_in_sprite']
        angle, (long_side, short_side) = long_axis_angle(hard)
        out.append(Template(row['id'], row['zoom'], bgr[y1:y2, x1:x2], hard[y1:y2, x1:x2],
                            (ox1 - x1, oy1 - y1, ox2 - x2, oy2 - y2), angle, long_side, short_side))
    if not out:
        raise ValueError(f'No {class_name} sprites in bank')
    return out


def masked_ncc(a, b, mask):
    """Normalised correlation of a and b over mask pixels."""
    if mask.sum() < 8:
        return 0.
    x, y = a[mask].astype(np.float32), b[mask].astype(np.float32)
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator > 1e-8 else 0.


def local_masked_match(scene_gray, scene_high, tgray, thigh, tmask, cx, cy, radius=8, step=2, min_visible=.35, weights=(.45, .55), offsets=None):
    """Best masked correlation of a posed template centred near (cx, cy).

    Only template pixels that fall inside the image are scored, so an object
    cut by the crop edge is compared on its visible part. Returns
    (score, x1, y1, visible_fraction) with the template's top-left corner or None.
    """
    H, W = scene_gray.shape
    th, tw = tmask.shape
    total = float(tmask.sum())
    best = None
    if offsets is None:
        offsets = [(dx, dy) for dy in range(-radius, radius + 1, step) for dx in range(-radius, radius + 1, step)]
    for dx, dy in offsets:
        if True:
            x1, y1 = int(round(cx - tw / 2 + dx)), int(round(cy - th / 2 + dy))
            sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(W, x1 + tw), min(H, y1 + th)
            if sx2 <= sx1 or sy2 <= sy1:
                continue
            m = tmask[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1]
            visible = m.sum() / total
            if visible < min_visible:
                continue
            g = masked_ncc(scene_gray[sy1:sy2, sx1:sx2], tgray[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1], m)
            h = masked_ncc(scene_high[sy1:sy2, sx1:sx2], thigh[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1], m)
            score = weights[0] * g + weights[1] * h
            if best is None or score > best[0]:
                best = (score, x1, y1, float(visible))
    return best


class SiftMatcher:
    """Comparison branch: SIFT on flat-filled sprites, geometric verification in the scene."""
    def __init__(self, templates, upsample=2., ratio=.8, min_inliers=3, reprojection=2.5, iterations=3000):
        self.upsample, self.ratio, self.min_inliers, self.reprojection, self.iterations = upsample, ratio, min_inliers, reprojection, iterations
        self.sift = cv2.SIFT_create(nfeatures=18000, contrastThreshold=.012, edgeThreshold=15, sigma=1.2)
        self.bank = []
        for t in templates:
            gray = cv2.cvtColor(t.bgr, cv2.COLOR_BGR2GRAY)
            enlarged = cv2.resize(gray, None, fx=upsample, fy=upsample, interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(t.mask.astype(np.uint8) * 255, enlarged.shape[1::-1], interpolation=cv2.INTER_NEAREST)
            keypoints, descriptors = self.sift.detectAndCompute(enlarged, mask)
            if descriptors is None or len(keypoints) < min_inliers:
                continue
            points = (np.float32([k.pt for k in keypoints]) + .5) / upsample
            self.bank.append((t, points, descriptors))
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def match(self, bgr, scale=1.):
        if not self.bank:
            return []
        gray = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), None, fx=self.upsample, fy=self.upsample, interpolation=cv2.INTER_CUBIC)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 2:
            return []
        scene = (np.float32([k.pt for k in keypoints]) + .5) / self.upsample
        rows = []
        for template, points, tdesc in self.bank:
            pairs = self.matcher.knnMatch(descriptors, tdesc, k=2)
            good = [a for a, b in pairs if a.distance < self.ratio * b.distance and a.distance < 330.]
            if len(good) < self.min_inliers:
                continue
            src = np.float32([points[m.trainIdx] for m in good])
            dst = scene[[m.queryIdx for m in good]]
            cv2.setRNGSeed(0)
            transform, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.reprojection,
                                                             maxIters=self.iterations, confidence=.995, refineIters=10)
            if transform is None:
                continue
            chosen = [i for i, ok in zip(range(len(good)), inliers.ravel()) if ok]
            if len({good[i].trainIdx for i in chosen}) < self.min_inliers:
                continue
            fitted_scale = float(np.linalg.norm(transform[:, 0]))
            if not .35 <= fitted_scale / scale <= 2.8:
                continue
            h, w = template.mask.shape
            corners = cv2.transform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), transform)[0]
            low, high = corners.min(0), corners.max(0)
            error = float(np.median(np.linalg.norm(cv2.transform(src[chosen][None], transform)[0] - dst[chosen], axis=1)))
            distance = float(np.median([good[i].distance for i in chosen]))
            score = float(np.clip(.5 + .045 * min(len(chosen), 8) + .12 * (1 - distance / 330.) - .02 * error, 0, 1))
            rows.append(dict(family='sift', bbox=[float(low[0]), float(low[1]), float(high[0]), float(high[1])], score=score,
                             inliers=len(chosen), reprojection_error=error, template_id=template.id,
                             angle=float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0]))), scale=fitted_scale / scale))
        return sorted(rows, key=lambda r: -r['score'])


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


def nms(rows, threshold=.35):
    kept = []
    for row in sorted(rows, key=lambda r: -r['score']):
        if not any(iou(row['bbox'], k['bbox']) > threshold for k in kept):
            kept.append(row)
    return kept


def fit_bars(mask, half=6, min_separation=50):
    """Two straight bars through the mask centroid: the fuselage (longer) and the crossing bar.

    Returns dict(centroid, fuselage_angle, fuselage_extent, wing_angle, wing_extent) with
    angles in image degrees and extents (lo, hi) along each bar from the centroid.
    """
    ys, xs = np.where(mask)
    cx, cy = xs.mean(), ys.mean()
    p = np.column_stack([xs - cx, ys - cy]).astype(np.float32)

    def cover(a):
        d = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))], np.float32)
        n = np.array([-d[1], d[0]], np.float32)
        on = np.abs(p @ n) <= half
        along = p @ d
        return int(on.sum()), (float(along[on].min()), float(along[on].max())) if on.any() else (0., 0.)

    scored = sorted(((cover(a)[0], a) for a in range(0, 180)), reverse=True)
    a1 = scored[0][1]
    others = [(c, a) for c, a in scored if min_separation < abs(a - a1) % 180 < 180 - min_separation]
    if not others:
        raise ValueError('Mask has no crossing bar')
    a2 = max(others)[1]
    e1, e2 = cover(a1)[1], cover(a2)[1]
    if e2[1] - e2[0] > e1[1] - e1[0]:
        a1, a2, e1, e2 = a2, a1, e2, e1
    return dict(centroid=(float(cx), float(cy)), fuselage_angle=float(a1), fuselage_extent=e1, wing_angle=float(a2), wing_extent=e2)


class CorrelationProposer:
    """Heading-sweep masked correlation at reduced resolution: a coarse, structure-level proposer.

    Templates are blurred and downscaled so only the broad shape is matched, which keeps
    the proposer from rewarding exact pixels. The scene is edge-replicated by half a
    template so objects cut by the crop still produce a peak.
    """
    def __init__(self, templates, headings=range(0, 360, 15), downscale=.5, blur=1., weights=(.4, .6), peaks_per_pose=3, threshold=.2):
        self.templates, self.headings, self.downscale, self.blur, self.weights = list(templates), list(headings), downscale, blur, weights
        self.peaks_per_pose, self.threshold = peaks_per_pose, threshold
        self._posed = {}

    def posed(self, template, heading, scale):
        key = (template.id, heading, round(scale, 4))
        if key not in self._posed:
            bgr, mask = template.posed(heading, scale * self.downscale)
            bgr = cv2.GaussianBlur(bgr, (0, 0), self.blur)
            gray, high = features(bgr)
            angle = fit_bars(mask)['fuselage_angle'] if mask.sum() > 64 else float(heading)
            self._posed[key] = (gray, high, mask.astype(np.uint8), angle)
        return self._posed[key]

    def propose(self, image, scale=1., templates=None):
        templates = self.templates if templates is None else templates
        small = cv2.resize(image, None, fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), self.blur)
        gray, high = features(small)
        rows = []
        for template in templates:
            for heading in self.headings:
                tg, th, tmask, angle = self.posed(template, heading, scale)
                h, w = tmask.shape
                py, px = h // 2, w // 2
                g = cv2.copyMakeBorder(gray, py, py, px, px, cv2.BORDER_REPLICATE)
                hi = cv2.copyMakeBorder(high, py, py, px, px, cv2.BORDER_REPLICATE)
                if h > g.shape[0] or w > g.shape[1]:
                    continue
                response = self.weights[0] * cv2.matchTemplate(g, tg, cv2.TM_CCOEFF_NORMED, mask=tmask) \
                    + self.weights[1] * cv2.matchTemplate(hi, th, cv2.TM_CCOEFF_NORMED, mask=tmask)
                response = np.nan_to_num(response, nan=-1., posinf=-1., neginf=-1.)
                for _ in range(self.peaks_per_pose):
                    _, peak, _, (x, y) = cv2.minMaxLoc(response)
                    if peak < self.threshold:
                        break
                    cx, cy = (x - px + w / 2) / self.downscale, (y - py + h / 2) / self.downscale
                    rows.append(dict(cx=float(cx), cy=float(cy), heading=float(angle), proposer_score=float(peak), template_id=template.id,
                                     size=(w / self.downscale, h / self.downscale)))
                    response[max(0, y - h // 3):y + h // 3 + 1, max(0, x - w // 3):x + w // 3 + 1] = -1.
        return sorted(rows, key=lambda r: -r['proposer_score'])

    @staticmethod
    def merge(rows, radius):
        """Keep the best proposal per location, remembering the other headings seen there."""
        kept = []
        for r in rows:
            for k in kept:
                if np.hypot(r['cx'] - k['cx'], r['cy'] - k['cy']) <= radius:
                    k['alternative_headings'].append(r['heading'])
                    break
            else:
                kept.append(dict(r, alternative_headings=[]))
        return kept
