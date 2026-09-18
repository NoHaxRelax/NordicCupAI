"""Part model of an X-shaped object: two bars, dark spots, and the ground between the arms.

Built automatically from the training sprites. Parts are regions, not pixels: each
part is scored by its median lightness in the scene relative to the whole set, so a
global change of lighting cancels, and the pose family includes anisotropic scale and
shear so moderate perspective is tolerated. Dark spots are kept only when they recur
in every training sprite, which excludes one-off shading.
"""
import cv2
import numpy as np

from .common import fit_bars


def _grid(u0, u1, v0, v1, n=5):
    us, vs = np.meshgrid(np.linspace(u0, u1, n), np.linspace(v0, v1, n))
    return np.column_stack([us.ravel(), vs.ravel()])


def _disc(u, v, r, n=21):
    k = int(np.ceil(np.sqrt(n)))
    pts = _grid(u - r, u + r, v - r, v + r, k)
    return pts[np.hypot(pts[:, 0] - u, pts[:, 1] - v) <= r]


class PartModel:
    def __init__(self, class_name, parts, pattern, chroma, fuselage_length, wing_length, wing_angle, half_width, outline):
        self.class_name, self.parts, self.pattern, self.chroma = class_name, parts, np.asarray(pattern, np.float32), float(chroma)
        self.fuselage_length, self.wing_length, self.wing_angle, self.half_width, self.outline = fuselage_length, wing_length, wing_angle, half_width, outline
        self.kinds = [p['kind'] for p in parts]
        self.points = np.concatenate([p['points'] for p in parts]).astype(np.float32)
        self.index = np.concatenate([[i] * len(p['points']) for i, p in enumerate(parts)])

    @classmethod
    def from_templates(cls, class_name, templates, bins=6, spot_drop=40, max_spots=6, spot_tolerance=8.):
        """Geometry from the sharpest sprite; lightness pattern averaged over all sprites."""
        ref = max(templates, key=lambda t: (t.zoom, t.mask.sum()))
        geometry = fit_bars(ref.mask)
        cx, cy = geometry['centroid']
        a1, a2 = geometry['fuselage_angle'], geometry['wing_angle']
        d1 = np.array([np.cos(np.radians(a1)), np.sin(np.radians(a1))]); n1 = np.array([-d1[1], d1[0]])
        to_object = lambda xy: np.column_stack([(xy - (cx, cy)) @ d1, (xy - (cx, cy)) @ n1])
        ys, xs = np.where(ref.mask)
        obj = to_object(np.column_stack([xs, ys]).astype(np.float32))
        wing_angle = (a2 - a1) % 180.
        dw = np.array([np.cos(np.radians(wing_angle)), np.sin(np.radians(wing_angle))]); nw = np.array([-dw[1], dw[0]])
        # bar half-width: median distance from the bar axis of mask pixels near that axis
        near_f = obj[np.abs(obj[:, 1]) <= 14]
        half_width = float(np.clip(np.percentile(np.abs(near_f[:, 1]), 60), 3, 12)) if len(near_f) else 6.
        f_lo, f_hi = geometry['fuselage_extent']
        w_lo, w_hi = geometry['wing_extent']
        parts = []
        edges = np.linspace(f_lo, f_hi, bins + 1)
        for i in range(bins):
            parts.append(dict(kind='fuselage', points=_grid(edges[i], edges[i + 1], -half_width, half_width)))
        edges = np.linspace(w_lo, w_hi, bins + 1)
        for i in range(bins):
            g = _grid(edges[i], edges[i + 1], -half_width, half_width)
            parts.append(dict(kind='wing', points=np.column_stack([g[:, 0] * dw[0] + g[:, 1] * nw[0], g[:, 0] * dw[1] + g[:, 1] * nw[1]])))
        # recurring dark spots, in object coordinates
        spot_sets = []
        for t in templates:
            g = fit_bars(t.mask)
            tc = np.array(g['centroid']); ta = g['fuselage_angle']
            td = np.array([np.cos(np.radians(ta)), np.sin(np.radians(ta))]); tn = np.array([-td[1], td[0]])
            L = cv2.cvtColor(t.bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
            body = np.median(L[t.mask])
            dark = (t.mask & (L < body - spot_drop)).astype(np.uint8)
            count, _, stats, centroids = cv2.connectedComponentsWithStats(dark)
            spots = []
            for i in range(1, count):
                if stats[i, 4] >= 6:
                    rel = centroids[i] - tc
                    spots.append((float(rel @ td), float(rel @ tn), float(np.sqrt(stats[i, 4] / np.pi))))
            # the bar fit has a 180-degree ambiguity per sprite: align to the reference by the better of the two flips
            spot_sets.append(spots)
        ref_spots = spot_sets[templates.index(ref)]
        aligned = []
        for spots in spot_sets:
            best = None
            for flip in (1., -1.):
                cand = [(u * flip, v * flip, r) for u, v, r in spots]
                agreement = sum(any(np.hypot(u - ru, v - rv) <= spot_tolerance for ru, rv, _ in ref_spots) for u, v, _ in cand)
                if best is None or agreement > best[0]:
                    best = (agreement, cand)
            aligned.append(best[1])
        recurring = []
        for u, v, r in sorted(ref_spots, key=lambda s: -s[2]):
            if all(any(np.hypot(u - su, v - sv) <= spot_tolerance for su, sv, _ in spots) for spots in aligned):
                recurring.append((u, v, max(2.5, r)))
        for u, v, r in recurring[:max_spots]:
            parts.append(dict(kind='spot', points=_disc(u, v, r)))
        # ground between the arms: discs on the bisectors of the four angular gaps, outside the mask
        radius = .5 * min(f_hi - f_lo, w_hi - w_lo)
        gap_dirs = [0., wing_angle, 180., wing_angle + 180.]
        gap_dirs = sorted(a % 360 for a in gap_dirs)
        for i in range(4):
            a, b = gap_dirs[i], gap_dirs[(i + 1) % 4] + (360 if i == 3 else 0)
            mid = np.radians((a + b) / 2)
            u, v = .55 * radius * np.cos(mid), .55 * radius * np.sin(mid)
            pts = _disc(u, v, .18 * radius)
            sprite_xy = np.column_stack([cx + pts[:, 0] * d1[0] + pts[:, 1] * n1[0], cy + pts[:, 0] * d1[1] + pts[:, 1] * n1[1]]).round().astype(int)
            inside = (sprite_xy[:, 0] >= 0) & (sprite_xy[:, 0] < ref.mask.shape[1]) & (sprite_xy[:, 1] >= 0) & (sprite_xy[:, 1] < ref.mask.shape[0])
            keep = np.ones(len(pts), bool)
            keep[inside] = ~ref.mask[sprite_xy[inside, 1], sprite_xy[inside, 0]]
            parts.append(dict(kind='quadrant', points=pts[keep] if keep.sum() >= 5 else pts))
        model = cls(class_name, parts, np.zeros(len(parts)), 0., f_hi - f_lo, w_hi - w_lo, wing_angle, half_width,
                    outline=cls._outline(ref.mask, cx, cy, d1, n1))
        # lightness pattern and chroma, averaged over every training sprite (each aligned to the reference)
        vectors, chromas = [], []
        for t, spots in zip(templates, aligned):
            g = fit_bars(t.mask)
            flip = 1.
            if spots and ref_spots:
                agree = lambda f: sum(any(np.hypot(u * f - ru, v * f - rv) <= spot_tolerance for ru, rv, _ in ref_spots) for u, v, _ in spots)
                flip = 1. if agree(1.) >= agree(-1.) else -1.
            pad = 12
            padded = cv2.copyMakeBorder(t.bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            lab = cv2.cvtColor(padded, cv2.COLOR_BGR2LAB).astype(np.float32)
            L = lab[:, :, 0]; chroma = np.hypot(lab[:, :, 1] - 128, lab[:, :, 2] - 128)
            heading = g['fuselage_angle'] if flip > 0 else g['fuselage_angle'] + 180.
            pts = model.pose_points(heading, 1., 1., 1., 0., g['centroid'][0] + pad, g['centroid'][1] + pad)
            vector, visible = model.sample(L, pts)
            structural = np.isin(np.array(model.kinds), ['fuselage', 'wing', 'spot'])
            if visible[structural].mean() >= .8:
                vectors.append(vector)
                cv, _ = model.sample(chroma, pts)
                chromas.append(np.median([cv[i] for i, k in enumerate(model.kinds) if k in ('fuselage', 'wing')]))
        if not vectors:
            raise ValueError(f'Could not sample the {class_name} part model on its own sprites')
        model.pattern = np.mean(vectors, axis=0).astype(np.float32)
        model.chroma = float(np.median(chromas))
        return model

    @staticmethod
    def _outline(mask, cx, cy, d1, n1):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(max(contours, key=cv2.contourArea)).reshape(-1, 2).astype(np.float32)
        return np.column_stack([(hull - (cx, cy)) @ d1, (hull - (cx, cy)) @ n1])

    def _matrix(self, heading, scale, sx, sy, shear):
        d = np.array([np.cos(np.radians(heading)), np.sin(np.radians(heading))]); n = np.array([-d[1], d[0]])
        affine = np.array([[sx, np.tan(np.radians(shear)) * sy], [0., sy]]) * scale
        return np.column_stack([d, n]) @ affine  # maps object (u, v) to image offsets

    def pose_points(self, heading, scale, sx, sy, shear, cx, cy):
        return self.points @ self._matrix(heading, scale, sx, sy, shear).T + (cx, cy)

    def pose_outline(self, heading, scale, sx, sy, shear, cx, cy):
        return self.outline @ self._matrix(heading, scale, sx, sy, shear).T + (cx, cy)

    def sample(self, channel, pts):
        """Median of each part over its points inside the image; visible when at least half its points are inside."""
        H, W = channel.shape
        xi, yi = pts[:, 0].round().astype(int), pts[:, 1].round().astype(int)
        inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        values = np.full(len(pts), np.nan, np.float32)
        values[inside] = channel[yi[inside], xi[inside]]
        vector = np.full(len(self.parts), np.nan, np.float32)
        visible = np.zeros(len(self.parts), bool)
        for i in range(len(self.parts)):
            sel = self.index == i
            v = values[sel]
            ok = ~np.isnan(v)
            if ok.sum() >= max(3, .5 * sel.sum()):
                vector[i] = np.median(v[ok]); visible[i] = True
        return vector, visible

    def score(self, L, chroma, pts, min_visible=.6):
        vector, visible = self.sample(L, pts)
        kinds = np.array(self.kinds)
        bars = visible & np.isin(kinds, ['fuselage', 'wing'])
        if visible.mean() < min_visible or bars.sum() < 6:
            return None
        # The reference pattern covers bars and spots only: the sprite holds no real ground between the arms.
        structural = visible & np.isin(kinds, ['fuselage', 'wing', 'spot'])
        a, b = self.pattern[structural], vector[structural]
        a, b = a - a.mean(), b - b.mean()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        pattern_ncc = float(a @ b / denominator) if denominator > 1e-6 else 0.
        quads = visible & (kinds == 'quadrant')
        bar_values = vector[bars]
        xness = float(abs(bar_values.mean() - vector[quads].mean()) / (bar_values.std() + 3.)) if quads.any() else 0.
        spots = visible & (kinds == 'spot')
        spot_drop = float(np.median(bar_values) - np.median(vector[spots])) if spots.any() else 0.
        model_spot_drop = float(np.median(self.pattern[np.isin(kinds, ['fuselage', 'wing'])]) - np.median(self.pattern[kinds == 'spot'])) if (kinds == 'spot').any() else 1.
        spot_score = float(np.clip(spot_drop / max(model_spot_drop, 1.), 0, 1))
        cvec, _ = self.sample(chroma, pts)
        bar_chroma = float(np.nanmedian(cvec[bars]))
        return dict(pattern_ncc=pattern_ncc, xness=xness, spot_score=spot_score, bar_chroma=bar_chroma, visible_fraction=float(visible.mean()),
                    parts_visible=int(visible.sum()))
