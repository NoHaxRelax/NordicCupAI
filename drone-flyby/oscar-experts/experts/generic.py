"""Generic per-object expert: candidate generator built only from the class's training sprites.

Every class gets the same skeleton; a class spec adds its distinctive measurements.
  proposers   -> heading-sweep masked correlation at half resolution (all classes), plus a
                 colour-blob proposer for classes whose paint is rare in the scene
  pose        -> fine masked correlation (gray + high-pass) at the proposed heading, visible pixels only
  colour      -> foreground Lab agreement with the sprite (distance and correlation)
  signatures  -> class-specific measurements from the spec (recorded, never a gate)
  competitors -> best masked correlation of other classes' sprites of similar size at the same spot
  size        -> the only physical gate: altitude fixes the object's pixel size per zoom
Candidates are never ruled out on appearance. Boxes are the posed mask extent plus the organiser offset.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .common import CorrelationProposer, SiftMatcher, features, fit_bars, load_templates, local_masked_match, masked_ncc

ZOOM_FOR_SCALE = {.25: 0, .5: 1, 1.: 2}


@dataclass
class ClassSpec:
    class_name: str
    heading_step: int = 15
    proposer_threshold: float = .2
    proposer_downscale: float = 1.     # full resolution: half resolution lost 40-50 px objects among ground peaks
    proposer_single_template: bool = True  # sweep one template per zoom; every template still competes in the fine pose
    proposer_blur: float = .8
    max_candidates: int = 24
    max_colour_candidates: int = 16
    size_window: tuple = (.6, 1.6)         # accepted long side relative to the sprite, native px
    fine_offsets: tuple = (-8., 0., 8.)
    scales: tuple = (1.,)                  # altitude fixes the object's size; the gate sees any residual mismatch
    min_visible: float = .35
    colour_blob: dict | None = None        # dict(chroma_threshold, min_area_fraction) to enable the colour proposer
    signatures: tuple = ()                 # callables (ctx, candidate) -> dict of scores
    competitors: tuple = ()                # class names checked at each candidate
    sift: bool = False
    dedup_fraction: float = .33


class SpriteStats:
    """Foreground colour statistics of one class, per zoom, from its training sprites only."""
    def __init__(self, templates):
        self.by_zoom = {}
        for zoom in {t.zoom for t in templates}:
            labs = []
            for t in templates:
                if t.zoom != zoom:
                    continue
                lab = cv2.cvtColor(t.bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
                labs.append(lab[t.mask])
            pixels = np.concatenate(labs)
            self.by_zoom[zoom] = dict(mean=pixels.mean(0), std=np.maximum(pixels.std(0), (6., 3., 3.)),
                                      chroma=float(np.median(np.hypot(pixels[:, 1] - 128, pixels[:, 2] - 128))))

    def get(self, zoom):
        return self.by_zoom.get(zoom) or next(iter(self.by_zoom.values()))


class GenericExpert:
    def __init__(self, bank, spec, all_templates=None, gate=None):
        self.spec = spec
        self.settings = spec  # evaluators read .settings on every expert
        self.gate = gate  # optional fitted logistic gate (fit_gates.py): adds gate_logit/gate_pass, never drops in explain mode
        self.class_name = spec.class_name
        self.templates = load_templates(bank, spec.class_name)
        self.stats = SpriteStats(self.templates)
        self.proposer = CorrelationProposer(self.templates, headings=range(0, 360, spec.heading_step), threshold=spec.proposer_threshold, downscale=spec.proposer_downscale, blur=spec.proposer_blur)
        self.sift = SiftMatcher(self.templates) if spec.sift else None
        self.long_side = float(np.median([max(t.mask.shape) for t in self.templates if t.zoom == max(x.zoom for x in self.templates)]))
        self.organizer_offset = np.median([t.organizer_offset for t in self.templates], axis=0)
        self.angles = {t.id: (fit_bars(t.mask)['fuselage_angle'] if t.mask.sum() > 64 else 0.) for t in self.templates}
        self.competitors = {}
        for name in spec.competitors:
            try:
                self.competitors[name] = [t for t in (all_templates or {}).get(name) or load_templates(bank, name)]
            except ValueError:
                continue
        self._posed = {}

    def templates_for(self, zoom):
        matching = [t for t in self.templates if t.zoom == zoom]
        return matching or self.templates

    def proposer_templates_for(self, zoom):
        """One template per zoom for the heading sweep (the largest mask); the fine pose still tries all."""
        matching = self.templates_for(zoom)
        return [max(matching, key=lambda t: t.mask.sum())] if self.spec.proposer_single_template else matching

    def posed(self, template, angle, scale):
        key = (template.id, round(angle, 1), round(scale, 4))
        if key not in self._posed:
            bgr, mask, box = template.posed_with_box(angle, scale)
            gray, high = features(bgr)
            self._posed[key] = (bgr, gray, high, mask, box)
            if len(self._posed) > 800:
                self._posed.pop(next(iter(self._posed)))
        return self._posed[key]

    def colour_proposals(self, lab, s, zoom):
        """Connected components of pixels near the sprite's paint colour, within the size window."""
        cfg = self.spec.colour_blob
        st = self.stats.get(zoom)
        z = (lab - st['mean']) / st['std']
        probability = np.exp(-.5 * (z[:, :, 1] ** 2 + z[:, :, 2] ** 2))
        mask = (probability > cfg.get('probability', .35)).astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        rows = []
        expected = self.long_side * s
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            long_side = max(w, h)
            if not cfg.get('min_fraction', .3) * expected <= long_side <= cfg.get('max_fraction', 1.6) * expected:
                continue
            if area < cfg.get('min_area', 6) * s * s:
                continue
            component = labels == label
            pts = np.column_stack(np.where(component))[:, ::-1].astype(np.float32)
            (_, _), (rw, rh), rangle = cv2.minAreaRect(pts)
            heading = (rangle if rw >= rh else rangle + 90.) % 180.
            rows.append(dict(cx=float(centroids[label][0]), cy=float(centroids[label][1]), heading=float(heading), proposer_score=float(probability[component].mean()),
                             template_id='colour', size=(float(w), float(h)), source='colour', component_area=int(area)))
        return rows

    @staticmethod
    def _pad(gray, high, margin):
        return (cv2.copyMakeBorder(gray, margin, margin, margin, margin, cv2.BORDER_REFLECT),
                cv2.copyMakeBorder(high, margin, margin, margin, margin, cv2.BORDER_REFLECT), margin)

    def _match_window(self, padded, tg, th, tmask, cx, cy, radius):
        """Best masked correlation of one posed template within +-radius of (cx, cy), via matchTemplate.

        The scene is reflection-padded so poses partly outside the image still score; the
        template mask keeps the comparison on the object's own pixels.
        """
        gray_p, high_p, margin = padded
        h, w = tmask.shape
        x0 = int(round(cx - w / 2 - radius)) + margin
        y0 = int(round(cy - h / 2 - radius)) + margin
        x1, y1 = x0 + w + 2 * radius, y0 + h + 2 * radius
        if x0 < 0 or y0 < 0 or x1 > gray_p.shape[1] or y1 > gray_p.shape[0]:
            return None
        mask = tmask.astype(np.uint8)
        if mask.sum() < 8:
            return None
        g = cv2.matchTemplate(gray_p[y0:y1, x0:x1], tg, cv2.TM_CCOEFF_NORMED, mask=mask)
        hh = cv2.matchTemplate(high_p[y0:y1, x0:x1], th, cv2.TM_CCOEFF_NORMED, mask=mask)
        response = np.nan_to_num(.45 * g + .55 * hh, nan=-1., posinf=-1., neginf=-1.)
        _, peak, _, (px, py) = cv2.minMaxLoc(response)
        return float(peak), x0 + px - margin, y0 + py - margin

    def fine_pose(self, padded, image_shape, template, cx, cy, s, headings, partial):
        H, W = image_shape
        best = None
        for base in headings:
            for offset in self.spec.fine_offsets:
                for scale in self.spec.scales:
                    angle = base + offset
                    _, tg, th, tmask, _ = self.posed(template, angle, s * scale)
                    thh, tww = tmask.shape
                    radius = max(4, min(tww, thh) // 3) if partial else 6
                    hit = self._match_window(padded, tg, th, tmask, cx, cy, radius)
                    if hit is None:
                        continue
                    peak, x1, y1 = hit
                    vis_x = max(0, min(W, x1 + tww) - max(0, x1)); vis_y = max(0, min(H, y1 + thh) - max(0, y1))
                    visible = (vis_x * vis_y) / float(tww * thh)
                    if visible < self.spec.min_visible:
                        continue
                    if best is None or peak > best[0]:
                        best = (peak, x1, y1, visible, template, angle, scale, tmask)
        return best

    def detect(self, image, pixels_per_source_pixel=1., zoom=None, explain=False, proposals=None):
        """proposals: optional precomputed correlation proposals (SharedProposer), skipping this expert's own sweep."""
        if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('Expected uint8 BGR image')
        s = float(pixels_per_source_pixel)
        if not np.isfinite(s) or s <= 0:
            raise ValueError('Invalid image scale')
        zoom = ZOOM_FOR_SCALE.get(s, 2) if zoom is None else int(zoom)
        spec = self.spec
        H, W = image.shape[:2]
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, chroma = lab[:, :, 0], np.hypot(lab[:, :, 1] - 128, lab[:, :, 2] - 128)
        gray, high = features(image)
        templates = self.templates_for(zoom)
        padded = self._pad(gray, high, int(self.long_side * s * 1.5) + 16)
        # Correlation peaks and colour blobs have different score scales: cap each source on its own so
        # colour proposals are never crowded out by a wall of weak correlation peaks.
        radius = spec.dedup_fraction * self.long_side * s
        raw = proposals if proposals is not None else [dict(p, source='correlation') for p in self.proposer.propose(image, s, self.proposer_templates_for(zoom))]
        proposals = self.proposer.merge(raw, radius=radius)[:spec.max_candidates]
        if spec.colour_blob:
            colour = self.proposer.merge(sorted(self.colour_proposals(lab, s, zoom), key=lambda p: -p['proposer_score']), radius=radius)[:spec.max_colour_candidates]
            proposals = self.proposer.merge(proposals + colour, radius=radius)
        ctx = dict(image=image, lab=lab, L=L, chroma=chroma, gray=gray, high=high, scale=s, zoom=zoom, expert=self)
        rows, candidates = [], []
        for prop in proposals:
            cx, cy = prop['cx'], prop['cy']
            half = .5 * self.long_side * s
            partial = cx < half or cy < half or cx > W - half or cy > H - half
            candidate = dict(cx=cx, cy=cy, proposer_score=prop['proposer_score'], proposer_source=prop['source'], proposer_heading=prop['heading'], partial=bool(partial))
            headings = sorted({prop['heading'] % 360., (prop['heading'] + 180.) % 360.} | {h % 360. for h in prop.get('alternative_headings', [])[:1]})
            best = None
            for template in templates:
                hit = self.fine_pose(padded, (H, W), template, cx, cy, s, [h - self.angles[template.id] for h in headings] + [h - self.angles[template.id] + 180. for h in headings], partial)
                if hit and (best is None or hit[0] > best[0]):
                    best = hit
            if best is None:
                candidate['rejected_by'] = 'not_visible'; candidates.append(candidate); continue
            correlation, x1, y1, visible, template, angle, scale, tmask = best
            th_, tw_ = tmask.shape
            fitted_long = max(th_, tw_) / s / scale
            # size gate: the only physical exclusion
            ratio = max(th_, tw_) / (self.long_side * s)
            if not spec.size_window[0] <= ratio <= spec.size_window[1]:
                candidate['rejected_by'] = 'size'; candidates.append(candidate); continue
            # colour agreement on the visible mask pixels
            sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(W, x1 + tw_), min(H, y1 + th_)
            m = tmask[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1]
            region = lab[sy1:sy2, sx1:sx2][m]
            st = self.stats.get(zoom)
            colour_distance = float(np.sqrt((((region.mean(0) - st['mean']) / st['std']) ** 2).sum())) if len(region) else 99.
            bgr_posed, _, _, _, posed_box = self.posed(template, angle, s * scale)
            bgr_posed = bgr_posed[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1]
            colour_ncc = float(np.mean([masked_ncc(image[sy1:sy2, sx1:sx2][:, :, c], bgr_posed[:, :, c], m) for c in range(3)])) if m.sum() >= 8 else 0.
            region_chroma = float(np.median(chroma[sy1:sy2, sx1:sx2][m])) if len(region) else 0.
            candidate.update(correlation=float(correlation), visible_fraction=float(visible), template_id=template.id, angle=float(angle), fitted_scale=float(scale),
                             colour_distance=colour_distance, colour_ncc=colour_ncc, region_chroma=region_chroma, sprite_chroma=st['chroma'],
                             mask_bbox=[float(x1), float(y1), float(x1 + tw_), float(y1 + th_)])
            for signature in spec.signatures:
                try:
                    candidate.update(signature(ctx, candidate, template, tmask))
                except Exception as error:  # a broken signature must never drop a candidate
                    candidate[f'{signature.__name__}_error'] = str(error)[:120]
            # competitors of similar size at the same spot
            comp = {}
            for name, others in self.competitors.items():
                others_z = [t for t in others if t.zoom == zoom] or others
                best_other = None
                for t in others_z[:2]:
                    for base in range(0, 360, 45):
                        _, tg, th, om, _ = self.posed(t, base, s)
                        hit = self._match_window(padded, tg, th, om, cx, cy, 4)
                        if hit and (best_other is None or hit[0] > best_other):
                            best_other = hit[0]
                comp[name] = float(best_other) if best_other is not None else 0.
            candidate['competitor_scores'] = comp
            candidate['competitor_margin'] = float(correlation - max(comp.values())) if comp else 1.
            dx1, dy1, dx2, dy2 = posed_box  # organiser box turned with the pose, already at the posed scale
            bbox = [x1 + dx1, y1 + dy1, x1 + tw_ + dx2, y1 + th_ + dy2]
            candidate['bbox'] = [float(np.clip(bbox[0], 0, W)), float(np.clip(bbox[1], 0, H)), float(np.clip(bbox[2], 0, W)), float(np.clip(bbox[3], 0, H))]
            score = float(np.clip(.6 * correlation + .2 * max(0., colour_ncc) + .2 * np.exp(-colour_distance / 3.), 0, 1))
            candidate.update({'class': self.class_name, 'family': 'expert', 'score': score, 'rejected_by': None})
            if self.gate is not None:
                from .fit_gates import apply_gate
                logit, passed = apply_gate(self.gate, candidate, zoom)
                candidate.update(gate_logit=float(logit), gate_probability=float(1 / (1 + np.exp(-logit))), gate_pass=bool(passed))
                candidate['score'] = candidate['gate_probability']
                if not passed:
                    candidate['rejected_by'] = 'gate'
                    candidates.append(candidate)
                    continue
            candidates.append(candidate)
            rows.append(candidate)
        rows = self._dedup(rows, s)
        families = {'expert': rows, 'sift': [dict(r, **{'class': self.class_name}) for r in self.sift.match(image, s)] if self.sift else []}
        return (families, candidates) if explain else families

    def _dedup(self, rows, s):
        kept = []
        for row in sorted(rows, key=lambda r: -r['score']):
            limit = self.spec.dedup_fraction * self.long_side * s
            if not any(np.hypot(row['cx'] - k['cx'], row['cy'] - k['cy']) < limit for k in kept):
                kept.append(row)
        return kept
