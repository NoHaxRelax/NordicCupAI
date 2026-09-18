"""Hangar expert: a large, flat, near-black roof with a light rim and a rounded end.

Candidates are never ruled out on appearance: only size and visibility are gates. Every
other measure is recorded on the candidate for the verifier that comes after.

Every component reports its own score so the branches can be compared:
  size gate        -> long side of the dark component, native px
  proposer         -> dark connected components, fill ratio, aspect, heading from the rectangle
  correlation      -> masked gray/high-pass correlation against the posed sprite (visible pixels only)
  colour           -> near-zero chroma inside the roof
  rim contrast     -> ring lightness minus interior lightness (tree shade fails this, concrete apron passes)
  sift             -> geometric comparison branch, reported separately, never fused
The output box is the posed mask extent plus the organiser offset stored in the bank.
Inputs are the image and its scale; no frame index, track or label is consumed.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .common import features, load_templates, local_masked_match, nms, SiftMatcher

ZOOM_FOR_SCALE = {.25: 0, .5: 1, 1.: 2}


@dataclass(frozen=True)
class HangarSettings:
    dark_lightness: int = 35          # OpenCV Lab L (0-255) threshold for roof pixels; sprite interior p90 is 10, edge band 36
    min_long_side: float = 150.       # native px, gate on the dark component
    max_long_side: float = 260.
    min_aspect: float = 1.3
    max_aspect: float = 3.5
    min_fill: float = .3              # component area over its minimum-area rectangle
    min_partial_area: float = .2      # fraction of the template area for edge-touching components
    min_partial_fill: float = 0.      # a hangar cut by a straight edge is still rectangular (.56-.94); edge shade is ragged (.43-.60)
    min_rim_contrast: float = 0.      # ring L median minus interior L median, OpenCV 0-255 scale; sprite measures 140
    max_interior_chroma: float = 255.
    angle_offsets: tuple = (-10., 0., 10.)
    partial_heading_step: int = 30     # coarse heading sweep for edge-cut components
    partial_step: int = 8             # slide step along a cut axis, image px
    local_radius: int = 8
    local_step: int = 2
    min_visible: float = .35
    correlation_threshold: float = 0.
    score_threshold: float = 0.
    nms_iou: float = .35
    sift: bool = True

    def __post_init__(self):
        if not 0 < self.min_long_side < self.max_long_side or not 1 <= self.min_aspect < self.max_aspect:
            raise ValueError('Invalid hangar size or aspect window')
        if not all(0 <= v <= 1 for v in (self.min_fill, self.min_partial_area, self.min_partial_fill, self.min_visible, self.correlation_threshold, self.score_threshold, self.nms_iou)):
            raise ValueError('Invalid hangar threshold')


class HangarExpert:
    class_name = 'hangar'

    def __init__(self, bank, settings=None):
        self.settings = settings or HangarSettings()
        self.templates = load_templates(bank, self.class_name)
        self.sift = SiftMatcher(self.templates) if self.settings.sift else None
        self.expected_area = float(np.median([t.mask.sum() for t in self.templates]))
        self._posed = {}

    def templates_for(self, zoom):
        matching = [t for t in self.templates if t.zoom == zoom]
        return matching or self.templates

    def posed(self, template, angle, scale):
        key = (template.id, round(angle, 1), round(scale, 4))
        if key not in self._posed:
            bgr, mask = template.posed(angle, scale)
            gray, high = features(bgr)
            self._posed[key] = (gray, high, mask)
            if len(self._posed) > 512:
                self._posed.pop(next(iter(self._posed)))
        return self._posed[key]

    def detect(self, image, pixels_per_source_pixel=1., zoom=None, explain=False):
        """Return accepted rows; with explain=True also return every candidate and why it was rejected."""
        if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('Expected uint8 BGR image')
        s = float(pixels_per_source_pixel)
        if not np.isfinite(s) or s <= 0:
            raise ValueError('Invalid image scale')
        zoom = ZOOM_FOR_SCALE.get(s, 2) if zoom is None else int(zoom)
        cfg = self.settings
        H, W = image.shape[:2]
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        chroma = np.hypot(lab[:, :, 1].astype(np.float32) - 128, lab[:, :, 2].astype(np.float32) - 128)
        dark = cv2.morphologyEx((L <= cfg.dark_lightness).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(dark)
        gray, high = features(image)
        templates = self.templates_for(zoom)
        rows, candidates = [], []
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            component = labels == label
            touches_edge = x == 0 or y == 0 or x + w >= W or y + h >= H
            candidate = dict(component_area=int(area), bbox_component=[int(x), int(y), int(x + w), int(y + h)], partial=bool(touches_edge))
            if area < 16:
                continue
            pts = np.column_stack(np.where(component))[:, ::-1].astype(np.float32)
            (rcx, rcy), (rw, rh), rangle = cv2.minAreaRect(pts)
            long_side, short_side = max(rw, rh), min(rw, rh)
            heading = (rangle if rw >= rh else rangle + 90.) % 180.
            fill = float(area / max(1., rw * rh))
            candidate.update(long_side_native=long_side / s, short_side_native=short_side / s, aspect=long_side / max(1., short_side), fill=fill, heading=heading)
            if touches_edge:
                if area < cfg.min_partial_area * self.expected_area * s * s or short_side / s > cfg.max_long_side:
                    candidate['rejected_by'] = 'partial_size'; candidates.append(candidate); continue
                if fill < cfg.min_partial_fill:
                    candidate['rejected_by'] = 'partial_fill'; candidates.append(candidate); continue
            else:
                if not cfg.min_long_side <= long_side / s <= cfg.max_long_side:
                    candidate['rejected_by'] = 'size'; candidates.append(candidate); continue
                if not cfg.min_aspect <= long_side / max(1., short_side) <= cfg.max_aspect:
                    candidate['rejected_by'] = 'aspect'; candidates.append(candidate); continue
                if fill < cfg.min_fill:
                    candidate['rejected_by'] = 'fill'; candidates.append(candidate); continue
            interior = cv2.erode(component.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            if interior.sum() < 8:
                interior = component
            ring = cv2.dilate(component.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~component
            rim_contrast = float(np.median(L[ring]) - np.median(L[interior])) if ring.any() else 0.
            interior_chroma = float(np.median(chroma[interior]))
            candidate.update(rim_contrast=rim_contrast, interior_chroma=interior_chroma)
            if interior_chroma > cfg.max_interior_chroma:
                candidate['rejected_by'] = 'chroma'; candidates.append(candidate); continue
            if not touches_edge and rim_contrast < cfg.min_rim_contrast:
                candidate['rejected_by'] = 'rim_contrast'; candidates.append(candidate); continue
            best = None
            cut_x, cut_y = (x == 0 or x + w >= W), (y == 0 or y + h >= H)
            for template in templates:
                if touches_edge:
                    # A cut shape gives no usable heading or centre: sweep headings coarsely and
                    # slide the template along each cut axis, then refine around the best.
                    angles = [hd - template.angle + flip for hd in range(0, 180, cfg.partial_heading_step) for flip in (0., 180.)]
                else:
                    angles = [base + flip + offset for base in (heading - template.angle, template.angle - heading)
                              for flip in (0., 180.) for offset in cfg.angle_offsets]
                for angle in angles:
                    tgray, thigh, tmask = self.posed(template, angle, s)
                    th, tw = tmask.shape
                    if touches_edge:
                        xs = range(-tw // 2, tw // 2 + 1, cfg.partial_step) if cut_x else range(-6, 7, 3)
                        ys = range(-th // 2, th // 2 + 1, cfg.partial_step) if cut_y else range(-6, 7, 3)
                        offsets = [(dx, dy) for dy in ys for dx in xs]
                        hit = local_masked_match(gray, high, tgray, thigh, tmask, rcx, rcy, min_visible=cfg.min_visible, offsets=offsets)
                        if hit:
                            fine = [(dx, dy) for dy in range(-3, 4) for dx in range(-3, 4)]
                            hit = local_masked_match(gray, high, tgray, thigh, tmask, hit[1] + tw / 2, hit[2] + th / 2, min_visible=cfg.min_visible, offsets=fine) or hit
                    else:
                        hit = local_masked_match(gray, high, tgray, thigh, tmask, rcx, rcy, cfg.local_radius, cfg.local_step, cfg.min_visible)
                    if hit and (best is None or hit[0] > best[0]):
                        best = (hit[0], hit[1], hit[2], hit[3], template, angle, tmask.shape)
            if best is None:
                candidate['rejected_by'] = 'no_visible_pose'; candidates.append(candidate); continue
            correlation, x1, y1, visible, template, angle, (th, tw) = best
            candidate.update(correlation=float(correlation), visible_fraction=visible, template_id=template.id, angle=float(angle))
            if correlation < cfg.correlation_threshold:
                candidate['rejected_by'] = 'correlation'; candidates.append(candidate); continue
            rim_score = float(np.clip(rim_contrast / 120., 0, 1))
            chroma_score = float(np.exp(-interior_chroma / 10.))
            score = float(np.clip(.6 * correlation + .2 * rim_score + .2 * chroma_score, 0, 1))
            dx1, dy1, dx2, dy2 = template.organizer_offset
            bbox = [x1 + dx1 * s, y1 + dy1 * s, x1 + tw + dx2 * s, y1 + th + dy2 * s]
            bbox = [float(np.clip(bbox[0], 0, W)), float(np.clip(bbox[1], 0, H)), float(np.clip(bbox[2], 0, W)), float(np.clip(bbox[3], 0, H))]
            candidate.update(score=score, bbox=bbox, mask_bbox=[x1, y1, x1 + tw, y1 + th], rim_score=rim_score, chroma_score=chroma_score)
            if score < cfg.score_threshold:
                candidate['rejected_by'] = 'score'; candidates.append(candidate); continue
            candidate.update({'class': self.class_name, 'family': 'hangar_expert', 'rejected_by': None})
            candidates.append(candidate)
            rows.append(candidate)
        accepted = rows  # one candidate per dark region; adjacent regions are never merged away
        sift_rows = [dict(r, **{'class': self.class_name}) for r in self.sift.match(image, s)] if self.sift else []
        if explain:
            return accepted, sift_rows, candidates
        return accepted, sift_rows
