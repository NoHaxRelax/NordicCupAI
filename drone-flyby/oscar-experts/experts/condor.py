"""Condor expert: an X-shaped four-engine aircraft recognised by shape and a coarse lightness pattern.

Branches, each reported on every candidate so they can be compared:
  proposer      -> heading-sweep masked correlation at half resolution (coarse structure only)
  part model    -> two bars, recurring dark spots, ground between the arms; scored as a relative
                   pattern over an affine pose family (heading, scale, one-axis stretch, shear)
  colour        -> low chroma on the bars
  competitors   -> the same part scoring with the jet, medium-plane and small-plane models
  pixel         -> full masked correlation with the sprite (the exact-look branch, kept separate)
  sift          -> geometric comparison branch, kept separate
Heading is free over 360 degrees. The output box is the posed outline plus the organiser offset.
"""
from dataclasses import dataclass
from itertools import product

import cv2
import numpy as np

from .common import CorrelationProposer, features, load_templates, local_masked_match, nms, SiftMatcher, fit_bars
from .parts import PartModel

ZOOM_FOR_SCALE = {.25: 0, .5: 1, 1.: 2}


@dataclass(frozen=True)
class CondorSettings:
    heading_step: int = 15
    proposer_threshold: float = .2
    max_candidates: int = 12
    duplicate_fraction: float = .33         # same object only if fitted centres are this close, in fuselage lengths
    heading_sweep: int = 30                  # the part model resolves heading itself over 360 degrees
    heading_refine: tuple = (-15., -7.5, 7.5, 15.)
    scales: tuple = (1.,)
    stretches: tuple = (.87, 1., 1.15)      # one-axis scale, covers foreshortening
    shears: tuple = (0.,)
    centre_offsets: tuple = (-4., 0., 4.)
    min_pattern: float = 0.           # recorded, not a gate: the verifier rules candidates out, not the expert
    min_xness: float = 0.             # bars must differ from the ground between the arms (true condor .6-.8, noise .02-.5)
    min_spot: float = 0.              # the recurring dark spots must be present (true .35-1.0, noise .05-.1)
    min_proposer: float = .2          # coarse-structure correlation floor for acceptance (true .9+, ground .35-.5)
    max_bar_chroma: float = 255.
    min_visible: float = .6
    competitor_margin: float = -1.
    competitors: tuple = ('jet_plane', 'medium_plane', 'small_plane')
    competitor_size_window: tuple = (.7, 1.4)   # altitude fixes size: only aircraft this close in length can be confused
    pixel_angle_offsets: tuple = (-10., -5., 0., 5., 10.)
    pixel_threshold: float = .35
    score_threshold: float = 0.
    nms_iou: float = .35
    sift: bool = False                       # comparison branch: 1/42 on training tiles, and the slowest stage

    def __post_init__(self):
        if not all(0 <= v <= 1 for v in (self.proposer_threshold, self.min_pattern, self.min_spot, self.min_proposer, self.min_visible, self.pixel_threshold, self.score_threshold, self.nms_iou)):
            raise ValueError('Invalid condor threshold')
        if self.heading_step < 1 or self.max_candidates < 1:
            raise ValueError('Invalid proposer settings')


class CondorExpert:
    class_name = 'condor'

    def __init__(self, bank, settings=None):
        self.settings = settings or CondorSettings()
        cfg = self.settings
        self.templates = load_templates(bank, self.class_name)
        self.model = PartModel.from_templates(self.class_name, self.templates)
        self.competitors = {}
        for name in cfg.competitors:
            try:
                self.competitors[name] = PartModel.from_templates(name, load_templates(bank, name))
            except ValueError:
                continue
        self.proposer = CorrelationProposer(self.templates, headings=range(0, 360, cfg.heading_step), threshold=cfg.proposer_threshold)
        self.sift = SiftMatcher(self.templates) if cfg.sift else None
        self.sprite_angles = {t.id: fit_bars(t.mask)['fuselage_angle'] for t in self.templates}
        self.organizer_offset = np.median([t.organizer_offset for t in self.templates], axis=0)

    def templates_for(self, zoom):
        matching = [t for t in self.templates if t.zoom == zoom]
        return matching or self.templates

    def proposer_templates_for(self, zoom):
        return [max(self.templates_for(zoom), key=lambda t: t.mask.sum())]

    def _combined(self, part, model):
        chroma_score = float(np.exp(-max(0., part['bar_chroma'] - model.chroma) / 10.))
        return float(np.clip(.55 * part['pattern_ncc'] + .2 * min(part['xness'] / 3., 1.) + .15 * part['spot_score'] + .1 * chroma_score, 0, 1)), chroma_score

    def _best_pose(self, model, L, chroma, cx, cy, s, headings, scales, stretches=(1.,), shears=(0.,), offsets=(0.,)):
        """Coordinate descent over the pose family, each stage sampled as one batch; same result as the per-pose version."""
        def stage(poses):
            pts = np.stack([model.pose_points(p['heading'], s * p['scale'], p['stretch'], 1., p['shear'], p['cx'], p['cy']) for p in poses])
            best = None
            for pose, part in zip(poses, model.score_many(L, chroma, pts, self.settings.min_visible)):
                if part is None:
                    continue
                combined, _ = self._combined(part, model)
                if best is None or combined > best[0]:
                    best = (combined, part, pose)
            return best
        best = stage([dict(heading=h, scale=sc, stretch=1., shear=0., cx=cx, cy=cy) for h in headings for sc in scales])
        if best is None:
            return None
        pose = best[2]
        refined = stage([dict(pose, heading=pose['heading'] + d) for d in self.settings.heading_refine])
        if refined is not None and refined[0] > best[0]:
            best = refined
        pose = best[2]
        warped = stage([dict(pose, stretch=st, shear=sh) for st in stretches for sh in shears])
        if warped is not None and warped[0] > best[0]:
            best = warped
        pose = best[2]
        shifted = stage([dict(pose, cx=cx + dx, cy=cy + dy) for dx in offsets for dy in offsets if not (dx == 0 and dy == 0)])
        if shifted is not None and shifted[0] > best[0]:
            best = shifted
        return best

    @staticmethod
    def _match_window(padded, tg, th, tmask, cx, cy, radius):
        gray_p, high_p, margin = padded
        h, w = tmask.shape
        x0 = int(round(cx - w / 2 - radius)) + margin; y0 = int(round(cy - h / 2 - radius)) + margin
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

    def _dedup(self, rows, s):
        """Merge only true duplicates: one object seen at several headings has one centre. Neighbours survive."""
        kept = []
        for row in sorted(rows, key=lambda r: -r['score']):
            limit = self.settings.duplicate_fraction * self.model.fuselage_length * s * row['pose']['scale']
            if not any(np.hypot(row['pose']['cx'] - k['pose']['cx'], row['pose']['cy'] - k['pose']['cy']) < limit for k in kept):
                kept.append(row)
        return kept

    def detect(self, image, pixels_per_source_pixel=1., zoom=None, explain=False, proposals=None):
        if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('Expected uint8 BGR image')
        s = float(pixels_per_source_pixel)
        if not np.isfinite(s) or s <= 0:
            raise ValueError('Invalid image scale')
        zoom = ZOOM_FOR_SCALE.get(s, 2) if zoom is None else int(zoom)
        cfg = self.settings
        H, W = image.shape[:2]
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, chroma = lab[:, :, 0], np.hypot(lab[:, :, 1] - 128, lab[:, :, 2] - 128)
        gray, high = features(image)
        templates = self.templates_for(zoom)
        margin = int(self.model.fuselage_length * s * 1.5) + 16
        padded = (cv2.copyMakeBorder(gray, margin, margin, margin, margin, cv2.BORDER_REFLECT), cv2.copyMakeBorder(high, margin, margin, margin, margin, cv2.BORDER_REFLECT), margin)
        raw = proposals if proposals is not None else self.proposer.propose(image, s, self.proposer_templates_for(zoom))
        proposals = self.proposer.merge(raw, radius=.3 * self.model.fuselage_length * s)[:cfg.max_candidates]
        rows, pixel_rows, candidates = [], [], []
        for prop in proposals:
            cx, cy = prop['cx'], prop['cy']
            candidate = dict(cx=cx, cy=cy, proposer_score=prop['proposer_score'], proposer_heading=prop['heading'], template_id=prop['template_id'])
            best = self._best_pose(self.model, L, chroma, cx, cy, s, range(0, 360, cfg.heading_sweep), cfg.scales, cfg.stretches, cfg.shears, cfg.centre_offsets)
            if best is None:
                candidate['rejected_by'] = 'not_visible'; candidates.append(candidate); continue
            combined, part, pose = best
            _, chroma_score = self._combined(part, self.model)
            candidate.update(part, pose=pose, chroma_score=chroma_score, part_score=combined)
            # competitors: same scoring with each other aircraft's model, free heading, same altitude scale
            margins, applicable = {}, {}
            fitted_length = self.model.fuselage_length * pose['scale']
            for name, model in self.competitors.items():
                ratio = model.fuselage_length / fitted_length
                applicable[name] = bool(cfg.competitor_size_window[0] <= ratio <= cfg.competitor_size_window[1])
                if not applicable[name]:
                    margins[name] = None
                    continue
                alt = self._best_pose(model, L, chroma, pose['cx'], pose['cy'], s, range(0, 360, cfg.heading_sweep), cfg.scales)
                margins[name] = float(alt[0]) if alt else 0.
            candidate['competitor_scores'] = margins
            candidate['competitor_applicable'] = applicable
            relevant = [v for k, v in margins.items() if applicable[k] and v is not None]
            candidate['competitor_margin'] = float(combined - max(relevant)) if relevant else 1.
            # the exact-look branch, for comparison: masked correlation with the sprite at this heading
            best_pixel = None
            for template in templates[:1]:
                for base_angle in {(pose['heading'] - self.sprite_angles[template.id]) % 360, (self.sprite_angles[template.id] - pose['heading']) % 360}:
                    for flip_offset in (0., 180.):
                        tbgr, tmask = template.posed(base_angle + flip_offset, s)
                        tg, th = features(tbgr)
                        hit = self._match_window(padded, tg, th, tmask, pose['cx'], pose['cy'], 6)
                        if hit is None:
                            continue
                        peak, px1, py1 = hit
                        th_, tw_ = tmask.shape
                        vis = (max(0, min(W, px1 + tw_) - max(0, px1)) * max(0, min(H, py1 + th_) - max(0, py1))) / float(tw_ * th_)
                        if vis >= cfg.min_visible and (best_pixel is None or peak > best_pixel[0]):
                            best_pixel = (peak, px1, py1, vis, template, tmask.shape)
            candidate['pixel_correlation'] = float(best_pixel[0]) if best_pixel else 0.
            outline = self.model.pose_outline(pose['heading'], s * pose['scale'], pose['stretch'], 1., pose['shear'], pose['cx'], pose['cy'])
            low, high_ = outline.min(0), outline.max(0)
            dx1, dy1, dx2, dy2 = self.organizer_offset
            bbox = [float(np.clip(low[0] + dx1 * s, 0, W)), float(np.clip(low[1] + dy1 * s, 0, H)), float(np.clip(high_[0] + dx2 * s, 0, W)), float(np.clip(high_[1] + dy2 * s, 0, H))]
            candidate.update(bbox=bbox, outline_bbox=[float(v) for v in (*low, *high_)], partial=bool(part['visible_fraction'] < .999))
            if best_pixel and best_pixel[0] >= cfg.pixel_threshold:
                _, px1, py1, _, template, (th_, tw_) = best_pixel
                pixel_rows.append({'class': self.class_name, 'family': 'pixel', 'score': float(best_pixel[0]),
                                   'bbox': [float(np.clip(px1 + dx1 * s, 0, W)), float(np.clip(py1 + dy1 * s, 0, H)), float(np.clip(px1 + tw_ + dx2 * s, 0, W)), float(np.clip(py1 + th_ + dy2 * s, 0, H))],
                                   'template_id': template.id})
            if part['pattern_ncc'] < cfg.min_pattern:
                candidate['rejected_by'] = 'pattern'; candidates.append(candidate); continue
            if part['bar_chroma'] > cfg.max_bar_chroma:
                candidate['rejected_by'] = 'chroma'; candidates.append(candidate); continue
            if part['xness'] < cfg.min_xness:
                candidate['rejected_by'] = 'xness'; candidates.append(candidate); continue
            if part['spot_score'] < cfg.min_spot:
                candidate['rejected_by'] = 'spots'; candidates.append(candidate); continue
            if prop['proposer_score'] < cfg.min_proposer:
                candidate['rejected_by'] = 'proposer'; candidates.append(candidate); continue
            if candidate['competitor_margin'] < cfg.competitor_margin:
                candidate['rejected_by'] = 'competitor'; candidates.append(candidate); continue
            if combined < cfg.score_threshold:
                candidate['rejected_by'] = 'score'; candidates.append(candidate); continue
            candidate.update({'class': self.class_name, 'family': 'condor_expert', 'score': combined, 'rejected_by': None})
            candidates.append(candidate)
            rows.append(candidate)
        families = {'condor_expert': self._dedup(rows, s), 'pixel': nms(pixel_rows, cfg.nms_iou),
                    'sift': [dict(r, **{'class': self.class_name}) for r in self.sift.match(image, s)] if self.sift else []}
        return (families, candidates) if explain else families
