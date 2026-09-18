"""Blob expert for tiny objects (small and medium launcher): paint colour, size and silhouette.

Built from the class's training sprites only. The sprite foreground gives a Lab colour model
(L, a, b mean and spread); the scene is thresholded on colour likelihood, connected components in
the class's size window become candidates, each compared with the sprite silhouettes at 24
headings (IoU of 32x32 masks) and with the ground around it (contrast). Every measure is recorded;
only the size window gates. Boxes are the component extent plus the organiser offset from the bank.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .common import load_templates


@dataclass(frozen=True)
class BlobSpec:
    class_name: str
    probability: float = .25          # colour likelihood threshold for the mask
    size_window: tuple = (.5, 2.2)    # component long side relative to the sprite, native px
    min_area_fraction: float = .25    # component area relative to the sprite's foreground area
    use_lightness: bool = True        # include L in the colour model (dark launcher) or chroma only (green paint)
    max_candidates: int = 30


class BlobExpert:
    def __init__(self, bank, spec, gate=None):
        self.spec = spec
        self.settings = spec
        self.class_name = spec.class_name
        self.gate = gate
        self.templates = load_templates(bank, spec.class_name)
        native = [t for t in self.templates if t.zoom == max(x.zoom for x in self.templates)] or self.templates
        pixels = np.concatenate([cv2.cvtColor(t.bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[t.mask] for t in native])
        self.center = np.median(pixels, axis=0)
        self.spread = np.maximum(pixels.std(axis=0), [10., 4., 4.])
        self.long_side = float(np.median([max(t.mask.shape) for t in native]))
        self.area = float(np.median([t.mask.sum() for t in native]))
        self.organizer_offset = np.median([t.organizer_offset for t in self.templates], axis=0)
        self.silhouettes, self.aspects = [], []
        for t in native:
            ys, xs = np.where(t.mask)
            patch = t.mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
            h, w = patch.shape
            canvas = np.zeros((64, 64), np.uint8)
            f = 36. / max(w, h)
            resized = cv2.resize(patch, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
            rh, rw = resized.shape
            canvas[(64 - rh) // 2:(64 - rh) // 2 + rh, (64 - rw) // 2:(64 - rw) // 2 + rw] = resized
            for angle in range(0, 360, 15):
                rot = cv2.warpAffine(canvas, cv2.getRotationMatrix2D((31.5, 31.5), angle, 1), (64, 64), flags=cv2.INTER_NEAREST)
                yy, xx = np.where(rot > 0)
                cropped = rot[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
                self.silhouettes.append(cv2.resize(cropped, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255)
                self.aspects.append(cropped.shape[1] / cropped.shape[0])
        self.silhouettes, self.aspects = np.array(self.silhouettes), np.array(self.aspects)
        ring_l = []
        for t in native:
            lab = cv2.cvtColor(t.bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            ring = cv2.dilate(t.mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~t.mask
            if ring.any():
                ring_l.append(np.median(lab[ring], axis=0) - np.median(lab[t.mask], axis=0))
        self.ring_delta = np.median(ring_l, axis=0) if ring_l else np.zeros(3)

    def detect(self, image, pixels_per_source_pixel=1., zoom=None, explain=False):
        if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('Expected uint8 BGR image')
        s = float(pixels_per_source_pixel)
        spec = self.spec
        H, W = image.shape[:2]
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        z = (lab - self.center) / self.spread
        d2 = z[:, :, 1] ** 2 + z[:, :, 2] ** 2 + (z[:, :, 0] ** 2 if spec.use_lightness else 0.)
        probability = np.exp(-.5 * d2)
        mask = cv2.morphologyEx((probability > spec.probability).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        rows, candidates = [], []
        expected = self.long_side * s
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            long_side = max(w, h)
            candidate = dict(cx=float(centroids[label][0]), cy=float(centroids[label][1]), component_area=int(area), long_side_native=long_side / s)
            if not spec.size_window[0] * expected <= long_side <= spec.size_window[1] * expected or area < spec.min_area_fraction * self.area * s * s:
                continue
            component = labels[y:y + h, x:x + w] == label
            target = cv2.resize(component.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA)
            inter = np.minimum(self.silhouettes, target).sum((1, 2)); union = np.maximum(self.silhouettes, target).sum((1, 2))
            shapes = inter / np.maximum(union, 1) * np.exp(-.8 * np.abs(np.log(self.aspects / max(1e-3, w / h))))
            best = int(shapes.argmax())
            colour = float(probability[y:y + h, x:x + w][component].mean())
            radius = max(3, round(5 * s))
            top, left = max(0, y - radius), max(0, x - radius); bottom, right = min(H, y + h + radius), min(W, x + w + radius)
            surround = lab[top:bottom, left:right]; ring = np.ones(surround.shape[:2], bool); ring[y - top:y + h - top, x - left:x + w - left] = False
            delta = np.median(surround[ring], axis=0) - np.median(lab[y:y + h, x:x + w][component], axis=0) if ring.any() else np.zeros(3)
            contrast = float(np.exp(-.5 * np.sum(((delta - self.ring_delta) / 8.) ** 2)))
            partial = x == 0 or y == 0 or x + w >= W or y + h >= H
            dx1, dy1, dx2, dy2 = self.organizer_offset
            bbox = [float(np.clip(x + dx1 * s, 0, W)), float(np.clip(y + dy1 * s, 0, H)), float(np.clip(x + w + dx2 * s, 0, W)), float(np.clip(y + h + dy2 * s, 0, H))]
            score = float(np.clip(.6 * shapes[best] + .2 * colour + .2 * contrast, 0, 1))
            candidate.update({'class': self.class_name, 'family': 'expert', 'bbox': bbox, 'mask_bbox': [float(x), float(y), float(x + w), float(y + h)],
                              'shape_similarity': float(shapes[best]), 'colour_similarity': colour, 'contrast_similarity': contrast, 'fill': float(area / max(1, w * h)),
                              'ring_L_delta': float(delta[0]), 'ring_chroma_delta': float(np.hypot(delta[1], delta[2])), 'partial': bool(partial),
                              'proposer_score': colour, 'proposer_source': 'colour', 'visible_fraction': 1., 'score': score, 'rejected_by': None})
            if self.gate is not None:
                from .fit_gates import apply_gate
                logit, passed = apply_gate(self.gate, candidate, int(zoom if zoom is not None else 2))
                candidate.update(gate_logit=float(logit), gate_probability=float(1 / (1 + np.exp(-logit))), gate_pass=bool(passed))
                candidate['score'] = candidate['gate_probability']
                if not passed:
                    candidate['rejected_by'] = 'gate'; candidates.append(candidate); continue
            candidates.append(candidate)
            rows.append(candidate)
        rows = sorted(rows, key=lambda r: -r['score'])[:spec.max_candidates]
        families = {'expert': rows, 'sift': []}
        return (families, candidates) if explain else families
