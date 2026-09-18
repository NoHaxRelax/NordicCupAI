"""Multi-view, multi-scale normalized pixel correlation on delivered camera views.

OpenCV images use BGR. Output boxes are in the input image's pixel coordinates.
Scores are similarity scores, not calibrated probabilities.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


def nms(rows, threshold=.35, limit=300):
    kept = []
    for row in sorted(rows, key=lambda r: (-r['score'], r['class'], r['bbox'], r['template_id'])):
        if not any(row['class'] == k['class'] and iou(row['bbox'], k['bbox']) > threshold for k in kept):
            kept.append(row)
            if len(kept) >= limit:
                break
    return kept


def features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
    # Remove broad illumination/background gradients, retain sprite detail.
    high = gray - cv2.GaussianBlur(gray, (0, 0), 2.)
    return gray, high


def correlation(a, b):
    a = a.astype(np.float32).reshape(-1, a.shape[-1] if a.ndim == 3 else 1)
    b = b.astype(np.float32).reshape(a.shape)
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.sum(a*b) / denom) if denom > 1e-8 else 0.


@dataclass(frozen=True)
class Settings:
    scales: tuple = (.85, 1., 1.18)
    angles: tuple = (-12., 0., 12.)
    proposal_threshold: float = .4
    score_threshold: float = .8
    peaks_per_template: int = 8
    nms_iou: float = .35
    max_detections: int = 300
    min_mask_pixels: int = 32
    mask_mode: str = 'context_penalty'

    def __post_init__(self):
        if not self.scales or any(not np.isfinite(s) or s <= 0 for s in self.scales):
            raise ValueError('Scales must be positive and finite')
        if not self.angles or any(not np.isfinite(a) for a in self.angles):
            raise ValueError('Angles must be finite')
        if not 0 <= self.proposal_threshold <= 1 or not 0 <= self.score_threshold <= 1:
            raise ValueError('Thresholds must be in [0, 1]')
        if self.peaks_per_template < 1 or self.max_detections < 1 or not 0 <= self.nms_iou <= 1:
            raise ValueError('Invalid peak/NMS settings')
        if self.min_mask_pixels < 8 or self.mask_mode not in ('context_penalty', 'masked_ncc', 'photometric'):
            raise ValueError('Invalid mask settings')


class TemplateDetector:
    def __init__(self, bank, settings=None):
        self.bank = Path(bank)
        self.settings = settings or Settings()
        self.manifest = json.loads((self.bank/'manifest.json').read_text())
        self.templates = []
        for row in self.manifest['templates']:
            path = self.bank/row['file']
            if sha(path) != row['sha256']:
                raise ValueError('Template checksum mismatch: '+str(path))
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Unreadable template: '+str(path))
            mask = None
            if row.get('mask_file'):
                mask_path = self.bank/row['mask_file']
                if sha(mask_path) != row['mask_sha256']:
                    raise ValueError('Mask checksum mismatch: '+str(mask_path))
                mask = cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape != image.shape[:2] or not np.any(mask):
                    raise ValueError('Invalid foreground mask')
            self.templates.append((row, image, mask))
        if not self.templates:
            raise ValueError('Empty template bank')
        self._variants = {}

    def variants(self, pixels_per_source_pixel):
        key = float(pixels_per_source_pixel)
        if not np.isfinite(key) or key <= 0:
            raise ValueError('pixels_per_source_pixel must be positive and finite')
        if key in self._variants:
            return self._variants[key]
        variants = []
        for row, original, foreground in self.templates:
            for scale in self.settings.scales:
                w, h = [round(v*key*scale) for v in original.shape[1::-1]]
                if min(w, h) < 5:
                    continue
                resized = cv2.resize(original, (w, h), interpolation=cv2.INTER_AREA if key*scale < 1 else cv2.INTER_LINEAR)
                resized_mask = (np.full((h,w),255,np.uint8) if foreground is None else
                                cv2.resize(foreground,(w,h),interpolation=cv2.INTER_NEAREST))
                for angle in self.settings.angles:
                    # Expanded rotation, with a validity mask for padded corners.
                    matrix = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), angle, 1.)
                    corners = cv2.transform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), matrix)[0]
                    low, high = np.floor(corners.min(0)), np.ceil(corners.max(0))
                    matrix[:, 2] -= low
                    rw, rh = (high-low).astype(int)
                    patch = cv2.warpAffine(resized, matrix, (rw, rh), borderMode=cv2.BORDER_REPLICATE)
                    mask = cv2.warpAffine(resized_mask, matrix, (rw, rh)) == 255
                    gray, highpass = features(patch)
                    minimum = self.settings.min_mask_pixels if foreground is not None else 8
                    if mask.sum() < minimum or gray[mask].std() < .015 or highpass[mask].std() < .003:
                        continue
                    variants.append((row, scale, angle, patch, gray, highpass, mask))
        # Bound cached resolutions for long-lived servers accepting arbitrary crops.
        if len(self._variants) >= 3:
            self._variants.pop(next(iter(self._variants)))
        self._variants[key] = variants
        return variants

    def detect(self, image, pixels_per_source_pixel=1.):
        """Blind image search. Supply .25/.5/1 for delivered L0/L1/L2 views.

        No frame index, target box, trajectory, or evaluation label is consumed.
        Fully visible objects only; cropped edge objects are a known limitation.
        """
        if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('Expected uint8 BGR image')
        gray, highpass = features(image)
        rows = []
        cfg = self.settings
        for row, scale, angle, patch, tg, th, mask in self.variants(pixels_per_source_pixel):
            h, w = tg.shape
            if h > image.shape[0] or w > image.shape[1]:
                continue
            # Two complementary normalized correlation maps reduce flat-ground matches.
            if cfg.mask_mode == 'photometric':
                core=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))
                if core.sum()<8:
                    core=mask.astype(np.uint8)
                difference=cv2.matchTemplate(image.astype(np.float32)/255.,patch.astype(np.float32)/255.,cv2.TM_SQDIFF_NORMED,mask=core)
                appearance=np.exp(-4*np.maximum(np.nan_to_num(difference,nan=10.,posinf=10.),0.))
                detail=cv2.matchTemplate(highpass,th,cv2.TM_CCOEFF_NORMED,mask=core)
                detail=np.clip(np.nan_to_num(detail,nan=0.,posinf=0.,neginf=0.),0.,1.)
                response=.5*detail+.5*appearance
            elif row.get('mask_file') and cfg.mask_mode == 'context_penalty':
                # Match foreground detail but normalize by the WHOLE candidate's
                # energy: a tiny dark fragment must not match arbitrary busy texture.
                kernel = (th-th[mask].mean())*mask
                response = .75*cv2.matchTemplate(highpass, kernel, cv2.TM_CCORR_NORMED)
                response += .25*cv2.matchTemplate(gray, tg, cv2.TM_CCOEFF_NORMED)
            else:
                kwargs = {'mask':mask.astype(np.uint8)} if row.get('mask_file') else {}
                response = .45*cv2.matchTemplate(gray, tg, cv2.TM_CCOEFF_NORMED, **kwargs)
                response += .55*cv2.matchTemplate(highpass, th, cv2.TM_CCOEFF_NORMED, **kwargs)
            response = np.nan_to_num(response, nan=-1., posinf=-1., neginf=-1.)
            for _ in range(cfg.peaks_per_template):
                _, peak, _, (x, y) = cv2.minMaxLoc(response)
                if peak < cfg.proposal_threshold:
                    break
                candidate = image[y:y+h, x:x+w]
                color = correlation(candidate[mask].reshape(-1, 1, 3), patch[mask].reshape(-1, 1, 3))
                # Color agreement checks raw pixels after the luminance/detail search.
                score = float(np.clip(peak if cfg.mask_mode=='photometric' else .75*peak + .25*max(0., color), 0., 1.))
                if score >= cfg.score_threshold:
                    rows.append({'class': row['class'], 'bbox': [x, y, x+w, y+h],
                                 'score': score, 'pixel_correlation': color,
                                 'template_id': row['id'], 'scale': scale, 'angle': angle})
                rx, ry = max(1, w//3), max(1, h//3)
                response[max(0,y-ry):y+ry+1, max(0,x-rx):x+rx+1] = -1.
        return nms(rows, cfg.nms_iou, cfg.max_detections)

    def tracking_detections(self, image, view):
        """Adapter for DroneTrackingWorkflow.process(request, detections, image=...)."""
        from drone.perspective_tracking.revisit import Detection
        if image.shape[1::-1] != view.image_size:
            raise ValueError('Image dimensions disagree with delivered view metadata')
        sx, sy = image.shape[1]/(view.region[2]-view.region[0]), image.shape[0]/(view.region[3]-view.region[1])
        if abs(sx-sy) > 1e-6:
            raise ValueError('Anisotropically resized views are unsupported')
        return [Detection(r['class'], tuple(r['bbox']), r['score']) for r in self.detect(image,sx)]

    def predict(self, image, source_region, source_size=(3840, 2160)):
        """Convert a delivered crop to organizer-style normalized full-frame boxes."""
        x1, y1, x2, y2 = source_region
        if not 0 <= x1 < x2 <= source_size[0] or not 0 <= y1 < y2 <= source_size[1]:
            raise ValueError('Invalid source region')
        sx, sy = image.shape[1]/(x2-x1), image.shape[0]/(y2-y1)
        if abs(sx-sy) > 1e-6:
            raise ValueError('Anisotropically resized views are unsupported')
        return [{'object_id': r['class'], 'confidence': r['score'],
                 'bbox': [(r['bbox'][0]/sx+x1)/source_size[0], (r['bbox'][1]/sy+y1)/source_size[1],
                          (r['bbox'][2]/sx+x1)/source_size[0], (r['bbox'][3]/sy+y1)/source_size[1]]}
                for r in self.detect(image, sx)]
