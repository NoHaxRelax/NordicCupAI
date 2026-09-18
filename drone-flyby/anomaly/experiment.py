"""Class-agnostic anomaly-localization probe at each camera zoom.

This intentionally never uses object names as model inputs or outputs.  It asks a
smaller question than the competition task: if a camera view contains an inserted
object, does generic visual unusualness put a proposal box over it?
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


VIEW_W, VIEW_H = 960, 540
SOURCE_W, SOURCE_H = 3840, 2160


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def robust_unit(value):
    lo, hi = np.percentile(value, (50, 99.7))
    return np.clip((value - lo) / max(float(hi - lo), 1e-6), 0, 1)


def anomaly_map(image):
    """Combine colour rarity, local contrast and structure without learned classes."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Globally rare colours are a useful compositing cue. Coarse bins make the
    # statistic stable at all three zooms and avoid memorising exact RGB values.
    quantized = (lab / 16).astype(np.int32)
    code = quantized[:, :, 0] * 256 + quantized[:, :, 1] * 16 + quantized[:, :, 2]
    counts = np.bincount(code.ravel(), minlength=4096)
    rarity = -np.log((counts[code] + 8) / (code.size + 8 * 4096))
    rarity = robust_unit(cv2.GaussianBlur(rarity.astype(np.float32), (0, 0), 1.2))

    # A sprite may also disagree with its immediate terrain even when its colour
    # is common elsewhere in the image.
    residuals = []
    for sigma in (2.0, 5.0, 11.0):
        local = cv2.GaussianBlur(lab, (0, 0), sigma)
        residuals.append(np.linalg.norm(lab - local, axis=2))
    contrast = robust_unit(np.maximum.reduce(residuals))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    structure = robust_unit(cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), 1.0))

    score = 0.42 * rarity + 0.38 * contrast + 0.20 * structure
    return cv2.GaussianBlur(score, (0, 0), 0.8)


def proposals(image, limit=300):
    score = anomaly_map(image)
    found = []
    # Multiple thresholds preserve compact small targets and larger structures.
    for percentile in (99.7, 99.2, 98.0, 96.0):
        mask = (score >= np.percentile(score, percentile)).astype(np.uint8) * 255
        for kernel_size in (3, 7):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            count, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
            for x, y, w, h, area in stats[1:count]:
                if area < 3 or w > 500 or h > 350:
                    continue
                region_score = float(score[y:y+h, x:x+w].mean())
                cx, cy = x+w/2, y+h/2
                # Segmentation commonly isolates just a wing, roof, or shadow.
                # Propose several class-neutral extents around the same locus.
                for extent, penalty in ((1.7, 1.0), (3.0, .82), (5.0, .65)):
                    bw, bh = max(7, w*extent), max(7, h*extent)
                    box = [max(0, round(cx-bw/2)), max(0, round(cy-bh/2)),
                           min(VIEW_W, round(cx+bw/2)), min(VIEW_H, round(cy+bh/2))]
                    found.append((region_score * np.sqrt(area) * penalty, box))

    # Score-ordered class-agnostic NMS.
    kept = []
    for confidence, box in sorted(found, reverse=True):
        if all(iou(box, other[1]) < 0.65 for other in kept):
            kept.append((confidence, box))
            if len(kept) == limit:
                break
    return kept


def render(image, box, zoom):
    region_w, region_h = SOURCE_W // 2**zoom, SOURCE_H // 2**zoom
    cx, cy = (box[0]+box[2]) / 2, (box[1]+box[3]) / 2
    left = int(np.clip(round(cx-region_w/2), 0, SOURCE_W-region_w))
    top = int(np.clip(round(cy-region_h/2), 0, SOURCE_H-region_h))
    view = image[top:top+region_h, left:left+region_w]
    if zoom != 2:
        view = cv2.resize(view, (VIEW_W, VIEW_H), interpolation=cv2.INTER_AREA)
    scale = VIEW_W / region_w
    local_box = [(box[0]-left)*scale, (box[1]-top)*scale,
                 (box[2]-left)*scale, (box[3]-top)*scale]
    return view, local_box, [left, top, left+region_w, top+region_h]


def load_appearances(scene):
    for path in sorted((scene/'annotations').glob('*.json')):
        doc = json.loads(path.read_text())
        image = cv2.imread(str(scene/'images'/path.with_suffix('.png').name))
        if image is None:
            raise FileNotFoundError(path)
        for index, annotation in enumerate(doc['annotations']):
            # Edge-clipped boxes cannot fairly meet IoU 0.5 in a target-centred crop.
            box = annotation['bbox']
            if box[2]-box[0] >= 4 and box[3]-box[1] >= 4:
                yield doc['frame'], index, annotation['object_id'], box, image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=Path, default=Path(__file__).parents[1]/'src/helsinki')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'results.json')
    parser.add_argument('--proposal-limit', type=int, default=300)
    args = parser.parse_args()

    rows = []
    cache = {}
    for frame, index, object_id, source_box, image in load_appearances(args.scene):
        for zoom in range(3):
            view, target, region = render(image, source_box, zoom)
            # L0 is identical for all objects in a frame; avoid recomputing it.
            key = (frame, zoom, *region)
            candidates = cache.setdefault(key, proposals(view, args.proposal_limit))
            overlaps = [iou(target, candidate[1]) for candidate in candidates]
            rows.append(dict(frame=frame, annotation_index=index, zoom=zoom,
                             object_id=object_id, source_box=source_box, view_region=region,
                             target_box=target, proposal_count=len(candidates),
                             best_iou=max(overlaps, default=0.0),
                             rank_at_iou_025=next((i+1 for i, value in enumerate(overlaps) if value >= .25), None),
                             rank_at_iou_050=next((i+1 for i, value in enumerate(overlaps) if value >= .50), None)))

    summary = {}
    for zoom in range(3):
        selected = [r for r in rows if r['zoom'] == zoom]
        for budget in (25, 100, 300):
            summary[f'L{zoom}/recall@{budget}/IoU0.25'] = sum(
                r['rank_at_iou_025'] is not None and r['rank_at_iou_025'] <= budget for r in selected) / len(selected)
            summary[f'L{zoom}/recall@{budget}/IoU0.50'] = sum(
                r['rank_at_iou_050'] is not None and r['rank_at_iou_050'] <= budget for r in selected) / len(selected)
        summary[f'L{zoom}/mean_best_iou'] = float(np.mean([r['best_iou'] for r in selected]))
        classes = defaultdict(list)
        for row in selected:
            classes[row['object_id']].append(row['rank_at_iou_025'] is not None)
        summary[f'L{zoom}/class_recall/IoU0.25'] = {name: sum(values)/len(values) for name, values in sorted(classes.items())}

    report = dict(
        method='class-agnostic colour-rarity + local-contrast + structure proposals',
        protocol={
            'labels_used_by_detector': False,
            'L0': 'one complete-frame view; proposals shared by every target in that frame',
            'L1_L2': 'one deterministic target-containing view per appearance (localization test, not autonomous search)',
            'ranking': 'fixed hand-written anomaly score; no parameter fitting',
        },
        appearances=len(rows)//3,
        proposal_limit=args.proposal_limit,
        summary=summary,
        rows=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
