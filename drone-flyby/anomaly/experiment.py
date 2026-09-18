"""Class-agnostic anomaly-localization probe at each camera zoom.

This intentionally never uses object names as model inputs or outputs.  It asks a
smaller question than the competition task: if a camera view contains an inserted
object, does generic visual unusualness put a proposal box over it?
"""

from __future__ import annotations

import argparse
import gzip
import json
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


def load_frames(scene):
    for path in sorted((scene/'annotations').glob('*.json')):
        doc = json.loads(path.read_text())
        image = cv2.imread(str(scene/'images'/path.with_suffix('.png').name))
        if image is None:
            raise FileNotFoundError(path)
        yield doc['frame'], doc['annotations'], image


def local_truth(annotations, region):
    left, top, right, bottom = region
    scale = VIEW_W / (right-left)
    truth = []
    for annotation in annotations:
        x1, y1, x2, y2 = annotation['bbox']
        # Only fully visible, non-degenerate objects are scored. This avoids
        # charging a method for guessing the off-camera extent of clipped boxes.
        if x1 >= left and y1 >= top and x2 <= right and y2 <= bottom and x2-x1 >= 4 and y2-y1 >= 4:
            truth.append(dict(object_id=annotation['object_id'],
                              box=[(x1-left)*scale, (y1-top)*scale,
                                   (x2-left)*scale, (y2-top)*scale]))
    return truth


def average_precision(views, threshold, budget):
    """Class-free 101-point interpolated AP with one-to-one matching per view."""
    total_truth = sum(len(view['truth']) for view in views)
    used = {view['id']: set() for view in views}
    ranked = sorted(((score, view, box)
                     for view in views for score, box in view['predictions'][:budget]),
                    key=lambda item: item[0], reverse=True)
    tp = fp = 0
    curve = []
    for score, view, box in ranked:
        available = [(iou(box, target['box']), index)
                     for index, target in enumerate(view['truth'])
                     if index not in used[view['id']]]
        overlap, index = max(available, default=(0.0, -1))
        if overlap >= threshold:
            tp += 1
            used[view['id']].add(index)
        else:
            fp += 1
        curve.append((tp/total_truth if total_truth else 0.0, tp/(tp+fp)))
    ap = sum(max((precision for recall, precision in curve if recall >= point/100), default=0.0)
             for point in range(101))/101
    return dict(ap=ap, true_positives=tp, false_positives=fp,
                false_negatives=total_truth-tp, ground_truth=total_truth,
                predictions=len(ranked), precision=tp/(tp+fp) if tp+fp else 0.0,
                recall=tp/total_truth if total_truth else 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=Path, default=Path(__file__).parents[1]/'src/helsinki')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'results.json')
    parser.add_argument('--raw-output', type=Path, default=Path(__file__).parent/'proposals.json.gz')
    parser.add_argument('--proposal-limit', type=int, default=300)
    args = parser.parse_args()

    views = []
    for frame, annotations, image in load_frames(args.scene):
        anchors = [None] + list(enumerate(annotations))
        for zoom in range(3):
            for anchor in anchors:
                if zoom == 0 and anchor is not None:
                    continue
                if zoom > 0 and anchor is None:
                    continue
                source_box = [0, 0, SOURCE_W, SOURCE_H] if anchor is None else anchor[1]['bbox']
                view_image, _, region = render(image, source_box, zoom)
                truth = local_truth(annotations, region)
                if not truth:
                    continue
                candidates = proposals(view_image, args.proposal_limit)
                views.append(dict(id=f'{frame}:L{zoom}:'+('full' if anchor is None else str(anchor[0])),
                                  frame=frame, zoom=zoom, anchor=None if anchor is None else anchor[0],
                                  view_region=region, truth=truth, predictions=candidates))

    summary = {}
    for zoom in range(3):
        selected = [view for view in views if view['zoom'] == zoom]
        for budget in (25, 100, 300):
            for threshold in (.25, .50):
                summary[f'L{zoom}/budget{budget}/IoU{threshold:.2f}'] = average_precision(
                    selected, threshold, budget)

    with gzip.open(args.raw_output, 'wt') as stream:
        json.dump(dict(view_results=views), stream, separators=(',', ':'))
    report = dict(
        method='class-agnostic colour-rarity + local-contrast + structure proposals',
        protocol={
            'labels_used_by_detector': False,
            'L0': 'one complete-frame view; proposals shared by every target in that frame',
            'L1_L2': 'one deterministic target-containing view per appearance (localization test, not autonomous search)',
            'ranking': 'fixed hand-written anomaly score; no parameter fitting',
        },
        frames=25,
        views=len(views),
        proposal_limit=args.proposal_limit,
        raw_proposals=args.raw_output.name,
        summary=summary,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
