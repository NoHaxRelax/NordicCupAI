#!/usr/bin/env python3
"""Offline SIFT search for the four classes absent from validation v4."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data/drone/reference/helsinki"
VALIDATION = ROOT / "data/drone/reconstructed-validation"
OUTPUT = ROOT / "artifacts/drone-validation-coverage/missing-class-sift-v2"
CLASSES = {"small_plane", "ta-ta", "jammer", "spacecraft"}
WIDTH, HEIGHT = 3840, 2160


def iou(a: list[float], b: list[float]) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def main() -> None:
    cv2.setNumThreads(4)
    cv2.setRNGSeed(0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    crop_root = OUTPUT / "candidates"
    crop_root.mkdir(exist_ok=True)
    sift = cv2.SIFT_create(nfeatures=16000, contrastThreshold=0.012, edgeThreshold=15)
    documents = {
        int(path.stem.split("_")[-1]): json.loads(path.read_text())
        for path in (REFERENCE / "annotations").glob("*.json")
    }
    by_class = defaultdict(list)
    image_cache = {}
    for frame, document in sorted(documents.items()):
        for annotation in document["annotations"]:
            if annotation["object_id"] not in CLASSES:
                continue
            box = list(map(int, annotation["bbox"]))
            if min(box[0], box[1]) <= 0 or box[2] >= WIDTH or box[3] >= HEIGHT:
                continue
            by_class[annotation["object_id"]].append((frame, box))

    bank = []
    for label, examples in sorted(by_class.items()):
        indices = sorted(set([0, len(examples) // 4, len(examples) // 2, 3 * len(examples) // 4, len(examples) - 1]))
        for index in indices:
            frame, box = examples[index]
            if frame not in image_cache:
                image_cache[frame] = cv2.imread(
                    str(REFERENCE / "images" / f"frame_{frame:06d}.png"), cv2.IMREAD_GRAYSCALE
                )
            x1, y1, x2, y2 = box
            padding = 4
            x1, y1, x2, y2 = max(0, x1 - padding), max(0, y1 - padding), min(WIDTH, x2 + padding), min(HEIGHT, y2 + padding)
            patch = image_cache[frame][y1:y2, x1:x2]
            enlarged = cv2.resize(patch, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            keypoints, descriptors = sift.detectAndCompute(enlarged, None)
            if descriptors is None or len(keypoints) < 3:
                continue
            points = np.float32([keypoint.pt for keypoint in keypoints]) / 2.0
            bank.append(
                {
                    "class": label,
                    "frame": frame,
                    "size": (x2 - x1, y2 - y1),
                    "points": points,
                    "descriptors": descriptors,
                }
            )

    frames = list(range(5, 13)) + list(range(13, 250, 4))
    proposals = []
    timing = []
    for frame in frames:
        start = time.monotonic()
        image = cv2.imread(str(VALIDATION / f"frame_{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
        review_height = HEIGHT if frame <= 12 else 700
        search = image[:review_height]
        keypoints, descriptors = sift.detectAndCompute(search, None)
        if descriptors is None:
            continue
        scene_points = np.float32([keypoint.pt for keypoint in keypoints])
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
        matcher.add([descriptors])
        matcher.train()
        frame_rows = []
        for template in bank:
            pairs = matcher.knnMatch(template["descriptors"], k=2)
            good = [left for left, right in pairs if left.distance < 0.88 * right.distance]
            if len(good) < 3:
                continue
            source = np.float32([template["points"][match.queryIdx] for match in good])
            target = np.float32([scene_points[match.trainIdx] for match in good])
            transform, inliers = cv2.estimateAffinePartial2D(
                source,
                target,
                method=cv2.RANSAC,
                ransacReprojThreshold=4,
                maxIters=4000,
                confidence=0.995,
                refineIters=10,
            )
            if transform is None:
                continue
            count = int(inliers.sum())
            fraction = float(inliers.mean())
            scale = float(np.linalg.norm(transform[:, 0]))
            if count < 3 or fraction < 0.40 or not 0.25 < scale < 4.0:
                continue
            width, height = template["size"]
            corners = np.float32([[[0, 0], [width, 0], [width, height], [0, height]]])
            projected = cv2.transform(corners, transform)[0]
            low, high = projected.min(axis=0), projected.max(axis=0)
            box = [float(low[0]), float(low[1]), float(high[0]), float(high[1])]
            if not (0 <= box[0] < box[2] <= WIDTH and 0 <= box[1] < box[3] <= review_height):
                continue
            median_distance = float(np.median([match.distance for match in good]))
            row = {
                "frame": frame,
                "class": template["class"],
                "bbox_source_xyxy": box,
                "inliers": count,
                "fraction": fraction,
                "scale": scale,
                "median_descriptor_distance": median_distance,
                "template_frame": template["frame"],
                "rank_score": count * fraction / (1.0 + median_distance / 100.0),
                "unverified": True,
            }
            frame_rows.append(row)
        frame_rows.sort(key=lambda row: row["rank_score"], reverse=True)
        unique = []
        for row in frame_rows:
            if any(
                row["class"] == other["class"]
                and iou(row["bbox_source_xyxy"], other["bbox_source_xyxy"]) > 0.35
                for other in unique
            ):
                continue
            unique.append(row)
        proposals.extend(unique[:12])
        timing.append(
            {
                "frame": frame,
                "review_height": review_height,
                "keypoints": len(keypoints),
                "proposals": len(unique[:12]),
                "seconds": time.monotonic() - start,
            }
        )
        print(json.dumps(timing[-1]), flush=True)

    proposals.sort(key=lambda row: row["rank_score"], reverse=True)
    top = proposals[:80]
    tiles = []
    for index, row in enumerate(top, start=1):
        image = cv2.imread(str(VALIDATION / f"frame_{row['frame']:06d}.png"))
        x1, y1, x2, y2 = row["bbox_source_xyxy"]
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        side = 192
        left = max(0, min(WIDTH - side, int(round(center_x - side / 2))))
        top_y = max(0, min(HEIGHT - side, int(round(center_y - side / 2))))
        tile = image[top_y : top_y + side, left : left + side].copy()
        local_box = np.round(np.asarray([x1 - left, y1 - top_y, x2 - left, y2 - top_y])).astype(int)
        cv2.rectangle(tile, tuple(local_box[:2]), tuple(local_box[2:]), (0, 255, 255), 2)
        label = f"{index:02d} f{row['frame']:03d} {row['class']} i{row['inliers']}"
        cv2.rectangle(tile, (0, 0), (side, 24), (0, 0, 0), -1)
        cv2.putText(tile, label, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        filename = f"{index:02d}-f{row['frame']:03d}-{row['class']}.png"
        cv2.imwrite(str(crop_root / filename), tile)
        row["review_crop"] = f"candidates/{filename}"
        tiles.append(tile)
    if tiles:
        blank = np.zeros_like(tiles[0])
        pages = []
        for page_start in range(0, len(tiles), 20):
            page_tiles = tiles[page_start : page_start + 20]
            page_tiles += [blank] * (20 - len(page_tiles))
            page = np.vstack([np.hstack(page_tiles[row : row + 5]) for row in range(0, 20, 5)])
            page_number = page_start // 20 + 1
            cv2.imwrite(str(OUTPUT / f"contact-{page_number:02d}.png"), page)
            pages.append(f"contact-{page_number:02d}.png")
    else:
        pages = []
    report = {
        "description": "Offline missing-class SIFT proposals; every proposal requires visual/API verification.",
        "classes": sorted(CLASSES),
        "frames_reviewed": frames,
        "full_frame_review": [5, 12],
        "later_review_region_source_xyxy": [0, 0, WIDTH, 700],
        "template_count": len(bank),
        "proposal_count": len(proposals),
        "top_candidates": top,
        "timing": timing,
        "contact_pages": pages,
        "live_queries": 0,
        "limitations": [
            "Cross-scene local-feature matches are noisy and are candidates, not annotations.",
            "Sampling every fourth later frame assumes top-entry residence long enough to intersect at least one reviewed frame.",
        ],
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"proposals": len(proposals), "top": len(top), "pages": pages}, indent=2))


if __name__ == "__main__":
    main()
