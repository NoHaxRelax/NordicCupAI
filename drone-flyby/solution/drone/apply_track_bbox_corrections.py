#!/usr/bin/env python3
"""Apply auditable, track-specific native-pixel bbox corrections.

The correction spec supports explicit per-frame boxes and a conservative dark
sprite component extractor.  Every changed row is mirrored into the per-frame
fine-tuning metadata, and removals update dataset-level counts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np


WIDTH, HEIGHT = 3840, 2160


def normalized(box: list[float]) -> list[float]:
    return [
        round(box[0] / WIDTH, 8),
        round(box[1] / HEIGHT, 8),
        round(box[2] / WIDTH, 8),
        round(box[3] / HEIGHT, 8),
    ]


def dark_component_box(image: np.ndarray, old_box: list[float], rule: dict) -> list[float]:
    left, top, right, bottom = rule.get("search_expand", [0, 0, 0, 0])
    x1 = max(0, int(np.floor(old_box[0] - left)))
    y1 = max(0, int(np.floor(old_box[1] - top)))
    x2 = min(WIDTH, int(np.ceil(old_box[2] + right)))
    y2 = min(HEIGHT, int(np.ceil(old_box[3] + bottom)))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"Empty search crop for {old_box}")

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    raw = (gray < int(rule["gray_threshold"])).astype(np.uint8)
    kw, kh = rule.get("close_kernel", [1, 1])
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((kh, kw), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    old_cx = (old_box[0] + old_box[2]) / 2
    old_cy = (old_box[1] + old_box[3]) / 2
    candidates: list[tuple[float, int]] = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if int(area) < int(rule.get("min_area", 20)):
            continue
        cx = x1 + x + width / 2
        cy = y1 + y + height / 2
        score = abs(cx - old_cx) + float(rule.get("vertical_distance_weight", 0.2)) * abs(cy - old_cy)
        candidates.append((score, index))
    if not candidates:
        raise ValueError(f"No component for box {old_box}")

    _, selected = min(candidates)
    ys, xs = np.where((labels == selected) & (raw > 0))
    if not len(xs):
        raise ValueError(f"Selected component has no original pixels for {old_box}")
    pad_left, pad_top, pad_right, pad_bottom = rule.get("padding", [0, 0, 0, 0])
    box = [
        max(0.0, x1 + float(xs.min()) - pad_left),
        max(0.0, y1 + float(ys.min()) - pad_top),
        min(float(WIDTH), x1 + float(xs.max() + 1) + pad_right),
        min(float(HEIGHT), y1 + float(ys.max() + 1) + pad_bottom),
    ]
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError(f"Degenerate corrected box {box}")
    return [round(value, 2) for value in box]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    rules = {row["track_id"]: row for row in spec["corrections"]}
    changes: list[dict] = []
    removals: list[dict] = []
    corrected: dict[tuple[int, str], list[float]] = {}
    removed: set[tuple[int, str]] = set()

    for path in sorted(args.dataset.glob("*.json")):
        payload = json.loads(path.read_text())
        rows = payload.get("annotations")
        if not isinstance(rows, list):
            continue
        changed_file = False
        retained = []
        for row in rows:
            track_id = row.get("track_id", payload.get("track_id", path.stem))
            rule = rules.get(track_id)
            if rule is None:
                retained.append(row)
                continue
            frame = int(row["frame"])
            key = (frame, track_id)
            if frame in set(rule.get("remove_frames", [])):
                removals.append({"file": path.name, "track_id": track_id, "frame": frame, "before": row["bbox_source_xyxy"]})
                removed.add(key)
                changed_file = True
                continue

            before = list(row["bbox_source_xyxy"])
            explicit = rule.get("explicit_frames", {}).get(str(frame))
            if explicit is not None:
                after = [float(value) for value in explicit]
                method = "explicit_native_pixel_review"
            elif rule["method"] == "dark_component":
                image = cv2.imread(str(args.images / f"frame_{frame:06d}.png"), cv2.IMREAD_COLOR)
                if image is None:
                    raise SystemExit(f"Missing source frame {frame}")
                after = dark_component_box(image, before, rule)
                method = "native_pixel_dark_component"
            elif rule["method"] == "explicit":
                raise ValueError(f"Missing explicit box for {track_id} frame {frame}")
            else:
                raise ValueError(f"Unknown method {rule['method']}")

            row["bbox_source_xyxy"] = after
            row["bbox_normalized_xyxy"] = normalized(after)
            row["bbox_geometry_correction"] = {
                "method": method,
                "evidence": rule["evidence"],
                "correction_set": spec["correction_set"],
            }
            corrected[key] = after
            changes.append({"file": path.name, "track_id": track_id, "frame": frame, "before": before, "after": after, "method": method})
            retained.append(row)
            changed_file = True
        if changed_file:
            payload["annotations"] = retained
            payload.setdefault("track_bbox_corrections", []).append({
                "correction_set": spec["correction_set"],
                "spec": args.spec.name,
                "status": "native_4k_reviewed_pending_validation_confirmation",
            })
            path.write_text(json.dumps(payload, indent=2) + "\n")

    metadata_dir = args.dataset / "finetune-metadata" / "annotations"
    mirrored_changes = 0
    mirrored_removals = 0
    for path in sorted(metadata_dir.glob("frame_*.json")):
        payload = json.loads(path.read_text())
        retained = []
        changed_file = False
        for row in payload.get("annotations", []):
            key = (int(payload["frame"]), row.get("track_id"))
            if key in removed:
                mirrored_removals += 1
                changed_file = True
                continue
            if key in corrected:
                row["bbox"] = corrected[key]
                row["bbox_geometry_correction"] = {
                    "method": next(change["method"] for change in changes if (change["frame"], change["track_id"]) == key),
                    "correction_set": spec["correction_set"],
                }
                mirrored_changes += 1
                changed_file = True
            retained.append(row)
        if changed_file:
            payload["annotations"] = retained
            payload["object_counts"] = dict(sorted(Counter(row["object_id"] for row in retained).items()))
            payload["provenance_counts"] = dict(sorted(Counter(row.get("provenance", "reviewed_positive") for row in retained).items()))
            path.write_text(json.dumps(payload, indent=2) + "\n")

    if mirrored_changes != len(changes) or mirrored_removals != len(removals):
        raise ValueError(
            f"Track/metadata mismatch: {len(changes)=} {mirrored_changes=} "
            f"{len(removals)=} {mirrored_removals=}"
        )

    dataset_path = args.dataset / "finetune-metadata" / "dataset.json"
    dataset = json.loads(dataset_path.read_text())
    all_rows = []
    frame_counts: dict[int, int] = {}
    for path in sorted(args.dataset.glob("*.json")):
        payload = json.loads(path.read_text())
        if isinstance(payload.get("annotations"), list):
            all_rows.extend(payload["annotations"])
    for path in sorted(metadata_dir.glob("frame_*.json")):
        payload = json.loads(path.read_text())
        frame_counts[int(payload["frame"])] = len(payload["annotations"])
    for frame_record in dataset["frames"]:
        frame_record["annotations"] = frame_counts[int(frame_record["frame"])]
    dataset["annotation_count"] = len(all_rows)
    dataset["class_counts"] = dict(sorted(Counter(row["class"] for row in all_rows).items()))
    dataset["provenance_counts"] = dict(sorted(Counter(row.get("provenance", "reviewed_positive") for row in all_rows).items()))
    dataset.setdefault("track_bbox_corrections", []).append({
        "correction_set": spec["correction_set"],
        "changed_boxes": len(changes),
        "removed_stale_boxes": len(removals),
        "spec": args.spec.name,
        "report": args.report.name,
        "status": "native_4k_reviewed_pending_validation_confirmation",
    })
    dataset_path.write_text(json.dumps(dataset, indent=2) + "\n")

    report = {
        "description": spec["description"],
        "correction_set": spec["correction_set"],
        "source_dataset": spec["source_dataset"],
        "validation_status": "pending_validation_confirmation",
        "evaluations_used": 0,
        "changed_boxes": len(changes),
        "removed_stale_boxes": len(removals),
        "changes": changes,
        "removals": removals,
        "limitations": spec["limitations"],
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"changed_boxes": len(changes), "removed_stale_boxes": len(removals), "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
