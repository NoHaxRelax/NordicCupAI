#!/usr/bin/env python3
"""Audit structural and metadata integrity of a drone validation training set."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


WIDTH, HEIGHT = 3840, 2160
DENOMINATOR = np.asarray([WIDTH, HEIGHT, WIDTH, HEIGHT], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    rows: list[dict] = []
    canonical_track_rows: Counter[tuple] = Counter()
    keys: set[tuple[int, str]] = set()
    for path in sorted(args.dataset.glob("*.json")):
        if path.name in {"completion-report.json", "score-anchor-repair-report.json"}:
            continue
        payload = json.loads(path.read_text())
        for row in payload.get("annotations", []):
            frame = int(row["frame"])
            track = row.get("track_id", path.stem)
            key = (frame, track)
            if key in keys:
                errors.append(f"duplicate frame/track: {key}")
            keys.add(key)
            box = np.asarray(row["bbox_source_xyxy"], dtype=float)
            if box.shape != (4,) or not np.all(np.isfinite(box)):
                errors.append(f"invalid box vector: {path.name} frame {frame}")
                continue
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= WIDTH and 0 <= y1 < y2 <= HEIGHT):
                errors.append(f"out-of-bounds box: {path.name} frame {frame} {box.tolist()}")
            normalized = np.asarray(row["bbox_normalized_xyxy"], dtype=float)
            if normalized.shape != (4,) or not np.allclose(normalized, box / DENOMINATOR, atol=1e-7):
                errors.append(f"stale normalized box: {path.name} frame {frame}")
            if row.get("provenance") == "score_confirmed_match":
                old = row.get("pre_score_anchor_repair_bbox_source_xyxy")
                if old is None or row.get("score_anchor_repair_status") != "replaced_from_score_anchor":
                    errors.append(f"incomplete direct-anchor provenance: {path.name} frame {frame}")
            rows.append(row)
            canonical_track_rows[(
                frame,
                row["class"],
                track,
                tuple(round(float(value), 6) for value in row["bbox_source_xyxy"]),
                row.get("provenance", "reviewed_positive"),
            )] += 1

    metadata_path = args.dataset / "finetune-metadata/dataset.json"
    metadata = json.loads(metadata_path.read_text())
    class_counts = Counter(row["class"] for row in rows)
    provenance_counts = Counter(row.get("provenance", "reviewed_positive") for row in rows)
    if len(rows) != int(metadata["annotation_count"]):
        errors.append(f"metadata annotation count {metadata['annotation_count']} != {len(rows)}")
    if dict(sorted(class_counts.items())) != metadata["class_counts"]:
        errors.append("metadata class counts differ")
    if dict(sorted(provenance_counts.items())) != metadata["provenance_counts"]:
        errors.append("metadata provenance counts differ")

    metadata_rows = 0
    canonical_metadata_rows: Counter[tuple] = Counter()
    for frame_record in metadata["frames"]:
        frame_payload = json.loads(
            (args.dataset / "finetune-metadata" / frame_record["annotation_file"]).read_text()
        )
        count = len(frame_payload["annotations"])
        metadata_rows += count
        if count != int(frame_record["annotations"]):
            errors.append(f"per-frame count differs: frame {frame_record['frame']}")
        for row in frame_payload["annotations"]:
            canonical_metadata_rows[(
                int(frame_payload["frame"]),
                row["object_id"],
                row.get("track_id"),
                tuple(round(float(value), 6) for value in row["bbox"]),
                row.get("provenance", "reviewed_positive"),
            )] += 1
    if metadata_rows != len(rows):
        errors.append(f"per-frame metadata total {metadata_rows} != {len(rows)}")
    missing_from_metadata = canonical_track_rows - canonical_metadata_rows
    extra_in_metadata = canonical_metadata_rows - canonical_track_rows
    if missing_from_metadata:
        errors.append(f"top-level rows missing or stale in per-frame metadata: {sum(missing_from_metadata.values())}")
    if extra_in_metadata:
        errors.append(f"per-frame rows missing or stale in top-level tracks: {sum(extra_in_metadata.values())}")

    report_path = args.dataset / "score-anchor-repair-report.json"
    direct_anchors = 0
    if report_path.exists():
        report = json.loads(report_path.read_text())
        direct_anchors = sum(row.get("provenance") == "score_confirmed_match" for row in rows)
        expected = int(report["summary"]["score_confirmed_anchors"])
        if direct_anchors != expected:
            errors.append(f"direct anchor count {direct_anchors} != {expected}")
        for comparison in report["anchor_comparisons"]:
            if float(comparison["iou_score_anchor_vs_repaired"]) != 1.0:
                errors.append(f"non-exact repaired anchor: {comparison['candidate_id']}")

    result = {
        "dataset": str(args.dataset),
        "annotations": len(rows),
        "tracks": len({row.get("track_id") for row in rows}),
        "classes": dict(sorted(class_counts.items())),
        "direct_score_anchors": direct_anchors,
        "exact_top_level_metadata_match": not missing_from_metadata and not extra_in_metadata,
        "errors": errors,
        "passed": not errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
