#!/usr/bin/env python3
"""Apply a validation-supported centred bbox scale to one annotation class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


WIDTH = 3840
HEIGHT = 2160


def scaled(box: list[float], factor: float) -> list[float]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    width, height = (x2 - x1) * factor, (y2 - y1) * factor
    return [
        round(max(0.0, cx - width / 2), 2),
        round(max(0.0, cy - height / 2), 2),
        round(min(WIDTH, cx + width / 2), 2),
        round(min(HEIGHT, cy + height / 2), 2),
    ]


def normalized(box: list[float]) -> list[float]:
    return [
        round(box[0] / WIDTH, 8),
        round(box[1] / HEIGHT, 8),
        round(box[2] / WIDTH, 8),
        round(box[3] / HEIGHT, 8),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--factor", type=float, required=True)
    parser.add_argument("--validation-score-before", type=float, required=True)
    parser.add_argument("--validation-score-after", type=float, required=True)
    args = parser.parse_args()

    if args.factor <= 0:
        raise ValueError("--factor must be positive")
    changes = []
    for path in sorted(args.dataset.glob("*.json")):
        payload = json.loads(path.read_text())
        rows = payload.get("annotations")
        if not isinstance(rows, list):
            continue
        changed = False
        for row in rows:
            if row.get("class") != args.class_name:
                continue
            before = list(row["bbox_source_xyxy"])
            after = scaled(before, args.factor)
            row["bbox_source_xyxy"] = after
            row["bbox_normalized_xyxy"] = normalized(after)
            row["bbox_geometry_correction"] = {
                "method": "centred_scale",
                "factor": args.factor,
                "evidence": "validation_only_class_geometry_probe",
            }
            changes.append(
                {
                    "file": path.name,
                    "track_id": row.get("track_id"),
                    "frame": row["frame"],
                    "before": before,
                    "after": after,
                }
            )
            changed = True
        if changed:
            payload.setdefault("bbox_geometry_correction", {})[args.class_name] = {
                "method": "centred_scale",
                "factor": args.factor,
                "validation_score_before": args.validation_score_before,
                "validation_score_after": args.validation_score_after,
            }
            path.write_text(json.dumps(payload, indent=2) + "\n")

    metadata_dir = args.dataset / "finetune-metadata" / "annotations"
    metadata_changes = 0
    for path in sorted(metadata_dir.glob("frame_*.json")):
        payload = json.loads(path.read_text())
        changed = False
        for row in payload.get("annotations", []):
            if row.get("object_id") != args.class_name:
                continue
            row["bbox"] = scaled(row["bbox"], args.factor)
            row["bbox_geometry_correction"] = {
                "method": "centred_scale",
                "factor": args.factor,
                "evidence": "validation_only_class_geometry_probe",
            }
            metadata_changes += 1
            changed = True
        if changed:
            path.write_text(json.dumps(payload, indent=2) + "\n")

    if metadata_changes != len(changes):
        raise ValueError(f"Track changes {len(changes)} != metadata changes {metadata_changes}")
    report = {
        "description": "Validation-supported class-wide bbox geometry correction.",
        "class": args.class_name,
        "method": "centred_scale",
        "factor": args.factor,
        "changed_boxes": len(changes),
        "validation_score_before": args.validation_score_before,
        "validation_score_after": args.validation_score_after,
        "validation_score_delta": args.validation_score_after - args.validation_score_before,
        "evaluations_used": 0,
        "changes": changes,
        "limitations": [
            "Participant-created validation pseudo-labels, not organizer ground truth.",
            "The selected scale is supported by a class-isolated validation probe at IoU 0.50; it does not reveal exact hidden boxes.",
        ],
    }
    report_path = args.dataset / f"{args.class_name}-bbox-geometry-correction-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    dataset_path = args.dataset / "finetune-metadata" / "dataset.json"
    dataset = json.loads(dataset_path.read_text())
    dataset.setdefault("bbox_geometry_corrections", []).append(
        {
            "class": args.class_name,
            "factor": args.factor,
            "changed_boxes": len(changes),
            "validation_score_before": args.validation_score_before,
            "validation_score_after": args.validation_score_after,
            "report": report_path.name,
        }
    )
    dataset_path.write_text(json.dumps(dataset, indent=2) + "\n")
    print(json.dumps({"dataset": str(args.dataset), "changed_boxes": len(changes), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
