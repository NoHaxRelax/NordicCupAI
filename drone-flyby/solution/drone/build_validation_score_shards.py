#!/usr/bin/env python3
"""Build three class-disjoint validation plans whose scores add to full mAP."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SIZE = (3840.0, 2160.0)
OFFICIAL_CLASSES = (
    "hangar",
    "helicopter",
    "jet_plane",
    "large_launcher",
    "large_tower",
    "medium_launcher",
    "medium_plane",
    "mine_roller",
    "small_launcher",
    "small_plane",
    "small_tower",
    "ta-ta",
    "tank",
    "condor",
    "jammer",
    "spacecraft",
)
CLASS_SHARDS = {
    "a": (
        "condor",
        "jet_plane",
        "large_launcher",
        "small_tower",
        "jammer",
        "spacecraft",
    ),
    "b": (
        "hangar",
        "large_tower",
        "medium_plane",
        "tank",
        "small_plane",
    ),
    "c": (
        "helicopter",
        "mine_roller",
        "small_launcher",
        "medium_launcher",
        "ta-ta",
    ),
}
CONFIDENCE_BY_PROVENANCE = {
    "score_confirmed_match": 1.0,
    "score_anchored_projective": 0.98,
    "reviewed_positive": 0.90,
    "algorithmic_shared_motion": 0.80,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_box(box: list[float]) -> list[float]:
    if len(box) != 4:
        raise ValueError(f"Expected four box coordinates, got {box}")
    normalized = [
        float(box[0]) / SOURCE_SIZE[0],
        float(box[1]) / SOURCE_SIZE[1],
        float(box[2]) / SOURCE_SIZE[0],
        float(box[3]) / SOURCE_SIZE[1],
    ]
    x1, y1, x2, y2 = normalized
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"Invalid normalized box {normalized} from {box}")
    return normalized


def validate_partition() -> None:
    flattened = [label for labels in CLASS_SHARDS.values() for label in labels]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Class shards overlap")
    if set(flattened) != set(OFFICIAL_CLASSES):
        missing = sorted(set(OFFICIAL_CLASSES) - set(flattened))
        extra = sorted(set(flattened) - set(OFFICIAL_CLASSES))
        raise ValueError(f"Class partition mismatch: missing={missing}, extra={extra}")


def load_predictions(annotation_root: Path) -> tuple[dict[int, list[dict]], Counter]:
    by_frame: dict[int, list[dict]] = {frame: [] for frame in range(1, 250)}
    class_counts: Counter[str] = Counter()
    for path in sorted(annotation_root.glob("frame_*.json")):
        payload = json.loads(path.read_text())
        frame = int(payload["frame"])
        if frame not in by_frame:
            raise ValueError(f"Frame outside validation range in {path}: {frame}")
        if int(payload["frame"]) != frame:
            raise ValueError(f"Frame mismatch in {path}")
        for row in payload["annotations"]:
            label = row["object_id"]
            if label not in OFFICIAL_CLASSES:
                raise ValueError(f"Unknown class {label!r} in {path}")
            provenance = row["provenance"]
            if provenance not in CONFIDENCE_BY_PROVENANCE:
                raise ValueError(f"Unknown provenance {provenance!r} in {path}")
            by_frame[frame].append(
                {
                    "object_id": label,
                    "bbox": normalized_box(row["bbox"]),
                    "confidence": CONFIDENCE_BY_PROVENANCE[provenance],
                }
            )
            class_counts[label] += 1
    return by_frame, class_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/drone/training/algorithmic-full-validation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/drone-api-tests/score-assessment-20260917",
    )
    parser.add_argument("--name-prefix", default="score-assessment-class")
    parser.add_argument("--name-suffix", default="20260917")
    parser.add_argument(
        "--shards",
        nargs="+",
        choices=tuple(CLASS_SHARDS),
        default=list(CLASS_SHARDS),
        help="Build only the selected class-disjoint shards (default: all three).",
    )
    args = parser.parse_args()

    validate_partition()
    metadata_root = args.dataset / "finetune-metadata"
    dataset_path = metadata_root / "dataset.json"
    report_path = args.dataset / "completion-report.json"
    predictions, class_counts = load_predictions(metadata_root / "annotations")

    expected_count = sum(class_counts.values())
    dataset = json.loads(dataset_path.read_text())
    if expected_count != int(dataset["annotation_count"]):
        raise ValueError(
            f"Per-frame rows ({expected_count}) disagree with dataset metadata "
            f"({dataset['annotation_count']})"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    plans = []
    union_count = 0
    selected_shards = tuple(dict.fromkeys(args.shards))
    for shard in selected_shards:
        classes = CLASS_SHARDS[shard]
        selected = {
            str(frame): [row for row in rows if row["object_id"] in classes]
            for frame, rows in predictions.items()
        }
        selected = {frame: rows for frame, rows in selected.items() if rows}
        count = sum(len(rows) for rows in selected.values())
        union_count += count
        name = f"{args.name_prefix}-{shard}-{args.name_suffix}"
        filename = f"class-shard-{shard}.json"
        plan = {
            "name": name,
            "target": [1920, 1080],
            "predictions_by_frame": selected,
            "score_assessment": {
                "shard": shard,
                "classes": list(classes),
                "frame_range_inclusive": [1, 249],
                "prediction_count": count,
                "confidence_by_provenance": CONFIDENCE_BY_PROVENANCE,
                "additivity_contract": (
                    "The three plans partition the official class list. Their returned "
                    "mAP scores add to the mAP of the prediction union because AP is "
                    "computed independently per represented class and macro-averaged."
                ),
            },
        }
        plan_path = args.output / filename
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        plans.append(
            {
                "shard": shard,
                "file": filename,
                "name": name,
                "classes": list(classes),
                "known_annotation_classes": [label for label in classes if class_counts[label]],
                "prediction_count": count,
                "frames_with_predictions": len(selected),
                "maximum_score_if_all_16_classes_are_present": len(classes) / 16.0,
                "sha256": sha256(plan_path),
            }
        )

    selected_classes = {
        label for shard in selected_shards for label in CLASS_SHARDS[shard]
    }
    selected_expected_count = sum(
        count for label, count in class_counts.items() if label in selected_classes
    )
    if union_count != selected_expected_count:
        raise ValueError(
            f"Shard union has {union_count} predictions, expected {selected_expected_count}"
        )

    manifest = {
        "description": (
            "Class-disjoint live-validation plans for an additive score assessment."
        ),
        "created_date": dt.date.today().isoformat(),
        "live_runs_queued": 0,
        "official_score_range": [0.0, 1.0],
        "official_score": "COCO mAP at IoU 0.50, macro-averaged over represented classes.",
        "why_not_frame_thirds": (
            "A validation attempt always scores all 249 frames. Missing two thirds reduce "
            "recall, and COCO AP across confidence-ranked detections is not additive by time."
        ),
        "class_partition": {shard: CLASS_SHARDS[shard] for shard in selected_shards},
        "plans": plans,
        "source": {
            "dataset": str(args.dataset),
            "dataset_metadata_sha256": sha256(dataset_path),
            "completion_report_sha256": sha256(report_path),
            "annotation_count": expected_count,
            "class_counts": dict(sorted(class_counts.items())),
            "official_classes_without_current_predictions": [
                label for label in OFFICIAL_CLASSES if not class_counts[label]
            ],
        },
        "execution_gate": (
            "Prepared only. Starting a public endpoint or queuing any validation attempt "
            "requires an explicit live-run instruction."
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"plans": plans, "union_predictions": union_count}, indent=2))


if __name__ == "__main__":
    main()
