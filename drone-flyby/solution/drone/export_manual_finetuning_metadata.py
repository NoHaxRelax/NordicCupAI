#!/usr/bin/env python3
"""Export manual validation pseudo-labels in the reference per-frame schema."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--coverage-ledger",
        type=Path,
        help="Optional completed spatial-review ledger to embed as a training-scope contract.",
    )
    args = parser.parse_args()

    coverage = None
    if args.coverage_ledger:
        coverage = json.loads(args.coverage_ledger.read_text())

    by_frame: dict[int, list[dict]] = defaultdict(list)
    for path in sorted(args.annotations.glob("*.json")):
        payload = json.loads(path.read_text())
        for row in payload.get("annotations", []):
            by_frame[int(row["frame"])].append({
                "object_id": row["class"],
                "bbox": row["bbox_source_xyxy"],
                "provenance": "manual_reference_match",
                "review_status": row.get("review_status", "direct_reviewed"),
            })

    annotation_dir = args.output / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    dataset_rows = []
    class_counts: Counter[str] = Counter()
    for frame in sorted(by_frame):
        annotations = by_frame[frame]
        object_counts = Counter(row["object_id"] for row in annotations)
        class_counts.update(object_counts)
        filename = f"frame_{frame:06d}.json"
        payload = {
            "frame": frame,
            "image_file": f"../../../reconstructed-validation/frame_{frame:06d}.png",
            "image_dimensions": [3840, 2160],
            "annotations": annotations,
            "object_counts": dict(sorted(object_counts.items())),
            "annotation_source": "manual visual review plus reviewed CSRT tracking",
        }
        if coverage:
            first_frame = int(coverage["sampling"]["first_frame"])
            top_band_height = int(coverage["sampling"]["top_band_height"])
            payload["reviewed_negative_regions_xyxy"] = (
                [[0, 0, 3840, 2160]]
                if frame == first_frame
                else [[0, 0, 3840, top_band_height]]
            )
        (annotation_dir / filename).write_text(json.dumps(payload, indent=2) + "\n")
        dataset_rows.append({"frame": frame, "annotation_file": f"annotations/{filename}", "annotations": len(annotations)})

    metadata = {
        "dataset_name": "Nordic AI Cup 2026 manual validation pseudo-labels",
        "schema": "Matches the reference per-frame object_id/bbox annotation shape; manual provenance fields are additive.",
        "image_root": "data/drone/reconstructed-validation",
        "image_dimensions": [3840, 2160],
        "annotated_frame_count": len(dataset_rows),
        "annotation_count": sum(class_counts.values()),
        "class_counts": dict(sorted(class_counts.items())),
        "frames": dataset_rows,
        "limitations": [
            "Participant-created manual pseudo-labels, not organizer ground truth.",
            "The coverage gate certifies only the reviewed regions under the user-supplied top-entry invariant.",
            "Frame 5 is reviewed over the full image; later frames are reviewed only in the top 540 px.",
            "Do not use the lower part of later full frames as negative detector background.",
            "Frames 1-4 are incomplete reconstructions and are excluded from the certified review scope.",
        ],
    }
    if coverage:
        sampling = coverage["sampling"]
        metadata["coverage_review"] = {
            "ledger": str(args.coverage_ledger),
            "completion": coverage["completion"],
            "review_contract": coverage["review_contract"],
            "sampling": {
                "first_frame": sampling["first_frame"],
                "last_frame": sampling["last_frame"],
                "stride": sampling["stride"],
                "top_band_height": sampling["top_band_height"],
                "motion_gate_px": sampling["motion_gate_px"],
                "max_measured_sample_displacement_y_px": sampling[
                    "max_measured_sample_displacement_y_px"
                ],
                "motion_gate_passed": sampling["motion_gate_passed"],
            },
        }
    (args.output / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"frames": len(dataset_rows), "annotations": sum(class_counts.values())}))


if __name__ == "__main__":
    main()
