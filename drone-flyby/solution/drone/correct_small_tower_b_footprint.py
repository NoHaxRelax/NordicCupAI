#!/usr/bin/env python3
"""Expand the late small-tower boxes to include its red annex."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


WIDTH = 3840
HEIGHT = 2160
TRACK_FILES = (
    "small-tower-b-early-238-239.json",
    "small-tower-b-240-249.json",
)


def normalized(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        round(x1 / WIDTH, 8),
        round(y1 / HEIGHT, 8),
        round(x2 / WIDTH, 8),
        round(y2 / HEIGHT, 8),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    changes: list[dict] = []
    for name in TRACK_FILES:
        path = args.dataset / name
        payload = json.loads(path.read_text())
        for row in payload["annotations"]:
            before = list(row["bbox_source_xyxy"])
            x1, y1, _x2, y2 = before
            # Native-resolution review shows that the tracked green structure
            # and the red annex form one rigid small_tower sprite.  Their full
            # footprint is consistently 89 source pixels wide in frames
            # 240-249.  Frames 238-239 are clipped by the top image edge but
            # retain the same visible horizontal footprint.
            after = [x1, y1, x1 + 89, y2]
            row["bbox_source_xyxy"] = after
            row["bbox_normalized_xyxy"] = normalized(after)
            row["label_contract"] = (
                "Direct native-resolution review against the public reference; "
                "box encloses the green structure and its red annex as one "
                "small_tower sprite. Participant-derived, not organizer ground truth."
            )
            row["footprint_correction"] = (
                "Expanded right edge after all-class detector audit exposed that "
                "the prior CSRT box followed only the green subcomponent."
            )
            changes.append({"file": name, "frame": row["frame"], "before": before, "after": after})
        payload["footprint_correction"] = {
            "status": "directly_reviewed",
            "method": (
                "Native 4K inspection of every visible frame plus consistent "
                "green-to-red component geometry at frames 241, 245 and 249."
            ),
            "reason": (
                "The red annex is part of the public-reference small_tower sprite, "
                "not a separate ta-ta or jammer object."
            ),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")

    metadata_dir = args.dataset / "finetune-metadata" / "annotations"
    metadata_changes = 0
    if metadata_dir.is_dir():
        corrected = {
            (row["frame"], Path(row["file"]).stem): row["after"]
            for row in changes
        }
        for frame in range(238, 250):
            frame_path = metadata_dir / f"frame_{frame:06d}.json"
            frame_payload = json.loads(frame_path.read_text())
            for row in frame_payload["annotations"]:
                key = (frame, row.get("track_id"))
                if key in corrected:
                    row["bbox"] = corrected[key]
                    row["footprint_correction"] = "green structure plus red annex"
                    metadata_changes += 1
            frame_path.write_text(json.dumps(frame_payload, indent=2) + "\n")

    report = {
        "description": "Directly reviewed small_tower full-footprint correction.",
        "source_dimensions": [WIDTH, HEIGHT],
        "changed_boxes": len(changes),
        "finetune_metadata_boxes_changed": metadata_changes,
        "changes": changes,
        "evidence": [
            "artifacts/drone-validation-coverage/late-small-object-contact.png",
            "artifacts/drone-validation-coverage/small-tower-entry-contact.png",
            "artifacts/drone-validation-coverage/frame238-small-tower-entry.png",
            "artifacts/drone-validation-coverage/ground-reference-review.png",
            "artifacts/drone-transfer-review/reference-tower-labels.png",
        ],
        "limitation": "Participant-derived validation pseudo-labels, not organizer ground truth.",
    }
    (args.dataset / "small-tower-footprint-correction-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({"changed_boxes": len(changes), "finetune_metadata_boxes_changed": metadata_changes, "dataset": str(args.dataset)}))


if __name__ == "__main__":
    main()
