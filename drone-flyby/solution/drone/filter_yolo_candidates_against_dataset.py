#!/usr/bin/env python3
"""Separate detector proposals that overlap existing validation annotations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def iou(a: list[float], b: list[float]) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def contains(box: list[float], point: tuple[float, float], margin: float = 0.0) -> bool:
    width = box[2] - box[0]
    height = box[3] - box[1]
    return (
        box[0] - width * margin <= point[0] <= box[2] + width * margin
        and box[1] - height * margin <= point[1] <= box[3] + height * margin
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--frame-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--centre-margin", type=float, default=0.2)
    parser.add_argument("--minimum-iou", type=float, default=0.1)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    known_by_frame: dict[int, list[dict]] = {}
    for path in sorted(args.frame_annotations.glob("frame_*.json")):
        payload = json.loads(path.read_text())
        known_by_frame[int(payload["frame"])] = payload.get("annotations", [])

    matched: list[dict] = []
    unmatched: list[dict] = []
    for candidate in report["predictions"]:
        box = candidate["bbox_source_xyxy"]
        centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        comparisons = []
        for known in known_by_frame.get(int(candidate["frame"]), []):
            known_box = known["bbox"]
            overlap = iou(box, known_box)
            known_centre = (
                (known_box[0] + known_box[2]) / 2,
                (known_box[1] + known_box[3]) / 2,
            )
            comparisons.append(
                {
                    "known_class": known["object_id"],
                    "known_track_id": known.get("track_id"),
                    "known_bbox_source_xyxy": known_box,
                    "iou": overlap,
                    "candidate_centre_in_known": contains(known_box, centre, args.centre_margin),
                    "known_centre_in_candidate": contains(box, known_centre),
                }
            )
        comparisons.sort(
            key=lambda row: (
                row["candidate_centre_in_known"],
                row["known_centre_in_candidate"],
                row["iou"],
            ),
            reverse=True,
        )
        best = comparisons[0] if comparisons else None
        row = dict(candidate)
        row["best_existing_match"] = best
        is_match = bool(
            best
            and (
                best["candidate_centre_in_known"]
                or best["known_centre_in_candidate"]
                or best["iou"] >= args.minimum_iou
            )
        )
        (matched if is_match else unmatched).append(row)

    def class_counts(rows: list[dict]) -> dict[str, int]:
        return {
            label: sum(row["class"] == label for row in rows)
            for label in sorted({row["class"] for row in rows})
        }

    output = {
        "description": "Detector proposals split by spatial agreement with the current validation dataset. Unmatched proposals remain candidates, not annotations.",
        "source_report": str(args.report),
        "settings": {
            "centre_margin": args.centre_margin,
            "minimum_iou": args.minimum_iou,
        },
        "matched_count": len(matched),
        "unmatched_count": len(unmatched),
        "matched_class_counts": class_counts(matched),
        "unmatched_class_counts": class_counts(unmatched),
        "matched_predictions": matched,
        "predictions": unmatched,
        "class_counts": class_counts(unmatched),
        "limitations": [
            "Spatial matching can miss fragment detections next to a known box.",
            "Unmatched proposals require native-resolution visual review and are not accepted automatically."
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "matched": len(matched),
        "unmatched": len(unmatched),
        "unmatched_class_counts": output["unmatched_class_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
