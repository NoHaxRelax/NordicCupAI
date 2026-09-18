#!/usr/bin/env python3
"""Clone a validation set and apply one validation-score-confirmed track class correction."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil


SOURCE_SIZE = [3840, 2160]


def rebuild_metadata(output: Path, report: dict) -> dict:
    by_frame = defaultdict(list)
    class_counts: Counter[str] = Counter()
    provenance_counts: Counter[str] = Counter()
    for path in sorted(output.glob("*.json")):
        if path.name in {"completion-report.json", "score-anchor-repair-report.json", "class-correction-report.json"}:
            continue
        payload = json.loads(path.read_text())
        for row in payload.get("annotations", []):
            item = {
                "object_id": row["class"],
                "bbox": row["bbox_source_xyxy"],
                "track_id": row.get("track_id", path.stem),
                "provenance": row.get("provenance", "reviewed_positive"),
                "evidence": row.get("evidence"),
                "review_status": row.get("review_status"),
                "anchor_frame": row.get("score_anchor_frame", row.get("anchor_frame")),
            }
            by_frame[int(row["frame"])].append(item)
            class_counts[item["object_id"]] += 1
            provenance_counts[item["provenance"]] += 1

    metadata_root = output / "finetune-metadata"
    annotation_root = metadata_root / "annotations"
    shutil.rmtree(annotation_root)
    annotation_root.mkdir(parents=True)
    frames = []
    for frame in range(5, 250):
        annotations = by_frame.get(frame, [])
        payload = {
            "frame": frame,
            "image_file": f"../../../reconstructed-validation/frame_{frame:06d}.png",
            "image_dimensions": SOURCE_SIZE,
            "annotations": annotations,
            "object_counts": dict(sorted(Counter(row["object_id"] for row in annotations).items())),
            "provenance_counts": dict(sorted(Counter(row["provenance"] for row in annotations).items())),
            "coverage_status": "score_anchor_repaired_and_class_corrected; discovery_not_certified",
        }
        filename = f"frame_{frame:06d}.json"
        (annotation_root / filename).write_text(json.dumps(payload, indent=2) + "\n")
        frames.append({"frame": frame, "annotation_file": f"annotations/{filename}", "annotations": len(annotations)})

    dataset = json.loads((metadata_root / "dataset.json").read_text())
    dataset.update({
        "dataset_name": "Nordic AI Cup 2026 score-anchored, class-corrected validation pseudo-labels v3",
        "annotation_count": sum(class_counts.values()),
        "class_counts": dict(sorted(class_counts.items())),
        "provenance_counts": dict(sorted(provenance_counts.items())),
        "frames": frames,
        "class_correction_report": "../class-correction-report.json",
        "class_correction_summary": report["summary"],
    })
    (metadata_root / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track", required=True, help="Source track filename stem.")
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--validation-score", type=float, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    shutil.copytree(args.source, args.output)

    source_path = args.output / f"{args.track}.json"
    payload = json.loads(source_path.read_text())
    rows = payload["annotations"]
    old_classes = {row["class"] for row in rows}
    if len(old_classes) != 1:
        raise ValueError(f"Track has multiple source classes: {old_classes}")
    old_class = next(iter(old_classes))
    new_track = args.track.replace(old_class.replace("_", "-"), args.class_name.replace("_", "-"), 1)
    for row in rows:
        row["pre_class_correction"] = row["class"]
        row["class"] = args.class_name
        row["track_id"] = new_track
        row["class_correction_evidence"] = args.evidence
        row["class_correction_validation_score"] = args.validation_score
    payload["track_id"] = new_track
    payload["class_correction"] = {
        "from": old_class,
        "to": args.class_name,
        "evidence": args.evidence,
        "validation_score": args.validation_score,
    }
    target_path = args.output / f"{new_track}.json"
    target_path.write_text(json.dumps(payload, indent=2) + "\n")
    source_path.unlink()

    report = {
        "description": "Participant track class correction supported by visual reference comparison and isolated validation score.",
        "summary": {
            "corrected_tracks": 1,
            "corrected_annotations": len(rows),
            "from_class": old_class,
            "to_class": args.class_name,
            "validation_score": args.validation_score,
        },
        "source_track": args.track,
        "output_track": new_track,
        "evidence": args.evidence,
        "limitations": ["Positive validation score confirms matching predictions for the class, not exact hidden box coordinates."],
    }
    dataset = rebuild_metadata(args.output, report)
    report["output_counts"] = {"annotations": dataset["annotation_count"], "classes": dataset["class_counts"]}
    (args.output / "class-correction-report.json").write_text(json.dumps(report, indent=2) + "\n")
    completion_path = args.output / "completion-report.json"
    completion = json.loads(completion_path.read_text())
    completion["class_correction"] = report["summary"] | {"report": str(args.output / "class-correction-report.json")}
    completion_path.write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(report["summary"] | report["output_counts"], indent=2))


if __name__ == "__main__":
    main()
