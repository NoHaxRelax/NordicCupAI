#!/usr/bin/env python3
"""Relabel one completed validation track without discarding prior corrections."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument("--to-class", required=True)
    parser.add_argument("--output-track", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--validation-score", type=float, required=True)
    args = parser.parse_args()

    source_path = args.dataset / f"{args.track}.json"
    target_path = args.dataset / f"{args.output_track}.json"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if target_path.exists():
        raise FileExistsError(target_path)
    payload = json.loads(source_path.read_text())
    rows = payload["annotations"]
    old_classes = {row["class"] for row in rows}
    if len(old_classes) != 1:
        raise ValueError(f"Expected one source class, got {sorted(old_classes)}")
    old_class = next(iter(old_classes))
    frames = set()
    for row in rows:
        frames.add(int(row["frame"]))
        row["pre_class_correction"] = old_class
        row["class"] = args.to_class
        row["track_id"] = args.output_track
        row["class_correction_evidence"] = args.evidence
        row["class_correction_validation_score"] = args.validation_score
    payload["track_id"] = args.output_track
    payload["class"] = args.to_class
    payload["class_correction"] = {
        "from": old_class,
        "to": args.to_class,
        "evidence": args.evidence,
        "isolated_validation_score": args.validation_score,
    }
    target_path.write_text(json.dumps(payload, indent=2) + "\n")
    source_path.unlink()

    metadata_root = args.dataset / "finetune-metadata"
    changed_metadata = 0
    for frame in sorted(frames):
        path = metadata_root / "annotations" / f"frame_{frame:06d}.json"
        item = json.loads(path.read_text())
        for row in item["annotations"]:
            if row.get("track_id") == args.track:
                row["pre_class_correction"] = row["object_id"]
                row["object_id"] = args.to_class
                row["track_id"] = args.output_track
                row["class_correction_evidence"] = args.evidence
                row["class_correction_validation_score"] = args.validation_score
                changed_metadata += 1
        item["object_counts"] = dict(sorted(Counter(row["object_id"] for row in item["annotations"]).items()))
        path.write_text(json.dumps(item, indent=2) + "\n")
    if changed_metadata != len(rows):
        raise ValueError(f"Changed {changed_metadata} metadata rows for {len(rows)} track rows")

    dataset_path = metadata_root / "dataset.json"
    dataset = json.loads(dataset_path.read_text())
    counts = Counter(dataset["class_counts"])
    counts[old_class] -= len(rows)
    counts[args.to_class] += len(rows)
    dataset["class_counts"] = {key: value for key, value in sorted(counts.items()) if value}
    dataset.setdefault("class_correction_history", []).append({
        "source_track": args.track,
        "output_track": args.output_track,
        "from": old_class,
        "to": args.to_class,
        "annotations": len(rows),
        "isolated_validation_score": args.validation_score,
    })
    dataset_path.write_text(json.dumps(dataset, indent=2) + "\n")

    report = {
        "description": "Visually and validation-score-supported completed-track relabel.",
        "source_track": args.track,
        "output_track": args.output_track,
        "from_class": old_class,
        "to_class": args.to_class,
        "corrected_annotations": len(rows),
        "frame_range": [min(frames), max(frames)],
        "evidence": args.evidence,
        "isolated_validation_score": args.validation_score,
        "evaluations_used": 0,
        "limitations": [
            "Participant-created validation pseudo-labels, not organizer ground truth.",
            "A positive isolated score supports class and overlap, not exact hidden box coordinates."
        ],
    }
    report_name = f"{args.track}-to-{args.to_class}-correction-report.json"
    (args.dataset / report_name).write_text(json.dumps(report, indent=2) + "\n")
    completion_path = args.dataset / "completion-report.json"
    completion = json.loads(completion_path.read_text())
    completion.setdefault("class_correction_history", []).append(report | {"report": report_name})
    completion_path.write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
