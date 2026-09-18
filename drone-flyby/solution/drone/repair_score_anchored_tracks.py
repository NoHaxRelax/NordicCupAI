#!/usr/bin/env python3
"""Build a non-destructive validation dataset repaired from live score anchors.

The score-confirmed boxes prove a correct class and IoU >= 0.50 for one frame.
They are not exact organizer ground truth.  This builder projects the nearest
confirmed anchor through the calibrated validation camera motion for every row
of the matched participant track and records both old and new geometry.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np

from complete_validation_bottom import SOURCE_SIZE, clip_box, iou, load_clock, visible
from perspective_tracking.motion import MotionModel


ROOT = Path(__file__).resolve().parents[1]
NORMALIZER = np.asarray([*SOURCE_SIZE, *SOURCE_SIZE], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data/drone/training/algorithmic-full-validation",
    )
    parser.add_argument(
        "--candidate-union",
        type=Path,
        default=ROOT / "artifacts/drone-validation-coverage/candidate-union.json",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "artifacts/drone-conditioned-shape/validation-calibration.json",
    )
    parser.add_argument(
        "--clock",
        type=Path,
        default=ROOT / "artifacts/drone-camera-pixels/validation-clocks.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/drone/training/score-anchored-validation-v2",
    )
    return parser.parse_args()


def load_anchors(path: Path) -> dict[str, list[dict]]:
    candidates = json.loads(path.read_text())["candidates"]
    manual = {
        row["candidate_id"]: row
        for row in candidates
        if row.get("source") == "manual_track"
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        if row.get("source") != "score_confirmed_seed":
            continue
        matches = row.get("matched_manual_seed_ids", [])
        if len(matches) != 1 or matches[0] not in manual:
            raise ValueError(f"Score seed does not resolve to one manual track: {row}")
        matched = manual[matches[0]]
        track_id = Path(matched["source_file"]).stem
        if matched["class"] != row["class"] or int(matched["frame"]) != int(row["frame"]):
            raise ValueError(f"Score/manual candidate mismatch: {row['candidate_id']}")
        grouped[track_id].append(
            {
                "candidate_id": row["candidate_id"],
                "frame": int(row["frame"]),
                "class": row["class"],
                "bbox_source_xyxy": list(map(float, row["bbox_source_xyxy"])),
                "box_certainty": row["box_certainty"],
            }
        )
    return {track: sorted(rows, key=lambda row: row["frame"]) for track, rows in grouped.items()}


def export_metadata(output: Path, report: dict) -> dict:
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for path in sorted(output.glob("*.json")):
        if path.name in ("completion-report.json", "score-anchor-repair-report.json"):
            continue
        payload = json.loads(path.read_text())
        for row in payload.get("annotations", []):
            by_frame[int(row["frame"])].append(
                {
                    "object_id": row["class"],
                    "bbox": row["bbox_source_xyxy"],
                    "track_id": row.get("track_id", path.stem),
                    "provenance": row.get("provenance", "reviewed_positive"),
                    "evidence": row.get("evidence"),
                    "review_status": row.get("review_status"),
                    "anchor_frame": row.get("score_anchor_frame", row.get("anchor_frame")),
                }
            )

    metadata_root = output / "finetune-metadata"
    annotation_root = metadata_root / "annotations"
    if annotation_root.exists():
        shutil.rmtree(annotation_root)
    annotation_root.mkdir(parents=True)
    class_counts: Counter[str] = Counter()
    provenance_counts: Counter[str] = Counter()
    frames = []
    for frame in range(5, 250):
        annotations = by_frame.get(frame, [])
        class_counts.update(row["object_id"] for row in annotations)
        provenance_counts.update(row["provenance"] for row in annotations)
        payload = {
            "frame": frame,
            "image_file": f"../../../reconstructed-validation/frame_{frame:06d}.png",
            "image_dimensions": list(SOURCE_SIZE),
            "annotations": annotations,
            "object_counts": dict(sorted(Counter(row["object_id"] for row in annotations).items())),
            "provenance_counts": dict(
                sorted(Counter(row["provenance"] for row in annotations).items())
            ),
            "coverage_status": "score_anchor_repaired_known_tracks; discovery_not_certified",
        }
        filename = f"frame_{frame:06d}.json"
        (annotation_root / filename).write_text(json.dumps(payload, indent=2) + "\n")
        frames.append(
            {
                "frame": frame,
                "annotation_file": f"annotations/{filename}",
                "annotations": len(annotations),
            }
        )
    dataset = {
        "dataset_name": "Nordic AI Cup 2026 score-anchored validation pseudo-labels v2",
        "schema": "Reference-compatible object_id/bbox rows with explicit score-anchor provenance.",
        "image_root": "data/drone/reconstructed-validation",
        "image_dimensions": list(SOURCE_SIZE),
        "frame_range_inclusive": [5, 249],
        "frame_count": 245,
        "annotation_count": sum(class_counts.values()),
        "class_counts": dict(sorted(class_counts.items())),
        "provenance_counts": dict(sorted(provenance_counts.items())),
        "frames": frames,
        "repair_report": "../score-anchor-repair-report.json",
        "repair_summary": report["summary"],
        "limitations": [
            "Score-confirmed anchors prove correct class and IoU >= 0.50, not exact hidden boxes.",
            "Projected rows inherit the shared-camera-motion assumption and are not independently score-confirmed.",
            "Only already known tracks are repaired; exhaustive object discovery remains incomplete.",
            "Frames 1-4 remain excluded.",
        ],
    }
    (metadata_root / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    return dataset


def canonicalize_normalized_boxes(output: Path) -> int:
    """Make normalized coordinates an exact rounded transform of pixel boxes."""
    changed = 0
    for path in sorted(output.glob("*.json")):
        if path.name in ("completion-report.json", "score-anchor-repair-report.json"):
            continue
        payload = json.loads(path.read_text())
        rows = payload.get("annotations", [])
        dirty = False
        for row in rows:
            box = np.asarray(row["bbox_source_xyxy"], dtype=float)
            normalized = np.round(box / NORMALIZER, 8).tolist()
            if row.get("bbox_normalized_xyxy") != normalized:
                row["bbox_normalized_xyxy"] = normalized
                changed += 1
                dirty = True
        if dirty:
            path.write_text(json.dumps(payload, indent=2) + "\n")
    return changed


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")
    shutil.copytree(args.source, args.output)

    model = MotionModel.from_dict(json.loads(args.calibration.read_text()))
    clock, _ = load_clock(args.clock, 5, 249)
    anchors = load_anchors(args.candidate_union)
    comparisons = []
    track_summaries = []
    total_repaired = 0
    total_removed = 0
    for track_id, track_anchors in sorted(anchors.items()):
        track_path = args.output / f"{track_id}.json"
        if not track_path.exists():
            raise FileNotFoundError(f"Matched track is absent from completed dataset: {track_path}")
        payload = json.loads(track_path.read_text())
        rows = payload.get("annotations", [])
        if not rows:
            raise ValueError(f"Matched track has no retained annotations: {track_id}")
        labels = {row["class"] for row in rows}
        if labels != {track_anchors[0]["class"]}:
            raise ValueError(f"Track class mismatch for {track_id}: {labels}")

        direct_by_frame = {row["frame"]: row for row in track_anchors}
        old_by_frame = {int(row["frame"]): list(map(float, row["bbox_source_xyxy"])) for row in rows}
        repaired_rows = []
        removed_rows = 0
        for original in rows:
            row = deepcopy(original)
            frame = int(row["frame"])
            anchor = min(track_anchors, key=lambda item: abs(clock[frame] - clock[item["frame"]]))
            if frame == anchor["frame"]:
                unbounded = np.asarray(anchor["bbox_source_xyxy"], dtype=float)
                provenance = "score_confirmed_match"
                evidence = "isolated_validation_score_match_iou_at_least_0_50"
                review_status = "score_confirmed_anchor"
            else:
                unbounded = model.box(
                    np.asarray(anchor["bbox_source_xyxy"], dtype=float),
                    clock[anchor["frame"]],
                    clock[frame],
                )
                provenance = "score_anchored_projective"
                evidence = "nearest_score_confirmed_anchor_plus_validation_camera_motion"
                review_status = "score_anchored_track_projection"
            previous = np.asarray(row["bbox_source_xyxy"], dtype=float)
            if not visible(unbounded):
                removed_rows += 1
                continue
            repaired = clip_box(unbounded)
            row["pre_score_anchor_repair_bbox_source_xyxy"] = row["bbox_source_xyxy"]
            row["score_anchor_repair_status"] = "replaced_from_score_anchor"
            row["bbox_source_xyxy"] = np.round(repaired, 2).tolist()
            row["bbox_normalized_xyxy"] = np.round(repaired / NORMALIZER, 8).tolist()
            row["provenance"] = provenance
            row["evidence"] = evidence
            row["review_status"] = review_status
            row["score_anchor_frame"] = anchor["frame"]
            row["score_anchor_candidate_id"] = anchor["candidate_id"]
            row["score_anchor_box_certainty"] = anchor["box_certainty"]
            row["repair_iou_vs_previous"] = round(iou(previous, repaired), 6)
            repaired_rows.append(row)
        payload["annotations"] = repaired_rows
        payload["score_anchor_repair"] = {
            "status": "replaced_entire_known_track",
            "anchor_count": len(track_anchors),
            "anchors": track_anchors,
            "projection": "nearest anchor by measured motion tick",
            "calibration_file": str(args.calibration),
            "clock_file": str(args.clock),
            "limitation": "Anchor boxes are threshold-confirmed matches, not exact organizer coordinates.",
        }
        track_path.write_text(json.dumps(payload, indent=2) + "\n")
        total_repaired += len(repaired_rows)
        total_removed += removed_rows
        track_summaries.append(
            {
                "track_id": track_id,
                "class": track_anchors[0]["class"],
                "anchors": len(track_anchors),
                "repaired_rows": len(repaired_rows),
                "replaced_rows": len(repaired_rows),
                "removed_nonvisible_rows": removed_rows,
                "mean_iou_vs_previous": float(
                    np.mean([row["repair_iou_vs_previous"] for row in repaired_rows])
                ),
            }
        )
        for anchor in track_anchors:
            comparisons.append(
                {
                    "candidate_id": anchor["candidate_id"],
                    "track_id": track_id,
                    "frame": anchor["frame"],
                    "class": anchor["class"],
                    "iou_score_anchor_vs_previous": iou(
                        np.asarray(anchor["bbox_source_xyxy"]),
                        np.asarray(old_by_frame[anchor["frame"]]),
                    ),
                    "iou_score_anchor_vs_repaired": 1.0,
                }
            )

    canonicalized_rows = canonicalize_normalized_boxes(args.output)
    report = {
        "description": "Non-destructive repair of known tracks from validation-score-confirmed anchors.",
        "source_dataset": str(args.source),
        "output_dataset": str(args.output),
        "summary": {
            "score_confirmed_anchors": sum(len(rows) for rows in anchors.values()),
            "affected_tracks": len(anchors),
            "repaired_rows": total_repaired,
            "replaced_rows": total_repaired,
            "removed_nonvisible_rows": total_removed,
            "canonicalized_normalized_rows": canonicalized_rows,
            "discovery_complete": False,
        },
        "tracks": track_summaries,
        "anchor_comparisons": comparisons,
        "method": {
            "projection": "nearest score-confirmed anchor through calibrated shared projective motion",
            "calibration_file": str(args.calibration),
            "clock_file": str(args.clock),
            "source_candidate_union": str(args.candidate_union),
        },
        "limitations": [
            "The live score establishes only class and IoU >= 0.50 at each anchor.",
            "Projection does not independently confirm other frames.",
            "This repair cannot find missing objects or missing classes.",
        ],
    }
    dataset = export_metadata(args.output, report)
    report["output_counts"] = {
        "annotations": dataset["annotation_count"],
        "classes": dataset["class_counts"],
        "provenance": dataset["provenance_counts"],
    }
    (args.output / "score-anchor-repair-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    completion_path = args.output / "completion-report.json"
    completion = json.loads(completion_path.read_text())
    completion["score_anchor_repair"] = report["summary"] | {
        "report": str(args.output / "score-anchor-repair-report.json")
    }
    completion_path.write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(report["summary"] | report["output_counts"], indent=2))


if __name__ == "__main__":
    main()
