#!/usr/bin/env python3
"""Complete corrected validation tracks through the lower image with shared motion.

The script preserves valid reviewed boxes, removes tracker rows outside audited
native-pixel visibility ranges, and appends explicitly marked algorithmic boxes
only after the final retained reviewed frame of an eligible track.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from perspective_tracking.motion import MotionModel, ProjectionError


SOURCE_SIZE = (3840, 2160)
TOP_BAND_HEIGHT = 540
REJECTED_TRACKS = {
    "medium-plane-a-108-150": "Native crops drift across boats, water, dirt and roofs rather than a medium-plane sprite.",
    "medium-plane-b-095-150": "Native crops drift across boats, water, dirt and roofs rather than a medium-plane sprite.",
    "medium-plane-c-110-150": "Native crops drift across boats, water, dirt and roofs rather than a medium-plane sprite.",
}
NON_PROJECTIVE_TRACKS: set[str] = set()
LINKED_TRACK_FRAGMENTS = {
    "large-tower-a-extension-037": "large-tower-038-067",
    "small-launcher-b-early-118": "small-launcher-b-119-140",
    "small-tower-b-early-238-239": "small-tower-b-240-249",
}
# Native-resolution audit cutoffs.  The following projected frame was either
# fully outside the image or no longer contained visible target pixels.
VISUAL_COMPLETION_CUTOFFS = {
    "helicopter-043-073": 74,
    "jet-plane-d-040-082": 84,
    "large-tower-038-067": 68,
    "medium-plane-d-058-090": 90,
    "medium-plane-e-057-089": 89,
    "small-tower-c-155-164": 187,
    "tank-015-036": 46,
    "tank-e-066-075": 98,
}
# Native-resolution corrections for reviewed CSRT tracks whose tracker was
# already latched before the sprite entered, or stayed latched after it left.
# Bounds are inclusive and preserve partially clipped sprites at image edges.
VISUAL_REVIEWED_FRAME_RANGES = {
    "hangar-103-148": {
        "first_visible_frame": 110,
        "last_visible_frame": None,
        "reason": "Frames 103-109 follow marina water and boats; the hangar first appears at the top edge in frame 110.",
    },
    "condor-103-150": {
        "first_visible_frame": 109,
        "last_visible_frame": None,
        "reason": "Frames 103-108 follow marina boats; the condor first appears at the top edge in frame 109.",
    },
    "helicopter-b-105-149": {
        "first_visible_frame": None,
        "last_visible_frame": 138,
        "reason": "Frame 138 still contains the helicopter clipped by the bottom edge; frame 139 jumps right onto empty grass.",
    },
}
FULL_FRAME_FIRST_TRACKS = {"annotations", "mine-roller-a-005-009"}
LABEL_CONTRACT = (
    "Projective continuation from the latest complete reviewed box using the "
    "validation-calibrated shared camera-motion model and measured motion clock. "
    "Participant-created pseudo-label, not a directly reviewed box or organizer ground truth."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--clock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=5)
    parser.add_argument("--end", type=int, default=249)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_clock(path: Path, start: int, end: int) -> tuple[dict[int, float], list[dict]]:
    payload = json.loads(path.read_text())
    rows = payload["rows"]["L0_full"]
    clock = {start: 0.0}
    for row in rows:
        source, target = int(row["from"]), int(row["to"])
        if source < start or target > end:
            continue
        if source not in clock:
            raise ValueError(f"Motion clock is discontinuous at frame {source}")
        clock[target] = clock[source] + float(row["motion_ticks"])
    missing = sorted(set(range(start, end + 1)) - set(clock))
    if missing:
        raise ValueError(f"Motion clock lacks frames: {missing}")
    anomalies = [
        {
            "from": int(row["from"]),
            "to": int(row["to"]),
            "motion_ticks": float(row["motion_ticks"]),
        }
        for row in rows
        if float(row["motion_ticks"]) != 1.0
    ]
    return clock, anomalies


def clip_box(box: np.ndarray, size: tuple[int, int] = SOURCE_SIZE) -> np.ndarray:
    return np.r_[
        np.maximum(box[:2], 0.0),
        np.minimum(box[2:], np.asarray(size, dtype=float)),
    ]


def visible(box: np.ndarray) -> bool:
    clipped = clip_box(box)
    return bool(np.all(clipped[2:] > clipped[:2]))


def complete(box: np.ndarray) -> bool:
    width, height = SOURCE_SIZE
    return bool(box[0] > 0 and box[1] > 0 and box[2] < width and box[3] < height)


def bottom_or_side_exit(box: np.ndarray) -> bool:
    width, height = SOURCE_SIZE
    return bool(box[3] >= height - 1 or box[0] <= 0 or box[2] >= width)


def iou(first: np.ndarray, second: np.ndarray) -> float:
    first, second = clip_box(first), clip_box(second)
    intersection = np.maximum(
        0.0, np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2])
    ).prod()
    union = np.prod(first[2:] - first[:2]) + np.prod(second[2:] - second[:2]) - intersection
    return float(intersection / union) if union > 0 else 0.0


def backtest_track(
    track_id: str, rows: list[dict], model: MotionModel, clock: dict[int, float]
) -> dict:
    ordered = sorted(rows, key=lambda row: int(row["frame"]))
    if track_id in FULL_FRAME_FIRST_TRACKS:
        anchor = ordered[0]
    else:
        eligible = [
            row
            for row in ordered
            if complete(np.asarray(row["bbox_source_xyxy"], dtype=float))
            and float(row["bbox_source_xyxy"][3]) <= TOP_BAND_HEIGHT
        ]
        if not eligible:
            return {
                "anchor_frame": None,
                "predicted_exit_frame": None,
                "comparisons": 0,
                "passes_iou_0_50": 0,
                "ious": [],
            }
        anchor = eligible[-1]
    anchor_frame = int(anchor["frame"])
    anchor_box = np.asarray(anchor["bbox_source_xyxy"], dtype=float)
    scores = []
    predicted_exit_frame = None
    for row in ordered:
        frame = int(row["frame"])
        if frame <= anchor_frame:
            continue
        prediction = model.box(anchor_box, clock[anchor_frame], clock[frame])
        # Stop at the modelled source exit.  Tracker boxes recorded after this
        # point can be a CSRT latch onto unrelated bottom-edge texture and are
        # not a meaningful geometric comparison.
        if not visible(prediction):
            predicted_exit_frame = frame
            break
        scores.append(iou(prediction, np.asarray(row["bbox_source_xyxy"], dtype=float)))
    return {
        "anchor_frame": anchor_frame,
        "predicted_exit_frame": predicted_exit_frame,
        "comparisons": len(scores),
        "passes_iou_0_50": sum(score >= 0.50 for score in scores),
        "pass_fraction_iou_0_50": (
            sum(score >= 0.50 for score in scores) / len(scores) if scores else None
        ),
        "median_iou": float(np.median(scores)) if scores else None,
        "minimum_iou": min(scores) if scores else None,
        "ious": scores,
    }


def completion_row(
    *,
    track_id: str,
    label: str,
    frame: int,
    clipped: np.ndarray,
    unbounded: np.ndarray,
    anchor_frame: int,
    anchor_tick: float,
    tick: float,
    calibration: Path,
    clock_path: Path,
) -> dict:
    normalized = clipped / np.asarray([*SOURCE_SIZE, *SOURCE_SIZE], dtype=float)
    return {
        "seed_id": f"algorithmic-{track_id}-{frame:03d}",
        "track_id": track_id,
        "frame": frame,
        "class": label,
        "bbox_source_xyxy": np.round(clipped, 2).tolist(),
        "bbox_normalized_xyxy": np.round(normalized, 8).tolist(),
        "prediction_unclipped_source_xyxy": np.round(unbounded, 2).tolist(),
        "evidence": "algorithmic_shared_motion_from_reviewed_refresh",
        "provenance": "algorithmic_shared_motion",
        "review_status": "algorithmic_bottom_completion",
        "label_contract": LABEL_CONTRACT,
        "anchor_frame": anchor_frame,
        "anchor_motion_tick": anchor_tick,
        "motion_tick": tick,
        "calibration_file": str(calibration),
        "clock_file": str(clock_path),
    }


def export_frame_metadata(output: Path, start: int, end: int, report: dict) -> None:
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for path in sorted(output.glob("*.json")):
        if path.name == "completion-report.json":
            continue
        payload = json.loads(path.read_text())
        for row in payload.get("annotations", []):
            provenance = row.get("provenance", "reviewed_positive")
            by_frame[int(row["frame"])].append(
                {
                    "object_id": row["class"],
                    "bbox": row["bbox_source_xyxy"],
                    "track_id": row.get("track_id", path.stem),
                    "provenance": provenance,
                    "evidence": row.get("evidence"),
                    "review_status": row.get("review_status", "directly_reviewed"),
                    "anchor_frame": row.get("anchor_frame"),
                }
            )

    metadata_root = output / "finetune-metadata"
    annotation_root = metadata_root / "annotations"
    annotation_root.mkdir(parents=True, exist_ok=True)
    class_counts: Counter[str] = Counter()
    provenance_counts: Counter[str] = Counter()
    frame_rows = []
    for frame in range(start, end + 1):
        annotations = by_frame.get(frame, [])
        class_counts.update(row["object_id"] for row in annotations)
        provenance_counts.update(row["provenance"] for row in annotations)
        object_counts = Counter(row["object_id"] for row in annotations)
        provenance_on_frame = Counter(row["provenance"] for row in annotations)
        filename = f"frame_{frame:06d}.json"
        payload = {
            "frame": frame,
            "image_file": f"../../../reconstructed-validation/frame_{frame:06d}.png",
            "image_dimensions": list(SOURCE_SIZE),
            "annotations": annotations,
            "object_counts": dict(sorted(object_counts.items())),
            "provenance_counts": dict(sorted(provenance_on_frame.items())),
            "coverage_status": (
                "direct_full_frame_review"
                if frame == start
                else "top_entry_review_plus_algorithmic_track_continuation"
            ),
        }
        (annotation_root / filename).write_text(json.dumps(payload, indent=2) + "\n")
        frame_rows.append(
            {
                "frame": frame,
                "annotation_file": f"annotations/{filename}",
                "annotations": len(annotations),
                "provenance_counts": dict(sorted(provenance_on_frame.items())),
            }
        )

    dataset = {
        "dataset_name": "Nordic AI Cup 2026 validation pseudo-labels with lower-frame motion completion",
        "schema": "Reference-compatible object_id/bbox rows with additive track and provenance fields.",
        "image_root": "data/drone/reconstructed-validation",
        "image_dimensions": list(SOURCE_SIZE),
        "frame_range_inclusive": [start, end],
        "frame_count": end - start + 1,
        "annotation_count": sum(class_counts.values()),
        "class_counts": dict(sorted(class_counts.items())),
        "provenance_counts": dict(sorted(provenance_counts.items())),
        "frames": frame_rows,
        "method": report["method"],
        "validation_backtest": report["validation_backtest"],
        "coverage_contract": {
            "discovery": "Frame 5 was reviewed across the full image. Later frames were reviewed in the top 540 source pixels.",
            "continuation": "Eligible reviewed tracks are projected through the lower image with the validation-calibrated shared motion model.",
            "conditional_full_frame_use": "Full-frame training coverage assumes the supplied invariant that every new object enters through the reviewed top band and that top-band discovery is complete.",
            "filtering": "Use provenance=reviewed_positive for reviewed boxes only, or include provenance=algorithmic_shared_motion for the completed tracks.",
        },
        "limitations": [
            "Participant-created pseudo-labels, not organizer ground truth.",
            "Algorithmic boxes are projective forecasts from reviewed boxes and are not direct visual annotations.",
            "The measured backtest is an overlap-threshold diagnostic against participant-reviewed boxes, not competition mAP.",
            "Three false medium-plane tracks are excluded after native-resolution review showed drift across unrelated background regions.",
            "Premature and stale reviewed tracker boxes are excluded outside native-resolution visible frame ranges.",
            "Native-resolution tail audits stop eight tracks before an empty or visibly mismatched projected exit box.",
            "Frames 1-4 are incomplete reconstructions and remain excluded.",
        ],
    }
    (metadata_root / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output is not empty: {args.output}; pass --overwrite to replace generated files")
    args.output.mkdir(parents=True, exist_ok=True)

    model = MotionModel.from_dict(json.loads(args.calibration.read_text()))
    if tuple(model.source_size) != SOURCE_SIZE:
        raise ValueError(f"Expected {SOURCE_SIZE}, got {model.source_size}")
    clock, clock_anomalies = load_clock(args.clock, args.start, args.end)
    source_files = sorted(args.annotations.glob("*.json"))
    if not source_files:
        raise SystemExit(f"No top-level track JSON files found in {args.annotations}")

    track_reports = []
    all_backtest_scores = []
    source_reviewed_count = 0
    retained_reviewed_count = 0
    excluded_post_exit_count = 0
    excluded_visual_review_count = 0
    excluded_false_track_count = 0
    algorithmic_count = 0
    for path in source_files:
        payload = json.loads(path.read_text())
        original_rows = sorted(payload.get("annotations", []), key=lambda row: int(row["frame"]))
        if not original_rows:
            continue
        track_id = path.stem
        classes = {row["class"] for row in original_rows}
        if len(classes) != 1:
            raise ValueError(f"Expected one class in {path}, got {sorted(classes)}")
        label = next(iter(classes))
        source_reviewed_count += len(original_rows)

        visual_range = VISUAL_REVIEWED_FRAME_RANGES.get(track_id, {})
        first_visible_frame = visual_range.get("first_visible_frame")
        last_visible_frame = visual_range.get("last_visible_frame")
        backtest_rows = [
            row
            for row in original_rows
            if (first_visible_frame is None or int(row["frame"]) >= first_visible_frame)
            and (last_visible_frame is None or int(row["frame"]) <= last_visible_frame)
        ]
        if not backtest_rows:
            raise ValueError(f"Native-resolution visible range removed every row from {path}")
        backtest = backtest_track(track_id, backtest_rows, model, clock)
        all_backtest_scores.extend(backtest.pop("ious"))
        if track_id in REJECTED_TRACKS:
            excluded_false_track_count += len(original_rows)
            rejection = REJECTED_TRACKS[track_id]
            output_payload = {
                "description": "Rejected validation pseudo-label track retained as an audit record without training annotations.",
                "source_dimensions": list(SOURCE_SIZE),
                "track_id": track_id,
                "class": label,
                "source_annotation_file": str(path),
                "rejection": {
                    "status": "rejected_false_track_native_resolution_audit",
                    "reason": rejection,
                    "excluded_annotations": len(original_rows),
                },
                "annotations": [],
                "limitations": [
                    "The source rows are excluded from training metadata, overlays and video.",
                    "Participant-created audit decision, not organizer ground truth.",
                ],
            }
            (args.output / path.name).write_text(json.dumps(output_payload, indent=2) + "\n")
            track_reports.append(
                {
                    "track_id": track_id,
                    "class": label,
                    "source_reviewed_frames": len(original_rows),
                    "retained_reviewed_frames": 0,
                    "excluded_false_track_frames": len(original_rows),
                    "excluded_post_exit_frames": 0,
                    "excluded_post_exit_frame_range": None,
                    "excluded_visual_review_frames": 0,
                    "excluded_visual_review_frame_ranges": [],
                    "reviewed_frame_range": None,
                    "source_reviewed_frame_range": [
                        int(original_rows[0]["frame"]),
                        int(original_rows[-1]["frame"]),
                    ],
                    "algorithmic_frames": 0,
                    "algorithmic_frame_range": None,
                    "outcome": "rejected_false_track_native_resolution_audit",
                    "rejection_reason": rejection,
                    "backtest": backtest,
                }
            )
            continue
        predicted_exit_frame = backtest.get("predicted_exit_frame")
        excluded_rows = (
            [row for row in original_rows if int(row["frame"]) >= predicted_exit_frame]
            if predicted_exit_frame is not None
            else []
        )
        model_retained_rows = (
            [row for row in original_rows if int(row["frame"]) < predicted_exit_frame]
            if predicted_exit_frame is not None
            else original_rows
        )
        excluded_visual_pre_rows = [
            row
            for row in model_retained_rows
            if first_visible_frame is not None and int(row["frame"]) < first_visible_frame
        ]
        excluded_visual_post_rows = [
            row
            for row in model_retained_rows
            if last_visible_frame is not None and int(row["frame"]) > last_visible_frame
        ]
        excluded_visual_rows = excluded_visual_pre_rows + excluded_visual_post_rows
        retained_original_rows = [
            row
            for row in model_retained_rows
            if (first_visible_frame is None or int(row["frame"]) >= first_visible_frame)
            and (last_visible_frame is None or int(row["frame"]) <= last_visible_frame)
        ]
        if not retained_original_rows:
            raise ValueError(f"Exit censor removed every row from {path}")
        rows = []
        for original in retained_original_rows:
            row = deepcopy(original)
            source_box = np.asarray(row["bbox_source_xyxy"], dtype=float)
            clipped_source_box = clip_box(source_box)
            if not np.allclose(source_box, clipped_source_box):
                row["bbox_source_xyxy_unclipped"] = row["bbox_source_xyxy"]
            row["bbox_source_xyxy"] = np.round(clipped_source_box, 2).tolist()
            row["bbox_normalized_xyxy"] = np.round(
                clipped_source_box / np.asarray([*SOURCE_SIZE, *SOURCE_SIZE], dtype=float), 8
            ).tolist()
            row.setdefault("track_id", track_id)
            row.setdefault("provenance", "reviewed_positive")
            rows.append(row)
        retained_reviewed_count += len(rows)
        excluded_post_exit_count += len(excluded_rows)
        excluded_visual_review_count += len(excluded_visual_rows)

        last_row = retained_original_rows[-1]
        last_frame = int(last_row["frame"])
        last_box = np.asarray(last_row["bbox_source_xyxy"], dtype=float)
        complete_rows = [
            row
            for row in retained_original_rows
            if complete(np.asarray(row["bbox_source_xyxy"], dtype=float))
        ]
        additions = []
        if excluded_rows:
            outcome = "trimmed_post_exit_tracker_latch"
        elif track_id in LINKED_TRACK_FRAGMENTS:
            outcome = "continued_by_reviewed_track_fragment"
        elif track_id in NON_PROJECTIVE_TRACKS:
            outcome = "excluded_independent_object_motion"
        elif VISUAL_COMPLETION_CUTOFFS.get(track_id, args.end) <= last_frame:
            outcome = "no_visible_target_after_native_resolution_audit"
        elif last_frame >= args.end:
            outcome = "sequence_end_already_reached"
        elif bottom_or_side_exit(last_box):
            outcome = "reviewed_track_already_reaches_source_exit"
        elif not complete_rows:
            outcome = "no_complete_reviewed_anchor"
        else:
            anchor = complete_rows[-1]
            anchor_frame = int(anchor["frame"])
            anchor_box = np.asarray(anchor["bbox_source_xyxy"], dtype=float)
            anchor_tick = clock[anchor_frame]
            exited = False
            last_allowed_frame = min(args.end, VISUAL_COMPLETION_CUTOFFS.get(track_id, args.end))
            for frame in range(last_frame + 1, last_allowed_frame + 1):
                try:
                    unbounded = model.box(anchor_box, anchor_tick, clock[frame])
                except ProjectionError:
                    outcome = "projection_became_unavailable"
                    break
                if not visible(unbounded):
                    exited = True
                    outcome = "extended_until_source_exit"
                    break
                clipped = clip_box(unbounded)
                additions.append(
                    completion_row(
                        track_id=track_id,
                        label=label,
                        frame=frame,
                        clipped=clipped,
                        unbounded=unbounded,
                        anchor_frame=anchor_frame,
                        anchor_tick=anchor_tick,
                        tick=clock[frame],
                        calibration=args.calibration,
                        clock_path=args.clock,
                    )
                )
            else:
                outcome = (
                    "extended_to_native_resolution_audit_cutoff"
                    if last_allowed_frame < args.end
                    else "extended_to_sequence_end"
                )
            if additions and not exited and additions[-1]["frame"] == args.end:
                outcome = "extended_to_sequence_end"
        rows.extend(additions)
        algorithmic_count += len(additions)

        output_payload = {
            "description": "Reviewed validation pseudo-label track with separately marked shared-motion lower-frame completion.",
            "source_dimensions": list(SOURCE_SIZE),
            "track_id": track_id,
            "class": label,
            "source_annotation_file": str(path),
            "completion_method": {
                "name": "validation_calibrated_shared_projective_motion_with_reviewed_refresh",
                "calibration_file": str(args.calibration),
                "clock_file": str(args.clock),
                "adapt_edges": False,
                "reviewed_boxes_replaced": False,
                "linked_continuation_track": LINKED_TRACK_FRAGMENTS.get(track_id),
                "reviewed_visible_frame_range": visual_range or None,
            },
            "annotations": rows,
            "limitations": [
                "Participant-created pseudo-labels, not organizer ground truth.",
                "Rows with review_status=algorithmic_bottom_completion are forecasts rather than directly reviewed boxes.",
            ],
        }
        (args.output / path.name).write_text(json.dumps(output_payload, indent=2) + "\n")
        track_reports.append(
            {
                "track_id": track_id,
                "class": label,
                "source_reviewed_frames": len(original_rows),
                "retained_reviewed_frames": len(retained_original_rows),
                "excluded_post_exit_frames": len(excluded_rows),
                "excluded_post_exit_frame_range": (
                    [int(excluded_rows[0]["frame"]), int(excluded_rows[-1]["frame"])]
                    if excluded_rows
                    else None
                ),
                "excluded_visual_review_frames": len(excluded_visual_rows),
                "excluded_visual_review_frame_ranges": [
                    [int(rows[0]["frame"]), int(rows[-1]["frame"])]
                    for rows in (excluded_visual_pre_rows, excluded_visual_post_rows)
                    if rows
                ],
                "reviewed_frame_range": [int(retained_original_rows[0]["frame"]), last_frame],
                "source_reviewed_frame_range": [
                    int(original_rows[0]["frame"]),
                    int(original_rows[-1]["frame"]),
                ],
                "algorithmic_frames": len(additions),
                "algorithmic_frame_range": (
                    [additions[0]["frame"], additions[-1]["frame"]] if additions else None
                ),
                "outcome": outcome,
                "backtest": backtest,
            }
        )

    passes = sum(score >= 0.50 for score in all_backtest_scores)
    report = {
        "description": "Audit and provenance record for lower-frame validation-track completion.",
        "source_annotations": str(args.annotations),
        "output_annotations": str(args.output),
        "frame_range_inclusive": [args.start, args.end],
        "method": {
            "name": "validation_calibrated_shared_projective_motion_with_reviewed_refresh",
            "formula": "H(a,b)=(I+(b-origin)A) inverse(I+(a-origin)A)",
            "anchor_policy": "Latest complete reviewed box in each eligible track.",
            "clock_policy": "Cumulative L0_full measured motion ticks, including frozen and double-step transitions.",
            "calibration_file": str(args.calibration),
            "calibration_diagnostics": model.diagnostics,
            "clock_file": str(args.clock),
            "clock_anomalies": clock_anomalies,
            "adapt_edges": False,
            "non_projective_exclusions": sorted(NON_PROJECTIVE_TRACKS),
            "rejected_false_tracks": REJECTED_TRACKS,
            "linked_track_fragments": LINKED_TRACK_FRAGMENTS,
            "native_resolution_visual_audit_cutoffs": VISUAL_COMPLETION_CUTOFFS,
            "native_resolution_reviewed_track_ranges": VISUAL_REVIEWED_FRAME_RANGES,
        },
        "validation_backtest": {
            "definition": "Forecast from each track's last complete top-band box, or frame 5 for full-frame first objects, against its later reviewed boxes.",
            "comparisons": len(all_backtest_scores),
            "passes_iou_0_50": passes,
            "pass_fraction_iou_0_50": passes / len(all_backtest_scores),
            "median_iou": float(np.median(all_backtest_scores)),
            "note": "Diagnostic against participant-reviewed boxes, not organizer ground truth or competition mAP.",
        },
        "counts": {
            "track_files": len(track_reports),
            "source_reviewed_annotations": source_reviewed_count,
            "retained_reviewed_annotations": retained_reviewed_count,
            "excluded_post_exit_tracker_boxes": excluded_post_exit_count,
            "excluded_visual_review_tracker_boxes": excluded_visual_review_count,
            "excluded_false_track_boxes": excluded_false_track_count,
            "algorithmic_annotations": algorithmic_count,
            "combined_annotations": retained_reviewed_count + algorithmic_count,
            "extended_tracks": sum(row["algorithmic_frames"] > 0 for row in track_reports),
        },
        "tracks": track_reports,
    }
    (args.output / "completion-report.json").write_text(json.dumps(report, indent=2) + "\n")
    export_frame_metadata(args.output, args.start, args.end, report)
    print(json.dumps(report["counts"] | {"validation_backtest": report["validation_backtest"]}, indent=2))


if __name__ == "__main__":
    main()
