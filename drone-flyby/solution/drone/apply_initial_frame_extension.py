#!/usr/bin/env python3
"""Clone v3 and add the score-supported frame 1-4 large-launcher extension."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/drone/training/score-anchored-validation-v3"
OUTPUT = ROOT / "data/drone/training/score-anchored-validation-v4"
PROJECTIONS = ROOT / "artifacts/drone-validation-coverage/initial-frame-audit/projected-boxes.json"
CAPTURE_ROOT = (
    ROOT
    / "data/drone/capture/score-anchored-v3-c-20260918"
    / "score-anchored-v3-class-c-20260918"
)
SOURCE_SIZE = [3840, 2160]
DENOMINATOR = [3840.0, 2160.0, 3840.0, 2160.0]


def capture_by_frame() -> dict[int, tuple[Path, dict]]:
    result = {}
    for path in sorted(CAPTURE_ROOT.glob("*/*.json")):
        row = json.loads(path.read_text())
        frame = int(row["frame"])
        result[frame] = (path.with_name(row["image_file"]), row)
    return result


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite {OUTPUT}")
    shutil.copytree(SOURCE, OUTPUT)

    projected = json.loads(PROJECTIONS.read_text())
    early = []
    for frame_row in projected:
        launcher = next(
            row for row in frame_row["annotations"] if row["class"] == "large_launcher"
        )
        box = list(map(float, launcher["bbox_source_xyxy"]))
        early.append(
            {
                "seed_id": f"score-group-large-launcher-{frame_row['frame']:03d}",
                "frame": int(frame_row["frame"]),
                "class": "large_launcher",
                "bbox_source_xyxy": box,
                "bbox_normalized_xyxy": [round(value / scale, 8) for value, scale in zip(box, DENOMINATOR)],
                "evidence": "isolated_four-frame_large_launcher_validation_probe_positive",
                "label_contract": (
                    "The four-box class-isolated probe scored positively; the group contains at least one "
                    "IoU>=0.50 match. Individual rows are camera-motion projections and not exact ground truth."
                ),
                "review_status": "score_positive_group_projected_track",
                "track_id": "annotations",
                "provenance": "score_anchored_projective",
                "score_group_frames": [1, 2, 3, 4],
                "score_group_validation_score": 0.003808073115003808,
            }
        )

    track_path = OUTPUT / "annotations.json"
    track = json.loads(track_path.read_text())
    if min(int(row["frame"]) for row in track["annotations"]) != 5:
        raise ValueError("Expected source large-launcher track to start at frame 5")
    track["annotations"] = early + track["annotations"]
    track["initial_frame_extension"] = {
        "frames": [1, 2, 3, 4],
        "class": "large_launcher",
        "probe_score": 0.003808073115003808,
        "probe_name": "initial-frames-1-4-large_launcher-20260918",
        "mine_roller_control_score": 0.0,
        "limitation": "Group-positive validation does not independently certify each projected row.",
    }
    track_path.write_text(json.dumps(track, indent=2) + "\n")

    capture = capture_by_frame()
    image_root = OUTPUT / "early-frame-inputs"
    image_root.mkdir()
    early_image_meta = {}
    for frame in range(1, 5):
        image_path, row = capture[frame]
        target = image_root / f"frame_{frame:06d}.png"
        shutil.copy2(image_path, target)
        early_image_meta[frame] = {
            "image_file": f"../../early-frame-inputs/{target.name}",
            "image_dimensions": [int(row["view"]["width"]), int(row["view"]["height"])],
            "image_source_region_xyxy": row["view"]["source_region_xyxy"],
            "resolution_level": row["view"]["resolution_level"],
        }

    by_frame = defaultdict(list)
    class_counts: Counter[str] = Counter()
    provenance_counts: Counter[str] = Counter()
    ignored = {
        "completion-report.json",
        "score-anchor-repair-report.json",
        "class-correction-report.json",
        "initial-frame-extension-report.json",
    }
    for path in sorted(OUTPUT.glob("*.json")):
        if path.name in ignored:
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
            frame = int(row["frame"])
            by_frame[frame].append(item)
            class_counts[item["object_id"]] += 1
            provenance_counts[item["provenance"]] += 1

    metadata_root = OUTPUT / "finetune-metadata"
    annotation_root = metadata_root / "annotations"
    shutil.rmtree(annotation_root)
    annotation_root.mkdir(parents=True)
    frames = []
    for frame in range(1, 250):
        annotations = by_frame.get(frame, [])
        image = early_image_meta.get(
            frame,
            {
                "image_file": f"../../../reconstructed-validation/frame_{frame:06d}.png",
                "image_dimensions": SOURCE_SIZE,
                "image_source_region_xyxy": [0, 0, *SOURCE_SIZE],
                "resolution_level": "assembled_native",
            },
        )
        payload = {
            "frame": frame,
            **image,
            "annotation_coordinate_space": "source_3840x2160",
            "annotations": annotations,
            "object_counts": dict(sorted(Counter(row["object_id"] for row in annotations).items())),
            "provenance_counts": dict(sorted(Counter(row["provenance"] for row in annotations).items())),
            "coverage_status": (
                "early_camera_input_with_source_region" if frame < 5 else
                "score_anchor_repaired_and_class_corrected; discovery_not_certified"
            ),
        }
        filename = f"frame_{frame:06d}.json"
        (annotation_root / filename).write_text(json.dumps(payload, indent=2) + "\n")
        frames.append({"frame": frame, "annotation_file": f"annotations/{filename}", "annotations": len(annotations)})

    report = {
        "description": "Frame 1-4 large-launcher extension supported by an isolated validation-only probe.",
        "source_dataset": str(SOURCE),
        "output_dataset": str(OUTPUT),
        "added_annotations": 4,
        "accepted_class": "large_launcher",
        "accepted_probe_score": 0.003808073115003808,
        "rejected_class": "mine_roller",
        "rejected_probe_score": 0.0,
        "evaluations_after_probe": 0,
        "limitations": [
            "The group-positive score proves at least one of four boxes matches, not all four independently.",
            "Frames 1-4 do not have complete native 4K reconstructions; exact captured camera inputs are bundled with source-region metadata.",
        ],
    }
    (OUTPUT / "initial-frame-extension-report.json").write_text(json.dumps(report, indent=2) + "\n")
    dataset = json.loads((metadata_root / "dataset.json").read_text())
    dataset.update(
        {
            "dataset_name": "Nordic AI Cup 2026 score-anchored validation pseudo-labels v4",
            "frame_range_inclusive": [1, 249],
            "frame_count": 249,
            "annotation_count": sum(class_counts.values()),
            "class_counts": dict(sorted(class_counts.items())),
            "provenance_counts": dict(sorted(provenance_counts.items())),
            "frames": frames,
            "initial_frame_extension_report": "../initial-frame-extension-report.json",
            "early_frame_image_contract": (
                "Frames 1-4 use exact 960x540 API camera inputs and record source_region_xyxy. "
                "Annotation bboxes remain in 3840x2160 source coordinates."
            ),
        }
    )
    dataset["limitations"] = [
        item for item in dataset.get("limitations", []) if item != "Frames 1-4 remain excluded."
    ] + report["limitations"]
    (metadata_root / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    print(json.dumps({"annotations": dataset["annotation_count"], "classes": dataset["class_counts"]}, indent=2))


if __name__ == "__main__":
    main()
