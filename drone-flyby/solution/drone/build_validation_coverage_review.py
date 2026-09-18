#!/usr/bin/env python3
"""Build an auditable native-pixel review pass for validation object discovery.

The review set contains the complete first usable frame and overlapping top-band
samples thereafter.  Each source tile is shown at 1:1 pixels.  Existing manual
annotations and score-confirmed seeds are a candidate union, but they never
replace visual inspection of the raw tiles.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


WIDTH, HEIGHT = 3840, 2160
TILE_WIDTH, TILE_HEIGHT = 960, 540
HEADER = 34


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def intersects(box: Iterable[float], region: Iterable[int]) -> bool:
    x1, y1, x2, y2 = map(float, box)
    rx1, ry1, rx2, ry2 = map(float, region)
    return min(x2, rx2) > max(x1, rx1) and min(y2, ry2) > max(y1, ry1)


def read_annotations(directory: Path) -> list[dict]:
    rows = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text())
        for row in payload.get("annotations", []):
            if not {"frame", "class", "bbox_source_xyxy"} <= row.keys():
                continue
            item = dict(row)
            item["source_file"] = str(path)
            rows.append(item)
    return rows


def candidate_union(manual: list[dict], confirmed_path: Path | None) -> list[dict]:
    candidates = []
    manual_by_frame = defaultdict(list)
    for row in manual:
        manual_by_frame[int(row["frame"])].append(row)
        candidates.append(
            {
                "candidate_id": row["seed_id"],
                "frame": int(row["frame"]),
                "class": row["class"],
                "bbox_source_xyxy": row["bbox_source_xyxy"],
                "source": "manual_track",
                "resolution_status": "materialized_manual_annotation",
                "source_file": row["source_file"],
            }
        )
    if confirmed_path and confirmed_path.is_file():
        payload = json.loads(confirmed_path.read_text())
        for row in payload.get("annotations", []):
            frame = int(row["frame"])
            matches = [
                candidate
                for candidate in manual_by_frame[frame]
                if candidate["class"] == row["class"]
                and iou(candidate["bbox_source_xyxy"], row["bbox_source_xyxy"]) >= 0.20
            ]
            candidates.append(
                {
                    "candidate_id": f'score-{row["seed_id"]}',
                    "frame": frame,
                    "class": row["class"],
                    "bbox_source_xyxy": row["bbox_source_xyxy"],
                    "source": "score_confirmed_seed",
                    "resolution_status": (
                        "covered_by_manual_annotation" if matches else "requires_manual_track"
                    ),
                    "matched_manual_seed_ids": [match["seed_id"] for match in matches],
                    "source_file": str(confirmed_path),
                    "box_certainty": row.get("box_certainty"),
                }
            )
    return candidates


def tile_with_header(image: np.ndarray, title: str) -> np.ndarray:
    output = np.zeros((TILE_HEIGHT + HEADER, TILE_WIDTH, 3), dtype=np.uint8)
    output[HEADER:] = image
    cv2.putText(
        output,
        title,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def draw_candidates(tile: np.ndarray, region: list[int], candidates: list[dict]) -> None:
    rx1, ry1, _, _ = region
    for row in candidates:
        if not intersects(row["bbox_source_xyxy"], region):
            continue
        x1, y1, x2, y2 = row["bbox_source_xyxy"]
        p1 = (round(x1 - rx1), round(y1 - ry1))
        p2 = (round(x2 - rx1), round(y2 - ry1))
        unresolved = row["resolution_status"] == "requires_manual_track"
        color = (255, 0, 255) if unresolved else (40, 230, 40)
        cv2.rectangle(tile, p1, p2, color, 2, cv2.LINE_AA)
        label = f'{row["class"]} {row["source"]}'
        cv2.putText(
            tile,
            label,
            (max(0, p1[0]), max(18, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def make_sheet(
    frame: int,
    regions: list[list[int]],
    image: np.ndarray,
    candidates: list[dict],
    raw_path: Path,
    overlay_path: Path,
) -> None:
    raw_cells, overlay_cells = [], []
    for region in regions:
        x1, y1, x2, y2 = region
        raw = image[y1:y2, x1:x2, :3].copy()
        overlay = raw.copy()
        draw_candidates(overlay, region, candidates)
        title = f"frame {frame:03d} | source x={x1}:{x2}, y={y1}:{y2} | native 1:1"
        raw_cells.append(tile_with_header(raw, title))
        overlay_cells.append(tile_with_header(overlay, title))
    while len(raw_cells) < 4:
        blank = np.zeros_like(raw_cells[0])
        raw_cells.append(blank)
        overlay_cells.append(blank.copy())
    raw_sheet = cv2.vconcat([cv2.hconcat(raw_cells[:2]), cv2.hconcat(raw_cells[2:4])])
    overlay_sheet = cv2.vconcat(
        [cv2.hconcat(overlay_cells[:2]), cv2.hconcat(overlay_cells[2:4])]
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(raw_path), raw_sheet):
        raise RuntimeError(f"Could not write {raw_path}")
    if not cv2.imwrite(str(overlay_path), overlay_sheet):
        raise RuntimeError(f"Could not write {overlay_path}")


def displacement(images: Path, first: int, second: int) -> dict:
    """Measure direct top-band background displacement between review samples."""
    a = cv2.imread(str(images / f"frame_{first:06d}.png"), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(images / f"frame_{second:06d}.png"), cv2.IMREAD_GRAYSCALE)
    a = cv2.resize(a[:TILE_HEIGHT], (WIDTH // 4, TILE_HEIGHT // 4), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b[:TILE_HEIGHT], (WIDTH // 4, TILE_HEIGHT // 4), interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.02)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None:
        return {"from": first, "to": second, "status": "insufficient_features"}
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [x for pair in pairs if len(pair) == 2 for x, y in [pair] if x.distance < 0.72 * y.distance]
    if len(good) < 20:
        return {"from": first, "to": second, "status": "insufficient_matches", "matches": len(good)}
    source = np.float32([ka[m.queryIdx].pt for m in good])
    target = np.float32([kb[m.trainIdx].pt for m in good])
    affine, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=4000
    )
    if affine is None or inliers is None:
        return {"from": first, "to": second, "status": "fit_failed", "matches": len(good)}
    probe = np.float32([[[WIDTH / 8, TILE_HEIGHT / 8]]])
    projected = cv2.transform(probe, affine)[0, 0]
    delta = (projected - probe[0, 0]) * 4
    return {
        "from": first,
        "to": second,
        "status": "measured",
        "matches": len(good),
        "inliers": int(inliers.sum()),
        "top_centre_displacement_xy_px": [round(float(delta[0]), 2), round(float(delta[1]), 2)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--manual-annotations", type=Path, required=True)
    parser.add_argument("--confirmed-seeds", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-frame", type=int, default=5)
    parser.add_argument("--last-frame", type=int, default=249)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be positive")

    manual = read_annotations(args.manual_annotations)
    candidates = candidate_union(manual, args.confirmed_seeds)
    by_frame = defaultdict(list)
    for candidate in candidates:
        by_frame[candidate["frame"]].append(candidate)

    output = args.output
    raw_dir, overlay_dir = output / "raw", output / "overlay"
    sheets = []

    image = cv2.imread(str(args.images / f"frame_{args.first_frame:06d}.png"), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (HEIGHT, WIDTH):
        raise RuntimeError("The first review frame is missing or not complete 4K")
    opening_regions = [
        [x, y, x + TILE_WIDTH, y + TILE_HEIGHT]
        for y in range(0, HEIGHT, TILE_HEIGHT)
        for x in range(0, WIDTH, TILE_WIDTH)
    ]
    for group_index in range(4):
        regions = opening_regions[group_index * 4 : (group_index + 1) * 4]
        sheet_id = f"opening-f{args.first_frame:03d}-g{group_index}"
        raw = raw_dir / f"{sheet_id}.png"
        overlay = overlay_dir / f"{sheet_id}.png"
        make_sheet(args.first_frame, regions, image, by_frame[args.first_frame], raw, overlay)
        sheets.append(
            {
                "sheet_id": sheet_id,
                "frame": args.first_frame,
                "review_scope": "opening_full_frame",
                "source_regions_xyxy": regions,
                "raw_file": str(raw.relative_to(output)),
                "overlay_file": str(overlay.relative_to(output)),
                "candidate_ids": [
                    row["candidate_id"]
                    for row in by_frame[args.first_frame]
                    if any(intersects(row["bbox_source_xyxy"], region) for region in regions)
                ],
                "primary_review": {"status": "unreviewed"},
                "small_object_review": {"status": "unreviewed"},
            }
        )

    sample_frames = list(range(args.first_frame, args.last_frame + 1, args.stride))
    if sample_frames[-1] != args.last_frame:
        sample_frames.append(args.last_frame)
    top_regions = [[x, 0, x + TILE_WIDTH, TILE_HEIGHT] for x in range(0, WIDTH, TILE_WIDTH)]
    for frame in sample_frames[1:]:
        image = cv2.imread(str(args.images / f"frame_{frame:06d}.png"), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (HEIGHT, WIDTH):
            raise RuntimeError(f"Frame {frame} is missing or not complete 4K")
        sheet_id = f"entry-f{frame:03d}"
        raw = raw_dir / f"{sheet_id}.png"
        overlay = overlay_dir / f"{sheet_id}.png"
        make_sheet(frame, top_regions, image, by_frame[frame], raw, overlay)
        sheets.append(
            {
                "sheet_id": sheet_id,
                "frame": frame,
                "review_scope": "top_entry_band",
                "source_regions_xyxy": top_regions,
                "raw_file": str(raw.relative_to(output)),
                "overlay_file": str(overlay.relative_to(output)),
                "candidate_ids": [
                    row["candidate_id"]
                    for row in by_frame[frame]
                    if any(intersects(row["bbox_source_xyxy"], region) for region in top_regions)
                ],
                "primary_review": {"status": "unreviewed"},
                "small_object_review": {"status": "unreviewed"},
            }
        )

    motion = [
        displacement(args.images, first, second)
        for first, second in zip(sample_frames, sample_frames[1:])
    ]
    # Uniform water or another low-texture band can make the overlap estimate
    # unmeasurable.  In that case, add a complete-frame recovery review at the
    # target sample instead of silently assuming continuity.
    recovery_frames = sorted({row["to"] for row in motion if row["status"] != "measured"})
    for frame in recovery_frames:
        image = cv2.imread(str(args.images / f"frame_{frame:06d}.png"), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (HEIGHT, WIDTH):
            raise RuntimeError(f"Recovery frame {frame} is missing or not complete 4K")
        recovery_ids = []
        for group_index in range(4):
            regions = opening_regions[group_index * 4 : (group_index + 1) * 4]
            sheet_id = f"recovery-f{frame:03d}-g{group_index}"
            recovery_ids.append(sheet_id)
            raw = raw_dir / f"{sheet_id}.png"
            overlay = overlay_dir / f"{sheet_id}.png"
            make_sheet(frame, regions, image, by_frame[frame], raw, overlay)
            sheets.append(
                {
                    "sheet_id": sheet_id,
                    "frame": frame,
                    "review_scope": "motion_recovery_full_frame",
                    "source_regions_xyxy": regions,
                    "raw_file": str(raw.relative_to(output)),
                    "overlay_file": str(overlay.relative_to(output)),
                    "candidate_ids": [
                        row["candidate_id"]
                        for row in by_frame[frame]
                        if any(intersects(row["bbox_source_xyxy"], region) for region in regions)
                    ],
                    "primary_review": {"status": "unreviewed"},
                    "small_object_review": {"status": "unreviewed"},
                }
            )
        for row in motion:
            if row["to"] == frame and row["status"] != "measured":
                row["coverage_recovery_sheet_ids"] = recovery_ids
    measured_dy = [
        abs(row["top_centre_displacement_xy_px"][1])
        for row in motion
        if row["status"] == "measured"
    ]
    unresolved = [row for row in candidates if row["resolution_status"] == "requires_manual_track"]
    ledger = {
        "schema": 1,
        "created_at": now(),
        "source_dimensions": [WIDTH, HEIGHT],
        "tile_dimensions": [TILE_WIDTH, TILE_HEIGHT],
        "review_contract": {
            "opening": f"Review every pixel in complete frame {args.first_frame}.",
            "later_frames": (
                f"Under the user-supplied top-entry invariant, review the top {TILE_HEIGHT} pixels "
                f"every {args.stride} frames at native 1:1 tile resolution."
            ),
            "candidate_union": (
                "Manual annotations and score-confirmed seeds are overlays only. Raw tiles must still "
                "be inspected; an absent proposal is never evidence of an empty tile."
            ),
            "completion_gate": (
                "Every sheet needs both a primary review and a separate small-object review, all "
                "confirmed candidates must be materialized or explicitly adjudicated, and every "
                "accepted object must be tracked across its visible span."
            ),
        },
        "sampling": {
            "first_frame": args.first_frame,
            "last_frame": args.last_frame,
            "stride": args.stride,
            "top_band_height": TILE_HEIGHT,
            "sample_frames": sample_frames,
            "motion_measurements": motion,
            "max_measured_sample_displacement_y_px": max(measured_dy) if measured_dy else None,
            "motion_gate_px": 460,
            "motion_gate_passed": bool(measured_dy)
            and max(measured_dy) <= 460
            and all(
                row["status"] == "measured" or row.get("coverage_recovery_sheet_ids")
                for row in motion
            ),
            "note": (
                "The 460 px motion gate leaves at least 80 px overlap between successive 540 px "
                "entry bands. A failed estimate creates a complete-frame recovery review instead "
                "of being accepted as continuous motion."
            ),
        },
        "candidate_summary": {
            "total": len(candidates),
            "by_source": dict(Counter(row["source"] for row in candidates)),
            "unresolved_score_confirmed": len(unresolved),
            "unresolved_candidate_ids": [row["candidate_id"] for row in unresolved],
        },
        "candidates": candidates,
        "sheets": sheets,
        "completion": {
            "status": "in_progress",
            "primary_reviewed": 0,
            "small_object_reviewed": 0,
            "total_sheets": len(sheets),
            "safe_for_reviewed_region_negative_training": False,
            "safe_for_full_frame_negative_training": False,
            "negative_training_scope": (
                f"Frame {args.first_frame} is reviewed over the complete {WIDTH} x {HEIGHT} image. "
                f"Later frames are reviewed only in the top {TILE_HEIGHT} px under the top-entry "
                "invariant. Do not use the unreviewed lower region as negative detector background."
            ),
        },
        "limitations": [
            "This is participant-derived review evidence, not organizer ground truth.",
            "The later-frame scope depends on the user-supplied invariant that new objects enter from the top.",
            "Frames 1-4 are incomplete reconstructions and are not certifiable as full images.",
            "Until the completion gate passes, unannotated pixels must be ignored for detector loss.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "coverage-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    (output / "candidate-union.json").write_text(
        json.dumps({"created_at": ledger["created_at"], "candidates": candidates}, indent=2) + "\n"
    )
    guide = f"""# Native validation coverage review

This folder is an auditable object-discovery pass, not a claim of organizer ground truth.

## Review contract

- `raw/` is the authoritative visual input. Every source tile is shown at native 1:1 pixels.
- `overlay/` adds known candidates: green means already represented by a manual annotation; magenta means a score-confirmed seed still needs a manual track.
- Frame {args.first_frame} is reviewed completely. Later sheets cover the top {TILE_HEIGHT} pixels at a stride of {args.stride} frames.
- The measured-motion gate must stay below 460 px so consecutive top bands overlap by at least 80 px.
- A failed motion estimate adds a complete-frame recovery review for the target sample.
- A sheet is not complete until both `primary_review` and `small_object_review` are recorded in `coverage-ledger.json`.
- Do not use unreviewed or partially reviewed image regions as detector background.

## Current blockers

- Unresolved score-confirmed candidates: {len(unresolved)}
- Review sheets awaiting primary pass: {len(sheets)}
- Review sheets awaiting small-object pass: {len(sheets)}
"""
    (output / "README.md").write_text(guide)
    print(
        json.dumps(
            {
                "sheets": len(sheets),
                "sample_frames": len(sample_frames),
                "manual_annotations": len(manual),
                "candidates": len(candidates),
                "unresolved_score_confirmed": len(unresolved),
                "max_measured_sample_displacement_y_px": ledger["sampling"][
                    "max_measured_sample_displacement_y_px"
                ],
                "motion_gate_passed": ledger["sampling"]["motion_gate_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
