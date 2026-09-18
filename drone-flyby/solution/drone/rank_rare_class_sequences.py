#!/usr/bin/env python3
"""Rank temporally coherent detector proposals for absent or rare classes."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np


WIDTH, HEIGHT = 3840, 2160


def iou(a: list[float], b: list[float]) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def centre_size(row: dict) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = row["bbox_source_xyxy"]
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def compatible(a: dict, b: dict) -> bool:
    gap = int(b["frame"]) - int(a["frame"])
    if not 1 <= gap <= 4:
        return False
    ax, ay, aw, ah = centre_size(a)
    bx, by, bw, bh = centre_size(b)
    # Sprite tracks move downward by roughly 45-85 px per ordinary frame in the
    # top band. A loose envelope retains entry clipping and clock irregularity.
    if not -15 * gap <= by - ay <= 115 * gap:
        return False
    if abs(bx - ax) > 90 * gap:
        return False
    if min(aw, ah, bw, bh) <= 0:
        return False
    return 0.45 <= bw / aw <= 2.2 and 0.35 <= bh / ah <= 2.8


def rank_sequences(rows: list[dict]) -> list[list[dict]]:
    rows = sorted(rows, key=lambda row: (int(row["frame"]), -float(row["confidence"])))
    score = []
    parent = []
    for i, row in enumerate(rows):
        base = 1.0 + np.log1p(float(row["confidence"]) / 0.005) + 0.15 * min(int(row.get("duplicate_tile_votes", 1)), 4)
        best, best_parent = base, -1
        for j in range(i):
            if compatible(rows[j], row):
                candidate = score[j] + base
                if candidate > best:
                    best, best_parent = candidate, j
        score.append(float(best))
        parent.append(best_parent)

    ordered = sorted(range(len(rows)), key=lambda index: score[index], reverse=True)
    sequences = []
    used_signatures = set()
    for end in ordered:
        indices = []
        cursor = end
        while cursor >= 0:
            indices.append(cursor)
            cursor = parent[cursor]
        indices.reverse()
        sequence = [rows[index] for index in indices]
        if len({row["frame"] for row in sequence}) < 3:
            continue
        signature = tuple((row["frame"], round(centre_size(row)[0] / 80), round(centre_size(row)[1] / 80)) for row in sequence)
        if signature in used_signatures:
            continue
        used_signatures.add(signature)
        sequences.append(sequence)
        if len(sequences) >= 12:
            break
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument(
        "--class-probabilities-top-per-frame",
        type=int,
        help="For classified reports, rank each requested class by its own probability and retain this many rows per frame.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    known = defaultdict(list)
    for path in args.dataset.glob("*.json"):
        payload = json.loads(path.read_text())
        for row in payload.get("annotations", []):
            known[(int(row["frame"]), row["class"])].append(row["bbox_source_xyxy"])

    gathered = defaultdict(list)
    for report_path in args.reports:
        report = json.loads(report_path.read_text())
        probability_rows = defaultdict(list)
        for row in report["predictions"]:
            if args.class_probabilities_top_per_frame and "class_probabilities" in row:
                for label in args.classes:
                    candidate = dict(row)
                    candidate["class"] = label
                    candidate["foreground_confidence"] = row["confidence"]
                    candidate["source_ensemble_class"] = row.get("ensemble_class")
                    candidate["confidence"] = row["class_probabilities"][label]
                    probability_rows[(int(row["frame"]), label)].append(candidate)
                continue
            label = row.get("ensemble_class", row.get("class"))
            if label not in args.classes:
                continue
            if any(iou(row["bbox_source_xyxy"], box) > 0.15 for box in known[(int(row["frame"]), label)]):
                continue
            candidate = dict(row)
            candidate["class"] = label
            if "ensemble_probability" in row:
                candidate["foreground_confidence"] = row["confidence"]
                candidate["confidence"] = row["ensemble_probability"]
            gathered[label].append(candidate)
        for (_, label), candidates in probability_rows.items():
            candidates.sort(key=lambda item: item["confidence"], reverse=True)
            gathered[label].extend(candidates[: args.class_probabilities_top_per_frame])

    # Deduplicate the sampled and every-frame reports.
    for label, rows in list(gathered.items()):
        unique = []
        for row in sorted(rows, key=lambda item: (item["frame"], -item["confidence"])):
            if any(other["frame"] == row["frame"] and iou(other["bbox_source_xyxy"], row["bbox_source_xyxy"]) > 0.35 for other in unique):
                continue
            unique.append(row)
        gathered[label] = unique

    result = {"description": "Temporally coherent detector proposals for manual native-4K review; not annotations.", "classes": {}}
    tiles = []
    tile_records = []
    for label in args.classes:
        sequences = rank_sequences(gathered[label])
        records = []
        for sequence_index, sequence in enumerate(sequences, start=1):
            sample_indices = sorted(set(np.linspace(0, len(sequence) - 1, min(5, len(sequence))).round().astype(int)))
            record = {
                "rank": sequence_index,
                "frames": [int(row["frame"]) for row in sequence],
                "detections": len(sequence),
                "mean_confidence": float(np.mean([row["confidence"] for row in sequence])),
                "max_confidence": float(max(row["confidence"] for row in sequence)),
                "rows": sequence,
                "review": "pending",
            }
            records.append(record)
            for sample_index in sample_indices:
                row = sequence[sample_index]
                frame = int(row["frame"])
                image = cv2.imread(str(args.images / f"frame_{frame:06d}.png"), cv2.IMREAD_COLOR)
                x1, y1, x2, y2 = row["bbox_source_xyxy"]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                side = 280
                left = max(0, min(WIDTH - side, int(round(cx - side / 2))))
                top = max(0, min(HEIGHT - side, int(round(cy - side / 2))))
                tile = image[top : top + side, left : left + side].copy()
                box = np.round(np.asarray([x1 - left, y1 - top, x2 - left, y2 - top])).astype(int)
                cv2.rectangle(tile, tuple(box[:2]), tuple(box[2:]), (0, 255, 255), 2)
                cv2.rectangle(tile, (0, 0), (side, 27), (0, 0, 0), -1)
                cv2.putText(tile, f"{label} #{sequence_index} f{frame}", (4, 19), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1, cv2.LINE_AA)
                tiles.append(tile)
                tile_records.append({"class": label, "rank": sequence_index, "frame": frame})
        result["classes"][label] = {"unmatched_predictions": len(gathered[label]), "sequences": records}

    pages = []
    blank = np.zeros((280, 280, 3), np.uint8)
    for page_start in range(0, len(tiles), 20):
        page_tiles = tiles[page_start : page_start + 20] + [blank] * max(0, 20 - len(tiles[page_start : page_start + 20]))
        page = np.vstack([np.hstack(page_tiles[row:row+5]) for row in range(0, 20, 5)])
        filename = f"rare-sequences-{page_start // 20 + 1:02d}.png"
        cv2.imwrite(str(args.output / filename), page)
        pages.append(filename)
    result["contact_pages"] = pages
    result["contact_tiles"] = tile_records
    result["limitations"] = [
        "Detector confidence and temporal coherence are proposal signals only.",
        "Map features can follow the same camera motion and must be rejected visually.",
        "A class with no coherent proposal may still be present below detector sensitivity.",
    ]
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pages": pages, "classes": {key: {"predictions": value["unmatched_predictions"], "sequences": len(value["sequences"])} for key, value in result["classes"].items()}}, indent=2))


if __name__ == "__main__":
    main()
