#!/usr/bin/env python3
"""Export training crops from reviewed validation annotations.

The exported labels are participant-derived pseudo labels, not organizer
annotations.  Callers choose which evidence types are eligible; unreviewed
visual candidates and unreviewed geometrically propagated boxes must remain
excluded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


WIDTH, HEIGHT = 3840, 2160


def crop_bounds(box: list[float], margin: float) -> list[int]:
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return [
        max(0, int(x1 - dx)),
        max(0, int(y1 - dy)),
        min(WIDTH, int(x2 + dx + 0.9999)),
        min(HEIGHT, int(y2 + dy + 0.9999)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.35)
    parser.add_argument(
        "--include-evidence",
        action="append",
        default=None,
        help="Annotation evidence to export; defaults to score_confirmed only.",
    )
    args = parser.parse_args()
    if not 0 <= args.margin <= 2:
        raise SystemExit("--margin must be between 0 and 2")
    allowed_evidence = set(args.include_evidence or ["score_confirmed"])

    seeds = json.loads(args.seeds.read_text())
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in seeds["annotations"]:
        if seed.get("evidence") not in allowed_evidence:
            continue
        frame = int(seed["frame"])
        if frame < 5:
            continue
        source = args.images / f"frame_{frame:06d}.png"
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (HEIGHT, WIDTH):
            raise RuntimeError(f"Missing complete reconstructed image: {source}")
        box = [float(value) for value in seed["bbox_source_xyxy"]]
        x1, y1, x2, y2 = crop_bounds(box, args.margin)
        if not (x1 < x2 and y1 < y2):
            raise RuntimeError(f"Invalid crop for {seed['seed_id']}")
        label = seed["class"]
        relative = Path(label) / f"{seed['seed_id']}.png"
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), image[y1:y2, x1:x2]):
            raise RuntimeError(f"Could not write {destination}")
        records.append({
            "split": "validation_participant_derived",
            "seed_id": seed["seed_id"],
            "class": label,
            "frame": frame,
            "crop_file": str(relative),
            "crop_source_xyxy": [x1, y1, x2, y2],
            "object_bbox_source_xyxy": box,
            "object_bbox_normalized_xyxy": seed["bbox_normalized_xyxy"],
            "evidence": seed["evidence"],
            "label_contract": seed.get(
                "label_contract",
                "class-correct IoU >= 0.50 match; box is not organizer ground truth",
            ),
            "source": seed.get("source"),
        })
    evidence_description = ", ".join(sorted(allowed_evidence))
    manifest = {
        "description": "Participant-derived validation object crops for training and appearance review.",
        "source_dimensions": [WIDTH, HEIGHT],
        "margin_fraction": args.margin,
        "included_evidence": sorted(allowed_evidence),
        "records": records,
        "limitations": [
            f"Only reviewed annotations with evidence in [{evidence_description}] are included.",
            "Each record's label_contract states what its evidence supports.",
            "These are participant-derived pseudo-labels, not organizer-exported ground truth.",
            "Do not use an unlabelled image region as background unless the coverage ledger marks that region reviewed.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"exported_crops": len(records), "classes": sorted({r['class'] for r in records})}))


if __name__ == "__main__":
    main()
