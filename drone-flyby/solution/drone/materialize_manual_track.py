#!/usr/bin/env python3
"""Convert a visually reviewed tracking proposal into training pseudo-labels."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


WIDTH, HEIGHT = 3840, 2160


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--track-id", help="Select one track from a multi-track proposal file.")
    parser.add_argument("--start", type=int, help="First tracked frame to materialize.")
    parser.add_argument("--end", type=int, help="Last tracked frame to materialize.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proposal = json.loads(args.proposal.read_text())
    if args.track_id:
        candidates = [track for track in proposal.get("tracks", []) if track.get("track_id") == args.track_id]
        if len(candidates) != 1:
            raise SystemExit("--track-id must select exactly one track")
        proposal = candidates[0]
    if proposal.get("class_hypothesis") != args.label:
        raise SystemExit("Proposal class hypothesis does not match --label")
    track_token = re.sub(r"[^a-z0-9]+", "-", proposal.get("track_id", "track").lower()).strip("-")
    annotations = []
    for row in proposal["boxes"]:
        if args.start is not None and int(row["frame"]) < args.start:
            continue
        if args.end is not None and int(row["frame"]) > args.end:
            continue
        x1, y1, x2, y2 = row["bbox_source_xyxy"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(WIDTH, x2), min(HEIGHT, y2)
        if not (0 <= x1 < x2 <= WIDTH and 0 <= y1 < y2 <= HEIGHT):
            raise SystemExit(f"Invalid box in frame {row['frame']}")
        annotations.append({
            "seed_id": f"manual-{track_token}-{int(row['frame']):03d}",
            "frame": int(row["frame"]),
            "class": args.label,
            "bbox_source_xyxy": [round(v, 2) for v in row["bbox_source_xyxy"]],
            "bbox_normalized_xyxy": [round(x1 / WIDTH, 8), round(y1 / HEIGHT, 8), round(x2 / WIDTH, 8), round(y2 / HEIGHT, 8)],
            "evidence": "manual_reference_match",
            "label_contract": "Directly reviewed class match to the public reference; CSRT-assisted box reviewed per frame, not organizer ground truth.",
            "review_status": "directly_reviewed_track",
        })
    payload = {
        "description": "Direct visual training pseudo-label track. Not organizer annotations.",
        "source_dimensions": [WIDTH, HEIGHT],
        "annotations": annotations,
        "limitations": ["Only reviewed positives are included.", "Boxes are participant-derived training pseudo-labels."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"label": args.label, "annotations": len(annotations)}))


if __name__ == "__main__":
    main()
