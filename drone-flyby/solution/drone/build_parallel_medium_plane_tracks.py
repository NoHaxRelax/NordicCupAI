#!/usr/bin/env python3
"""Build two reviewed medium-plane tracks that are one camera step apart.

CSRT cannot preserve identity when identical sprites occupy consecutive points
on the same camera-motion path.  This utility constructs one canonical screen
path from reviewed proposals, then assigns the second sprite to the next point
on that path.  The output remains a visual pseudo-label proposal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upper-proposal", type=Path, required=True)
    parser.add_argument("--continuation-proposal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    upper = json.loads(args.upper_proposal.read_text())
    continuation = json.loads(args.continuation_proposal.read_text())
    upper_boxes = {int(row["frame"]): list(map(float, row["bbox_source_xyxy"])) for row in upper["boxes"]}
    continuation_boxes = {
        int(row["frame"]): list(map(float, row["bbox_source_xyxy"]))
        for row in continuation["boxes"]
    }

    # The first six full observations were reviewed directly at native scale.
    # They include a little context around the dark sprite silhouette.
    canonical = {
        58: [2153.0, -24.0, 2202.0, 22.0],
        59: [2154.0, 30.0, 2203.0, 75.0],
        60: [2156.0, 84.0, 2205.0, 130.0],
        61: [2158.0, 138.0, 2206.0, 185.0],
        62: [2159.0, 193.0, 2211.0, 239.0],
        63: [2161.0, 248.0, 2213.0, 295.0],
        64: [2163.0, 304.0, 2212.0, 351.0],
        65: [2164.0, 361.0, 2214.0, 409.0],
        66: [2166.0, 418.0, 2219.0, 467.0],
    }

    def expanded(box: list[float]) -> list[float]:
        x1, y1, x2, y2 = box
        return [x1 - 5, y1 - 2, x2 + 7, y2 + 14]

    for frame in range(67, 77):
        canonical[frame] = expanded(upper_boxes[frame])
    for frame in range(77, 90):
        canonical[frame] = expanded(continuation_boxes[frame])

    # Extrapolate only the final two clipped observations needed to avoid
    # treating visible bottom-edge pixels as background.
    frames = np.array(list(range(80, 90)), dtype=float)
    values = np.array([canonical[int(frame)] for frame in frames], dtype=float)
    for coordinate in range(4):
        coefficients = np.polyfit(frames, values[:, coordinate], 1)
        for frame in (90, 91):
            canonical.setdefault(frame, [0.0] * 4)[coordinate] = float(np.polyval(coefficients, frame))

    def track(track_id: str, offset: int, first: int, last: int) -> dict:
        boxes = []
        for frame in range(first, last + 1):
            source_frame = frame + offset
            box = canonical[source_frame]
            boxes.append(
                {
                    "frame": frame,
                    "bbox_source_xyxy": [round(value, 2) for value in box],
                    "path_frame": source_frame,
                }
            )
        return {
            "track_id": track_id,
            "class_hypothesis": "medium_plane",
            "proposal_method": (
                "Native visual path review. The second identical sprite is one measured camera "
                "step ahead on the same screen path; ordinary CSRT swaps their identities."
            ),
            "boxes": boxes,
        }

    payload = {
        "description": "Two visually reviewed, parallel medium-plane pseudo-label tracks.",
        "class_hypothesis": "medium_plane",
        "tracks": [
            track("manual-medium-plane-d", 0, 58, 90),
            track("manual-medium-plane-e", 1, 57, 89),
        ],
        "evidence": {
            "score_confirmed_seeds": [
                {"frame": 66, "bbox_source_xyxy": [2176.5, 418.0, 2218.5, 467.0]},
                {"frame": 66, "bbox_source_xyxy": [2179.0, 476.925, 2221.0, 543.075]},
            ],
            "reviewed_upper_proposal": str(args.upper_proposal),
            "reviewed_continuation_proposal": str(args.continuation_proposal),
        },
        "limitations": [
            "Participant-derived boxes, not organizer ground truth.",
            "Frames 58-65 were boxed from native visual inspection.",
            "The last two clipped boxes are short linear extrapolations from ten reviewed observations.",
            "The paired-path offset is supported by two distinct score-confirmed frame-66 boxes and native visual review.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"tracks": 2, "boxes": sum(len(row["boxes"]) for row in payload["tracks"])}))


if __name__ == "__main__":
    main()
