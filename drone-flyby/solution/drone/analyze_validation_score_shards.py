#!/usr/bin/env python3
"""Verify three completed class-shard runs and sum their additive mAP scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def capture_rows(capture_root: Path, plan_name: str) -> list[dict]:
    folder = capture_root / plan_name
    paths = sorted(folder.glob("*/*.json"))
    if not paths:
        raise FileNotFoundError(f"No capture receipts found under {folder}")
    return [json.loads(path.read_text()) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=ROOT / "artifacts/drone-api-tests/score-assessment-20260917",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "artifacts/drone-api-tests/score-probes",
    )
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=ROOT / "data/drone/capture/score-assessment-20260917",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-missing-capture", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads((args.bundle / "manifest.json").read_text())
    summaries = []
    total_score = 0.0
    for item in manifest["plans"]:
        plan = json.loads((args.bundle / item["file"]).read_text())
        name = plan["name"]
        result_path = args.results_root / name / "result.json"
        result = json.loads(result_path.read_text())
        if result.get("errors"):
            raise ValueError(f"{name} has API errors: {result['errors']}")
        score = float(result["score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Invalid score for {name}: {score}")

        rows = capture_rows(args.capture_root, name)
        by_frame = {int(row["frame"]): row for row in rows}
        expected_frames = set(range(1, 250))
        if set(by_frame) != expected_frames:
            missing = sorted(expected_frames - set(by_frame))
            extra = sorted(set(by_frame) - expected_frames)
            if extra or len(missing) > args.max_missing_capture:
                raise ValueError(f"{name} capture mismatch: missing={missing}, extra={extra}")
        else:
            missing = []
        for frame, row in by_frame.items():
            expected = plan["predictions_by_frame"].get(str(frame), [])
            actual = row["response"]["annotations"]
            if actual != expected:
                raise ValueError(f"{name} frame {frame} response differs from its fixed plan")

        total_score += score
        summaries.append(
            {
                "shard": item["shard"],
                "name": name,
                "score": score,
                "frames_verified": len(by_frame),
                "missing_capture_frames": missing,
                "api_errors": [],
            }
        )

    if total_score > 1.0 + 1e-9:
        raise ValueError(f"Summed score exceeds the official range: {total_score}")
    summary = {
        "description": "Verified additive validation score from three class-disjoint runs.",
        "shards": summaries,
        "summed_full_prediction_map50": total_score,
        "interpretation": (
            "This equals the score of the union of the three fixed prediction plans. "
            "It is not evidence that the same labels or boxes transfer to evaluation."
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
