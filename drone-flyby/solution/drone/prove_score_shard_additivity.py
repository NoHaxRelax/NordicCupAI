#!/usr/bin/env python3
"""Prove class-shard score additivity with the official local COCO scorer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=ROOT / "artifacts/drone-api-tests/score-assessment-20260917",
    )
    parser.add_argument(
        "--official-source",
        type=Path,
        default=ROOT / "artifacts/drone-source-2026-09-17",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=ROOT / "data/drone/reference",
    )
    parser.add_argument("--scene", default="helsinki")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.official_source))
    import utils  # pylint: disable=import-outside-toplevel

    utils.DATA_DIRECTORY = args.reference_root.resolve()
    from local_evaluator import oracle_predictions, score  # pylint: disable=import-outside-toplevel

    manifest = json.loads((args.bundle / "manifest.json").read_text())
    predictions = oracle_predictions(args.scene)
    full_score, full_ap = score(args.scene, predictions)
    frames = sorted(predictions)
    first_cut = len(frames) // 3
    second_cut = 2 * len(frames) // 3
    temporal_parts = (
        set(frames[:first_cut]),
        set(frames[first_cut:second_cut]),
        set(frames[second_cut:]),
    )
    temporal_scores = []
    for selected_frames in temporal_parts:
        temporal_predictions = {
            frame: predictions[frame] if frame in selected_frames else []
            for frame in frames
        }
        temporal_scores.append(score(args.scene, temporal_predictions)[0])
    shard_rows = []
    summed_score = 0.0
    for shard, classes in manifest["class_partition"].items():
        class_set = set(classes)
        selected = {
            frame: [row for row in rows if row["object_id"] in class_set]
            for frame, rows in predictions.items()
        }
        shard_score, ap_by_class = score(args.scene, selected)
        summed_score += shard_score
        shard_rows.append(
            {
                "shard": shard,
                "classes": classes,
                "score": shard_score,
                "ap_by_class": ap_by_class,
            }
        )

    difference = summed_score - full_score
    if abs(difference) > 1e-12:
        raise ValueError(
            f"Class-shard scores did not reproduce the union: "
            f"sum={summed_score}, full={full_score}"
        )
    proof = {
        "description": "Official local-scorer proof that disjoint class scores add to union mAP.",
        "scene": args.scene,
        "full_oracle_score": full_score,
        "full_ap_by_class": full_ap,
        "shards": shard_rows,
        "summed_shard_score": summed_score,
        "difference_from_full_score": difference,
        "temporal_thirds_counterexample": {
            "scores": temporal_scores,
            "sum": sum(temporal_scores),
            "difference_from_full_score": sum(temporal_scores) - full_score,
            "note": "Even perfect predictions split by frame do not sum exactly under COCO AP.",
        },
        "official_local_evaluator_sha256": sha256(args.official_source / "local_evaluator.py"),
        "faster_coco_eval_version": "1.8.0",
    }
    output = args.output or args.bundle / "offline-additivity-proof.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
