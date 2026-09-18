#!/usr/bin/env python3
"""Build validation-only probes with deterministic bbox geometry variants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def transform(box: list[float], spec: str) -> list[float]:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    parts = spec.split(":")
    mode = parts[0]
    if mode == "scale":
        factor = float(parts[1])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        x1, x2 = cx - width * factor / 2, cx + width * factor / 2
        y1, y2 = cy - height * factor / 2, cy + height * factor / 2
    elif mode == "pad":
        left, top, right, bottom = (float(value) for value in parts[1].split(","))
        x1, y1, x2, y2 = (
            x1 - width * left,
            y1 - height * top,
            x2 + width * right,
            y2 + height * bottom,
        )
    else:
        raise ValueError(f"Unknown variant mode: {mode}")
    return [max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name-prefix", required=True)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="NAME=scale:FACTOR or NAME=pad:LEFT,TOP,RIGHT,BOTTOM",
    )
    args = parser.parse_args()

    source = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_plans = []
    for raw_variant in args.variant:
        name, spec = raw_variant.split("=", 1)
        predictions = json.loads(json.dumps(source["predictions_by_frame"]))
        for rows in predictions.values():
            for row in rows:
                row["bbox"] = transform(row["bbox"], spec)
        plan = {
            "name": f"{args.name_prefix}-{name}",
            "target": source["target"],
            "predictions_by_frame": predictions,
        }
        filename = f"variant-{name}.json"
        payload = json.dumps(plan, indent=2) + "\n"
        (args.output / filename).write_text(payload)
        manifest_plans.append(
            {
                "shard": name,
                "file": filename,
                "name": plan["name"],
                "geometry_transform": spec,
                "prediction_count": sum(len(rows) for rows in predictions.values()),
                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            }
        )

    manifest = {
        "description": "BBox-geometry validation-only diagnostic probes.",
        "plans": manifest_plans,
        "source_plan": str(args.plan),
        "live_runs_queued": 0,
        "competition_evaluation": False,
        "execution_boundary": "Validation only. The live runner refuses a nonzero evaluation count.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
