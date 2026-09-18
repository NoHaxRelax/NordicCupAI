#!/usr/bin/env python3
"""Build one validation-only class-relabel hypothesis from a fixed plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--from-class", required=True)
    parser.add_argument("--to-class", required=True)
    args = parser.parse_args()

    source = json.loads(args.plan.read_text())
    changed = 0
    for rows in source["predictions_by_frame"].values():
        for row in rows:
            if row["object_id"] != args.from_class:
                raise ValueError(f"Unexpected source class {row['object_id']!r}")
            row["object_id"] = args.to_class
            changed += 1
    if not changed:
        raise ValueError("Source plan has no predictions")
    source["name"] = args.name
    source["score_assessment"] = {
        "hypothesis": f"{args.from_class} track relabelled as {args.to_class}",
        "prediction_count": changed,
        "validation_only": True,
        "source_plan": str(args.plan),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    filename = "plan.json"
    plan_path = args.output / filename
    plan_path.write_text(json.dumps(source, indent=2) + "\n")
    manifest = {
        "description": "Isolated validation-only class-relabel hypothesis.",
        "competition_evaluation": False,
        "plans": [{
            "shard": f"{args.from_class}-to-{args.to_class}",
            "file": filename,
            "name": args.name,
            "prediction_count": changed,
            "sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        }],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
