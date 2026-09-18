#!/usr/bin/env python3
"""Record a review decision in the validation spatial-coverage ledger."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ALLOWED = {"unreviewed", "reviewed_empty", "reviewed_objects_resolved", "needs_followup"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--pass", dest="review_pass", choices=["primary_review", "small_object_review"], required=True)
    parser.add_argument("--sheet-id", action="append", required=True)
    parser.add_argument("--status", choices=sorted(ALLOWED - {"unreviewed"}), required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--note", required=True)
    args = parser.parse_args()

    payload = json.loads(args.ledger.read_text())
    wanted = set(args.sheet_id)
    available = {row["sheet_id"] for row in payload["sheets"]}
    missing = wanted - available
    if missing:
        raise SystemExit(f"Unknown sheet ids: {sorted(missing)}")
    stamp = datetime.now(timezone.utc).isoformat()
    for row in payload["sheets"]:
        if row["sheet_id"] in wanted:
            row[args.review_pass] = {
                "status": args.status,
                "reviewer": args.reviewer,
                "reviewed_at": stamp,
                "note": args.note,
            }
    primary = sum(row["primary_review"]["status"].startswith("reviewed_") for row in payload["sheets"])
    small = sum(row["small_object_review"]["status"].startswith("reviewed_") for row in payload["sheets"])
    total = len(payload["sheets"])
    unresolved = sum(
        row["resolution_status"] == "requires_manual_track" for row in payload["candidates"]
    )
    motion_ok = payload["sampling"]["motion_gate_passed"] and all(
        row["status"] == "measured" or row.get("coverage_recovery_sheet_ids")
        for row in payload["sampling"]["motion_measurements"]
    )
    complete = primary == total and small == total and unresolved == 0 and motion_ok
    payload["completion"] = {
        "status": "complete" if complete else "in_progress",
        "primary_reviewed": primary,
        "small_object_reviewed": small,
        "total_sheets": total,
        "unresolved_score_confirmed": unresolved,
        "motion_gate_passed": motion_ok,
        "safe_for_reviewed_region_negative_training": complete,
        "safe_for_full_frame_negative_training": False,
        "negative_training_scope": (
            "Frame 5 is reviewed over the complete 3840 x 2160 image. Later frames are "
            "reviewed only in the top 540 px under the top-entry invariant. Do not use the "
            "unreviewed lower region as negative detector background."
        ),
        "updated_at": stamp,
    }
    args.ledger.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["completion"], indent=2))


if __name__ == "__main__":
    main()
