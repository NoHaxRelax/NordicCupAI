"""Write non-destructive correction sidecars for the confounded pilot runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "short_overlap_sol"
POLICY = ROOT / "research" / "depth5_test" / "observed_gap_policy.py"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt_for(row):
    replay = row.get("replay")
    if not replay:
        return None
    return ROOT / Path(replay).parent.parent / (Path(replay).name[:-8] + ".json")


def main():
    audited = []
    for summary_name in ("summary.json", "frontier-summary.json",
                         "positive-controls-summary.json"):
        summary_path = OUT / summary_name
        for row in json.loads(summary_path.read_text())["rows"]:
            receipt = receipt_for(row)
            if receipt is None or not receipt.exists():
                continue
            overlap = row["overlap"]
            confounded = overlap < 54.9
            partial = row.get("seconds", 0) + .01 < row.get("requested_seconds", 0)
            if confounded:
                validity = "invalid_short_overlap_retention_test"
                reason = ("Unchanged ObservedGapPolicy requires each observed edge and "
                          "their overlap to be at least 54.9. It did not identify this "
                          "short fixture, so the bait never deployed into the refuge.")
            elif partial:
                validity = "quota_stopped_partial_control"
                reason = "The 60%-remaining quota sentinel stopped this accepted-wall control early."
            else:
                validity = "valid_long_wall_positive_control"
                reason = "The unchanged controller accepts this long-wall geometry; this is only a control."
            correction = {
                "schema": "short-overlap-correction-v1",
                "receipt": receipt.name,
                "original_receipt_sha256": sha256(receipt),
                "replay_unchanged": True,
                "overrides": {
                    "short_overlap_retention_valid": not confounded,
                    "evidence_class": validity,
                    "bait_deployment_expected": not confounded,
                    "exclude_from_short_overlap_success_denominator": confounded,
                },
                "reason": reason,
                "source_hashes": {
                    str(POLICY.relative_to(ROOT)): sha256(POLICY),
                },
            }
            sidecar = receipt.with_suffix(".correction.json")
            sidecar.write_text(json.dumps(correction, indent=2) + "\n")
            audited.append({
                "receipt": receipt.name,
                "overlap": overlap,
                "gap": row["gap"],
                "evidence_class": validity,
                "correction": sidecar.name,
            })
    counts = {}
    for row in audited:
        counts[row["evidence_class"]] = counts.get(row["evidence_class"], 0) + 1
    result = {
        "schema": "short-overlap-audit-v1",
        "conclusion": ("No run below overlap 54.9 tested retention because the unchanged "
                       "controller refused to map the fixture and did not deploy the bait."),
        "controller_constraints": {
            "minimum_edge_length": 54.9,
            "minimum_overlap": 54.9,
            "source": str(POLICY.relative_to(ROOT)),
            "sha256": sha256(POLICY),
        },
        "counts": counts,
        "runs": audited,
    }
    (OUT / "audit-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
