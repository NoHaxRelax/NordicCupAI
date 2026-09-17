"""Launch and fully score the frozen 10380--10387 leader-reserve batch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "research"), str(ROOT / "research/reliability_eval")]
from reliability_eval.score_receipt_v4 import score

LABEL = "leader-reserves-fresh8-10380"
OUT = ROOT / "results" / "integrated_guide" / LABEL
POLICY = "multi_site.leader_reserves:LeaderReserves"


def main():
    command = [
        sys.executable, str(ROOT / "research/integrated_guide/run_batch.py"),
        "--policy", POLICY, "--first-map", "10380", "--count", "8",
        "--label", LABEL, "--workers", "2",
    ]
    launch = subprocess.run(command, cwd=ROOT)
    execution_path = OUT / "execution.json"
    rows = (json.loads(execution_path.read_text()) if execution_path.exists()
            else [{"map_seed": seed, "fixture_seed": seed + 10000,
                   "status": "runner_interrupted_before_execution_summary",
                   "receipts": []} for seed in range(10380, 10388)])
    by_seed = {row["map_seed"]: row for row in rows}
    scored = []
    scores_dir = OUT / "v4-scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(10380, 10388):
        row = by_seed.get(seed, {"map_seed": seed, "fixture_seed": seed + 10000,
                                "status": "missing_execution_row", "receipts": []})
        receipts = row.get("receipts", [])
        if len(receipts) != 1:
            scored.append({**row, "pass": False,
                           "score_reason": ("stop_guard_or_not_started"
                                            if row.get("status") == "not_started_stop"
                                            else "missing_or_ambiguous_receipt")})
            continue
        receipt = ROOT / receipts[0]
        result = score(receipt)
        score_path = scores_dir / f"m{seed}.v4-score.json"
        score_path.write_text(json.dumps(result, indent=2) + "\n")
        scored.append({**row, "pass": bool(result.get("pass")),
                       "score_reason": result.get("reason"),
                       "delivery_time": result.get("active_bait_acquisition"),
                       "score": str(score_path.relative_to(ROOT)),
                       "replay_check": result.get("replay_check")})
    plan = json.loads((OUT / "PLAN.json").read_text()) if (OUT / "PLAN.json").exists() else None
    summary = {
        "schema": "leader-reserves-fresh8-v4-summary-v1",
        "policy": POLICY,
        "fixed_denominator": 8,
        "passes": sum(row["pass"] for row in scored),
        "failures": sum(not row["pass"] for row in scored),
        "batch_runner_returncode": launch.returncode,
        "plan": plan,
        "scorer_sha256": hashlib.sha256(
            (ROOT / "research/reliability_eval/score_receipt_v4.py").read_bytes()).hexdigest(),
        "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": scored,
        "fitted_cases_excluded": [10138, 10224],
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"summary": str((OUT / 'SUMMARY.json').relative_to(ROOT)),
                      "passes": summary["passes"], "denominator": 8}), flush=True)


if __name__ == "__main__":
    main()
