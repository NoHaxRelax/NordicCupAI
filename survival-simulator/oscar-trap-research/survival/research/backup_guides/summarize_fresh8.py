"""Summarize the frozen BurstGuides 10200--10207 receipts without replay loading."""
from __future__ import annotations

import glob
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "backup_guides" / "fresh8" / "summary.json"


def main():
    rows = []
    for seed in range(10200, 10208):
        paths = glob.glob(str(ROOT / "results" / "simple_chase" /
                              f"real-map-BurstGuides-m{seed}-f{seed + 10000}-*.json"))
        if not paths:
            rows.append({"map_seed": seed, "fixture_seed": seed + 10000,
                         "completed": False,
                         "failure": "native_action_contract" if seed == 10203 else "no_receipt"})
            continue
        path = Path(paths[-1])
        data = json.loads(path.read_text())
        births = [row["child_id"] for row in data["child_births"]]
        guide_ids = {data["guide_id"], *births}
        dead = {row["agent_id"] for row in data["deaths"]}
        entry = data["physical_entry_times"][0]
        delivery = data["guided_delivery_times"][0]
        rows.append({
            "map_seed": seed, "fixture_seed": seed + 10000,
            "completed": True, "receipt": str(path.relative_to(ROOT)),
            "replay": data["replay"], "v3_success": data["success"],
            "physical_entry_time": entry, "v3_delivery_time": delivery,
            "birth_requests": len(data["spawn_requests"]),
            "native_children": len(births),
            "guide_deaths": len(guide_ids & dead),
            "surviving_guides": sorted(guide_ids - dead),
            # ReleaseMixin events live on each independent child controller;
            # BurstGuides does not aggregate them into its public event list.
            "surviving_released": None,
        })
    payload = {
        "policy": "backup_guides.burst:BurstGuides",
        "policy_sha256": "642725a431ee4778c1adcf100ffb45e7d85faf9c4e086f5923097f43202b7d9d",
        "seeds": [10200, 10207],
        "rows": rows,
        "v3_passes": sum(row.get("v3_success", False) for row in rows),
        "completed": sum(row["completed"] for row in rows),
        "limitations": [
            "Seed 10203 raised ValueError: move_distance outside native limit and has no completed replay.",
            "The receipt exposes survivor IDs but does not aggregate per-controller released flags.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
