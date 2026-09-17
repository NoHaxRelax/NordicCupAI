"""Unselected native-map screen for overlap>=20 direct-arrival retention."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
from native_map_run_v2 import run, ROOT, SOURCE_HASHES

STOP = Path("/tmp/predator-intake-stop")
OUT = ROOT / "results" / "short_overlap_sol" / "native_v2"
SEEDS = list(range(10000, 10012))


def job(seed):
    if STOP.exists(): return {"map_seed": seed, "stopped": True}
    try:
        row = run(depth=5, predators=33, seconds=120, interval=.1,
                  native=True, every=1, lateral_spread=0, heading_jitter=.3,
                  approach=60, map_seed=seed)
    except ValueError as exc:
        if "unsupported map" not in str(exc): raise
        return {"map_seed": seed, "unsupported": True, "error": str(exc)}
    row["map_seed"] = seed
    print({k: row.get(k) for k in ("map_seed", "overlap", "gap", "seconds",
          "mapped", "bait_alive", "acquired", "physical_losses", "final_joint")},
          flush=True)
    return row


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(job, SEEDS))
    (OUT / "screen-summary.json").write_text(json.dumps({
        "schema": "native-short-overlap-v2-screen-v1",
        "source_hashes": SOURCE_HASHES,
        "selection": "first 12 census maps, no result-based map selection; lowest-overlap clear candidate >=19.9",
        "rows": rows,
    }, indent=2) + "\n")
