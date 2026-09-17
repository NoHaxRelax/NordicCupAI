"""Sequentially resume the six interrupted/unstarted native-map cases."""
from pathlib import Path
import json

from native_map_stream_v3 import run, ROOT, SOURCE_HASHES

STOP = Path("/tmp/predator-intake-stop")
OUT = ROOT / "results" / "short_overlap_sol" / "native_v3_stream"
SEEDS = list(range(10006, 10012))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        if STOP.exists():
            rows.append({"map_seed": seed, "stopped_before_start": True})
            break
        try:
            row = run(depth=5, predators=33, seconds=120, interval=.1,
                      native=True, every=1, lateral_spread=0,
                      heading_jitter=.3, approach=60, map_seed=seed)
        except ValueError as exc:
            if "unsupported map" not in str(exc):
                raise
            row = {"map_seed": seed, "unsupported": True, "error": str(exc)}
        else:
            row["map_seed"] = seed
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in
              ("map_seed", "unsupported", "seconds", "overlap", "gap",
               "bait_alive", "acquired", "physical_losses", "final_joint")}),
              flush=True)
    (OUT / "resume-summary.json").write_text(json.dumps({
        "schema": "native-short-overlap-streaming-v3",
        "source_hashes": SOURCE_HASHES,
        "selection": "sequential resume of interrupted/unstarted seeds10006..10011",
        "rows": rows,
    }, indent=2) + "\n")
