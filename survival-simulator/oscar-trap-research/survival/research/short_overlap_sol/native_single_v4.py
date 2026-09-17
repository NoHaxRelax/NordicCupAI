"""Run one native-map case so every map starts in a fresh capped process."""
import argparse
import json
from pathlib import Path
from native_map_stream_v4 import run, ROOT, SOURCE_HASHES

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-seed", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=120)
    args = parser.parse_args()
    row = run(depth=5, predators=33, seconds=args.seconds, interval=.1,
              native=True, every=1, lateral_spread=0, heading_jitter=.3,
              approach=60, map_seed=args.map_seed)
    row["map_seed"] = args.map_seed
    out = ROOT / "results" / "short_overlap_sol" / "native_v4_stream"
    (out / f"map-{args.map_seed}-s{args.seconds:g}-summary.json").write_text(
        json.dumps({"schema": "native-short-overlap-stream-v4",
                    "source_hashes": SOURCE_HASHES, "result": row}, indent=2) + "\n")
    print(json.dumps({k: row.get(k) for k in
          ("map_seed", "seconds", "overlap", "gap", "bait_alive",
           "acquired", "physical_losses", "final_joint")}), flush=True)
