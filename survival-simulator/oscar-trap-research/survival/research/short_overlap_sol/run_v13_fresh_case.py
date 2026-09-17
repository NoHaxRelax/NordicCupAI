"""Run one predeclared frozen-v13 fresh case into this track's results."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
from simple_chase import run_streaming_v2 as harness

harness.OUT = ROOT / "results" / "short_overlap_sol" / "v13_fresh"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-seed", type=int, required=True)
    parser.add_argument("--fixture-seed", type=int, required=True)
    args = parser.parse_args()
    path, result = harness.run(
        policy_spec="simple_chase.policy_v13_close_reacquire:SimpleChase",
        map_seed=args.map_seed, fixture_seed=args.fixture_seed,
        seconds=180, predators=1, station_bait=True,
        native_render=True, native_width=320, mode="direct_guide",
    )
    print(json.dumps({k: result.get(k) for k in
          ("map_seed", "fixture_seed", "seconds", "guide_alive",
           "physical_entered", "guided_delivered",
           "physical_losses_after_delivery", "success")}), flush=True)
    print(path)
