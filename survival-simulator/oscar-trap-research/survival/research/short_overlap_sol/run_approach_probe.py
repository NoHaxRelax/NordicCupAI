"""Run one resource-capped approach probe into this track's results."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
from simple_chase import run_streaming_v2 as harness

harness.OUT = harness.ROOT / "results" / "short_overlap_sol" / "approach_probe"

if __name__ == "__main__":
    path, result = harness.run(
        policy_spec="short_overlap_sol.approach_lane_v1:SimpleChase",
        map_seed=10008, fixture_seed=9308, seconds=120, predators=1,
        station_bait=True, native_render=True, native_width=400,
        mode="direct_guide",
    )
    print(json.dumps({k: result.get(k) for k in
          ("seconds", "guide_alive", "physical_entered", "guided_delivered",
           "physical_losses_after_delivery", "final_distance_from_mouth",
           "success")}), flush=True)
    print(path)
