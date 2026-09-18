"""Run the bounded emergency-commitment probe on one fixed fixture."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
from simple_chase import run_streaming_v2 as harness

harness.OUT = ROOT / "results" / "short_overlap_sol" / "v16_emergency_probe"

if __name__ == "__main__":
    path, result = harness.run(
        policy_spec="short_overlap_sol.policy_v16_emergency_commit:SimpleChase",
        map_seed=10032, fixture_seed=9632, seconds=180, predators=1,
        station_bait=True, native_render=True, native_width=320,
        mode="direct_guide",
    )
    print(json.dumps({k: result.get(k) for k in
          ("map_seed", "fixture_seed", "seconds", "guide_alive",
           "physical_entered", "guided_delivered",
           "physical_losses_after_delivery", "success")}), flush=True)
    print(path)
