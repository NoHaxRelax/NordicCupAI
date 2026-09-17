"""Execute the depth-5 direct-arrival fixture on an exact native-map gap."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "research" / "depth5_test"
sys.path[:0] = [str(HERE), str(BASE), str(ROOT / "vendor" / "survival-simulator")]
source_path = BASE / "run.py"
source = source_path.read_text()
source = source.replace(
    "from observed_gap_policy import ObservedGapPolicy",
    "from short_overlap_policy_v2 import ShortOverlapPolicyV2 as ObservedGapPolicy",
)
source = source.replace(
    "from src.elements.obstacle import Obstacle",
    "from src.elements.obstacle import Obstacle\nfrom src.core import SimulationCore",
)
source = source.replace(
    'OUT = ROOT / "results" / "depth5_test"',
    'OUT = ROOT / "results" / "short_overlap_sol" / "native_v2"',
)
source = source.replace(
    "        native=False, every=10):",
    "        native=False, every=10, map_seed=10000, site_index=0):",
)
source = source.replace(
    "    assert 11 <= gap <= 19 and 55 <= length <= 100 and 55 <= right_length <= 100\n"
    "    assert overlap >= 55",
    "    assert 11 <= gap <= 19 and 10.1 <= length <= 100 and 10.1 <= right_length <= 100\n"
    "    assert overlap >= 10.1",
)
start = source.index("    env = controlled_env()")
end_marker = "    env._update_spatial_grid()\n"
end = source.index(end_marker, start) + len(end_marker)
replacement = '''    core = SimulationCore(starting_agents=0, starting_predators=0, seed=map_seed)
    env = core.env
    census_path = ROOT / "results" / "real_map_census" / f"map-{map_seed}.json"
    census = json.loads(census_path.read_text())
    candidates = [c for c in census["candidates"]
                  if c["corridor"] and c["overlap"] >= 19.9]
    if not candidates:
        raise ValueError("unsupported map: no overlap>=20 clear-approach candidate")
    candidates.sort(key=lambda c: (c["overlap"], c["gap"], c["pair"], c["mouth"]))
    site = candidates[site_index]
    mouth = tuple(site["mouth"])
    inward = tuple(site["inward"])
    cross = (-inward[1], inward[0])
    gap = float(site["gap"])
    length = right_length = overlap = float(site["overlap"])
    face_offset = 0.
    horizontal = abs(inward[0]) > .5
    start = (mouth[0] - inward[0] * 15, mouth[1] - inward[1] * 15)
    goal = (mouth[0] + inward[0] * depth, mouth[1] + inward[1] * depth)
    bait = add_agent(env, *start, energy=150)
    bait.direction = math.atan2(inward[1], inward[0])
    env._next_agent_id = 1
    env._update_spatial_grid()
    assert not env._in_obstacle(start, radius=bait.size, obstacles=env.obstacles)
    assert not env._in_obstacle(goal, radius=bait.size, obstacles=env.obstacles)
'''
source = source[:start] + replacement + source[end:]
source = source.replace(
    'policy_file = Path(__file__).with_name("observed_guide_policy.py" if guides\n'
    '                                           else "observed_gap_policy.py")',
    'policy_file = (BASE / "observed_guide_policy.py" if guides else\n'
    '               HERE / "short_overlap_policy_v2.py")',
)
source = source.replace(
    'f"depth{depth:g}-observed-gap', 'f"native-shortv2-m{map_seed}-depth{depth:g}-observed-gap'
)
source = source.replace(
    '"observation-only-gap-intake-v2"', '"native-map-short-overlap-gap-v2"'
)
source = source.replace(
    'scenario="arranged single-axis sequential arrivals at one native-sized two-obstacle gap"',
    'scenario="exact native generated map; prepared direct arrivals at one census-selected gap"',
)
assert "SimulationCore(starting_agents=0" in source
assert "candidates.sort" in source
namespace = {"__file__": str(source_path), "__name__": "native_short_v2_fixture",
             "BASE": BASE, "HERE": HERE}
exec(compile(source, str(source_path), "exec"), namespace)
run = namespace["run"]
SOURCE_HASHES = {
    "base_fixture": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    "policy_v2": hashlib.sha256((HERE / "short_overlap_policy_v2.py").read_bytes()).hexdigest(),
    "adapter": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}

