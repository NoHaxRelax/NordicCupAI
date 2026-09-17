"""Short-overlap fixture using the frozen accepting policy v2."""
from pathlib import Path
import hashlib
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "research" / "depth5_test"
sys.path[:0] = [str(HERE), str(BASE)]
source_path = BASE / "run.py"
source = source_path.read_text()
source = source.replace(
    "from observed_gap_policy import ObservedGapPolicy",
    "from short_overlap_policy_v2 import ShortOverlapPolicyV2 as ObservedGapPolicy",
)
source = source.replace(
    'OUT = ROOT / "results" / "depth5_test"',
    'OUT = ROOT / "results" / "short_overlap_sol" / "v2"',
)
source = source.replace(
    "assert 11 <= gap <= 19 and 55 <= length <= 100 and 55 <= right_length <= 100\n"
    "    assert overlap >= 55",
    "assert 11 <= gap <= 19 and 10.1 <= length <= 100 and 10.1 <= right_length <= 100\n"
    "    assert overlap >= 10.1",
)
source = source.replace(
    'policy_file = Path(__file__).with_name("observed_guide_policy.py" if guides\n'
    '                                           else "observed_gap_policy.py")',
    'policy_file = (BASE / "observed_guide_policy.py" if guides else\n'
    '               HERE / "short_overlap_policy_v2.py")',
)
source = source.replace(
    'f"depth{depth:g}-observed-gap', 'f"shortv2-depth{depth:g}-observed-gap'
)
source = source.replace(
    '"observation-only-gap-intake-v2"', '"short-overlap-observation-gap-v2"'
)
assert "ShortOverlapPolicyV2 as ObservedGapPolicy" in source
assert 'results" / "short_overlap_sol" / "v2' in source
namespace = {"__file__": str(source_path), "__name__": "short_overlap_v2_fixture",
             "BASE": BASE, "HERE": HERE}
exec(compile(source, str(source_path), "exec"), namespace)
run = namespace["run"]
SOURCE_HASHES = {
    "base_fixture": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    "policy_v2": hashlib.sha256((HERE / "short_overlap_policy_v2.py").read_bytes()).hexdigest(),
}

