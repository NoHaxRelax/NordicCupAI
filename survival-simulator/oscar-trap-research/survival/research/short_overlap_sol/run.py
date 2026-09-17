"""Run the existing depth-5 arranged fixture with short wall overlap.

This adapter deliberately changes only the fixture's overlap assertion and
artifact destination. Simulator code, policy code, scoring, and replay
recording remain those of depth5_test/run.py.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "research" / "depth5_test"
sys.path.insert(0, str(BASE))

source_path = BASE / "run.py"
source = source_path.read_text()
source = source.replace(
    'OUT = ROOT / "results" / "depth5_test"',
    'OUT = ROOT / "results" / "short_overlap_sol"',
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
    'policy_file = BASE / ("observed_guide_policy.py" if guides\n'
    '                          else "observed_gap_policy.py")',
)
assert 'assert overlap >= 55' not in source
assert 'results" / "short_overlap_sol' in source

namespace = {
    "__file__": str(source_path),
    "__name__": "short_overlap_fixture",
    "BASE": BASE,
}
exec(compile(source, str(source_path), "exec"), namespace)
run = namespace["run"]
BASE_SOURCE_SHA256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

