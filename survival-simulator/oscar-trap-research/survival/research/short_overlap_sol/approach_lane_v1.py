"""Frozen minimal approach-lane fix for the map10008/fixture9308 failure.

The parent implementation changes the final-approach transition from an
8-unit point target to an observation-derived axial lane: guide 80--150 units
outside the mouth, cross-track error below15, visible follower below70, and
predator behind within cross20. No dynamic privileged state is introduced.
"""
import hashlib
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "simple_chase" / "policy_v10_approach_lane.py"
EXPECTED = "7e6cfad20eea082aa83e66a8cb505fe9ee9d19a5efb6aa601d6877a8bbd2d1e3"
if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != EXPECTED:
    raise RuntimeError("frozen approach-lane dependency changed")

from simple_chase.policy_v10_approach_lane import SimpleChase as _FrozenLane


class SimpleChase(_FrozenLane):
    pass

