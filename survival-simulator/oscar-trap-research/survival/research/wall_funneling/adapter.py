"""Observation-only policy factory for the existing debugger; no network I/O.

Requires an observed eligible wall and a crew on both sides. It is not a global
exploration policy. Resource support remains the caller's responsibility.
"""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[1]/'vendor/survival-simulator'))
from observed_policy import ObservedFunnel
from src.utils.DTOs import ActionRequest


def make_policy(seed=None):
    controller=ObservedFunnel(capacity=33,replenish=False,gather=False)
    def policy(agent_states,sim_time):
        actions=controller.act(agent_states,sim_time)
        policy.last_decisions=controller.decisions
        return [(a['agent_id'],ActionRequest(**a)) for a in actions]
    policy.last_decisions={}
    return policy
