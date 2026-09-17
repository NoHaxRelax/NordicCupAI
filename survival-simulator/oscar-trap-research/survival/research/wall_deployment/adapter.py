"""Factory for the existing local debugger's --policy option. No network I/O."""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[1]/'vendor/survival-simulator'))
from controller import WallPolicy
from src.utils.DTOs import ActionRequest


def make_policy(seed=None):
    controller=WallPolicy()
    def policy(agent_states,sim_time):
        actions=controller.act(agent_states,sim_time)
        policy.last_decisions=controller.decisions
        return [(a['agent_id'],ActionRequest(**a)) for a in actions]
    policy.last_decisions={}
    return policy
