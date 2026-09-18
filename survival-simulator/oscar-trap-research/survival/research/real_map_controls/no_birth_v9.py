"""Diagnostic: suppress native birth to isolate one guide's pursuit/routing.

Not a qualifying parent/child strategy. Inherits the frozen latency-safe v9
controller unchanged except spawn requests. No dynamic oracle after native
observation handoff. The harness's child-required success remains false.
"""
from real_map_guide_sol.policy_v9_latency_safe_frozen import RealMapGuidePolicy

class NoBirthDiagnostic(RealMapGuidePolicy):
    def act(self, observations, sim_time):
        actions=super().act(observations,sim_time)
        for action in actions:action['spawn_agent']=False
        return actions
