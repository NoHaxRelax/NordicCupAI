"""Diagnostic without childbirth; isolates frozen v13 guide routing.

Not a qualifying parent/child delivery. Only birth actions are suppressed;
the native observation handoff, movement and predator dynamics stay intact.
"""
from real_map_guide_sol.policy_v13_safe_moving_gaze_frozen import RealMapGuidePolicy

class NoBirthDiagnostic(RealMapGuidePolicy):
    def act(self, observations, sim_time):
        actions=super().act(observations,sim_time)
        for action in actions:action['spawn_agent']=False
        return actions
