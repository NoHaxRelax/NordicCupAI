"""No-birth diagnostic of frozen v14; not a qualifying child protocol."""
from real_map_guide_sol.policy_v14_rest_reacquisition_frozen import RealMapGuidePolicy

class NoBirthDiagnostic(RealMapGuidePolicy):
    def act(self, observations, sim_time):
        actions=super().act(observations,sim_time)
        for action in actions:action['spawn_agent']=False
        return actions
