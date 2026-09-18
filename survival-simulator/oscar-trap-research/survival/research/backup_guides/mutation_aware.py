"""Untested adapter that bounds v24 hypothetical caps by native child traits."""
from simple_chase.policy_v24_short_fallback import SimpleChase as Base


class MutationAwareGuide(Base):
    def act(self, observations, sim_time):
        if observations:
            state = observations[0]
            self._native_speed = float(state["speed"])
            self._native_sprint_speed = float(state["sprint_speed"])
        return super().act(observations, sim_time)

    def _project_endpoint(self, state, pose, target, sprint=False):
        bounded = dict(state)
        bounded["speed"] = min(float(state["speed"]),
                               getattr(self, "_native_speed", float(state["speed"])))
        bounded["sprint_speed"] = min(float(state["sprint_speed"]),
                                      getattr(self, "_native_sprint_speed",
                                              float(state["sprint_speed"])))
        return super()._project_endpoint(bounded, pose, target, sprint)
