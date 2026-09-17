"""Instrumentation-only safety-first policy for first-step model audit."""
import math
from short_overlap_sol.policy_v21_safety_first import SimpleChase as V21


class SimpleChase(V21):
    def _evade(self, aid, state, pose, pred, rule, turn=0., desired=None):
        before = pose.p
        observed = self.last_predator_point
        obs = getattr(self, '_mpc_observation', None)
        target = self._escape_plan(state, pose, desired)
        predicted = None
        if target is not None and observed is not None and obs is not None:
            heading = self._wrap_heading(math.atan2(before[1]-observed[1], before[0]-observed[0]) - obs.get('rel_dir', 0.))
            hidden = [self._model_predator(observed, heading, before, speed) for speed in (11.,15.)]
            endpoint = self._project_endpoint(state, pose, target, True)
            next_models = [self._model_predator(pp, hh, endpoint, speed)
                           for (pp, hh), speed in zip(hidden, (11.,15.))]
            predicted = dict(chosen_target=[round(x,4) for x in target],
                             projected_endpoint=[round(x,4) for x in endpoint],
                             predicted_sep=[round(math.dist(endpoint, pp),4) for pp,_ in next_models])
        action = super()._evade(aid, state, pose, pred, rule, turn=turn, desired=desired)
        if predicted:
            self.decisions[aid].update(predicted)
        return action

    @staticmethod
    def _wrap_heading(a):
        return (a+math.pi)%(2*math.pi)-math.pi
