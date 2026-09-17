"""Observation-only terrain, recovery, release and emergency-reserve integration.

Static map is permitted. Dynamic state is exclusively native agent DTOs and
public time. Single-guide and emergency-reserve variants are separate policies.
"""
import math
from short_overlap_sol.policy_v26_integrated import SimpleChase as TerrainGuide
from release_validation.release_mixin_v2 import ReleaseMixin
from real_map_guide_sol.policy_v15_hearing_reacquisition_frozen import Pose, wrap
from backup_guides.emergency import EmergencyGuides as EmergencyBase


class SweepMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sweep_index = None

    def act(self, observations, sim_time):
        snapshots = {aid:Pose(tuple(p.p),p.theta) for aid,p in self.poses.items()}
        actions = super().act(observations, sim_time)
        states = {s['agent_id']:s for s in observations}
        for index,action in enumerate(actions):
            aid=action['agent_id']
            if (aid==self.bait_id or self.decisions.get(aid,{}).get('rule')!='bounded_last_observation_reacquisition'
                    or sim_time-self.last_predator_seen<=3.):
                continue
            state=states[aid]
            # The parent already integrated its proposed action; restore the
            # pre-action observation-derived pose before replacing that action.
            pose=self._localize_uncached(state) or snapshots.get(aid)
            if pose is None:continue
            self.poses[aid]=pose
            points=self._search_points()
            if self._sweep_index is None:
                self._sweep_index=min(range(len(points)),key=lambda i:math.dist(pose.p,points[i]))
            target=points[self._sweep_index]
            if math.dist(pose.p,target)<18:
                self._sweep_index=(self._sweep_index+1)%len(points)
                target=points[self._sweep_index]
            route=self._path(pose.p,target,5.01)
            waypoint=route[0] if route else pose.p
            action=self._move(aid,state,pose,waypoint,'global_static_sweep_reacquisition',turn=0.,sprint=True)
            action['turn_angle']=action['move_direction']
            pose.theta=wrap(pose.theta+action['turn_angle'])
            self.decisions[aid]['forward_sweep_gaze']=True
            actions[index]=action
        return actions


class Guide(ReleaseMixin, SweepMixin, TerrainGuide):
    def _select_site(self):
        from replaceable_sites.selector import enumerate_sites
        static=dict(width=self.width,height=self.height,obstacles=self.rects)
        for gap,overlap in ((10.9,20.),(10.1,20.),(10.1,10.3)):
            sites=enumerate_sites(static,min_gap=gap,min_overlap=overlap)
            if sites:
                return min(enumerate(sites),key=lambda row:(bool(row[1]['boundary_indices']),bool(row[1]['approach_lane_offset']),row[0]))[1]
        raise ValueError('no eligible replaceable site including validated short overlap')

    def _localize_uncached(self,state):
        return super()._localize(state)

    def _localize(self,state):
        pose=self.poses.get(state['agent_id'])
        if pose is not None and self._edges_consistent(state,pose,tolerance=.00001):
            return Pose(tuple(pose.p),pose.theta)
        return self._localize_uncached(state)

    def act(self,observations,sim_time):
        state=next((s for s in observations if s['agent_id']==self.guide_id),None)
        if state is not None:self.native_sprint=float(state['sprint_speed'])
        return super().act(observations,sim_time)

    def _project_endpoint(self,state,pose,target,sprint=False):
        # Hypothetical spacing caps may represent sprinting even though the
        # parent calls this with sprint=False. Bound by actual native sprint;
        # normal walking DTOs already carry their own smaller walking cap.
        limit=getattr(self,'native_sprint',float(state['sprint_speed']))
        bounded={**state,'speed':min(float(state['speed']),limit),
                 'sprint_speed':min(float(state['sprint_speed']),limit)}
        return super()._project_endpoint(bounded,pose,target,sprint)

    def _move(self,aid,state,pose,target,rule,turn=0.,sprint=True):
        # Native child mutations can make walking speed exceed sprint speed.
        bounded={**state,'speed':min(float(state['speed']),float(state['sprint_speed']))}
        return super()._move(aid,bounded,pose,target,rule,turn,sprint)


class EmergencyGuides(EmergencyBase):
    def _controller(self,aid):
        if aid not in self.controllers:
            self.controllers[aid]=Guide(self.static_map,bait_id=self.bait_id,guide_id=aid)
        return self.controllers[aid]
