"""Select fit guides and confirm nearby handovers from ordinary sightings.

Route viability is a conservative estimate, not a promise of delivery: it
charges walking through river for the whole route plus a short sprint reserve.
The local guide still checks terrain, energy and capture on each actual move.
"""
import math

from models.entrapment.guide_relief import AgentForecastInput, forecast_travel
from models.entrapment.predator_following import predator_is_not_following
from models.exploration.navigation import Navigator


class GuideCoordinator:
    def __init__(self):
        self.navigator = Navigator(clearance=10.5)
        self.assessments = {}
        self.sightings = {}
        self.following = {}

    def update(self, group, states, now, tracks):
        self.navigator.update(group, now)
        self.navigator.prune(states)
        self.assessments = {}
        keys = {(t.key, aid) for t in tracks for aid in t.observers}
        self.sightings = {key: value for key, value in self.sightings.items() if key in keys}
        self.following = {}

    def assess(self, aid, state, pose, handoff, now):
        if aid in self.assessments:
            return self.assessments[aid]
        row = dict(viable=False, reason='unlocalized')
        self.assessments[aid] = row
        if pose.uncertainty > 8.:
            return row
        if state['energy'] < state['max_energy']/5:
            row['reason'] = 'below_sprint_threshold'
            return row
        # Prefer agents with a real same-terrain escape advantage. Having
        # enough energy does not make a mutated sprint trait faster than 15.
        if state['sprint_speed'] <= 15.:
            row['reason'] = 'no_sprint_speed_advantage'
            return row
        plan = self.navigator.steer(aid, pose.position, handoff, now)
        row['route_status'] = plan.status
        if plan.blocked or not math.isfinite(plan.remaining):
            row['reason'] = 'no_observed_route'
            return row
        walk = min(state['speed'], state['sprint_speed'])
        route_time = plan.remaining / (walk * .3 * 10.)
        forecast = forecast_travel(AgentForecastInput(
            aid, state['energy'], state['max_energy'], state['age'],
            state['speed'], state['sprint_speed'], plan.remaining,
            terrain_progress=.3), wants_sprint=False, horizon=max(1., route_time+1.))
        # Five full-speed moves plus turning and earliest native senescence.
        old = .01*(state['age']+route_time+.5) if state['age']+route_time+.5 > 60. else 0.
        sprint_reserve = 5*(.05*walk + .5*max(0.,state['sprint_speed']-walk)+.1+old)+1.
        enough = forecast.arrives and forecast.energy_at_arrival >= state['max_energy']/5+sprint_reserve
        row.update(viable=enough, reason='fit' if enough else 'insufficient_route_energy',
                   route_distance=round(plan.remaining,2), estimated_walk_seconds=round(route_time,2),
                   estimated_arrival_energy=round(forecast.energy_at_arrival,2),
                   required_arrival_energy=round(state['max_energy']/5+sprint_reserve,2))
        return row

    def observe(self, key, aid, observation, bait, edges, state, tick):
        memory = self.sightings.setdefault((key,aid), {})
        predator_is_not_following(observation,bait,edges,state,{'tick':tick},memory)
        status = memory.get('_following_debug',{}).get('status')
        positive = status in ('direct_chase','watched_pivot')
        memory['_confirmed_ticks'] = memory.get('_confirmed_ticks',0)+1 if positive else 0
        confirmed = memory['_confirmed_ticks'] >= 2
        self.following[(key,aid)] = confirmed
        return confirmed

    def snapshot(self):
        return dict(candidates=self.assessments.copy(),
                    following_observers=[dict(track=key,agent=aid) for (key,aid), yes in self.following.items() if yes])
