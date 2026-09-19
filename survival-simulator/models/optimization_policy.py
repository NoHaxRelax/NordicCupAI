"""Native-game colony without predator trapping.

Runs Nikolaj's shared-map/coordination stack alone: world estimation, global
planning, crowd/territory coordination, coordinated harvest and reproduction.
No entrapment module, no separate gathering policy with its own pose tracking.
"""
from models.exploration.expert_policy import ExpertConfig, ExpertPolicy, load_config
from models.exploration.global_planner import PlannerConfig, load_planner_config


class OptimizationPolicy:
    """ExpertPolicy alone. Deterministic: this stack uses no policy RNG."""

    def __init__(self, config=None, planner=None):
        if config is None:
            # The checked-in default ships harvest disabled so the trapping colony
            # stays in pure-exploration mode. Without trapping there is no site to
            # protect, so the coordinated harvest/population system runs from the start.
            settings = load_config()
            settings = settings.model_copy(update={'harvest': settings.harvest.model_copy(update={'enabled': True})})
        else:
            settings = ExpertConfig.model_validate(config)
        planner_settings = PlannerConfig.model_validate(planner) if planner is not None else load_planner_config()
        self.explorer = ExpertPolicy(settings, planner_settings)

    @property
    def estimator(self):
        return self.explorer.planner.estimator

    def __call__(self, states_list, sim_time, game_status="ok"):
        actions = self.explorer.actions_for_step(states_list, sim_time, game_status)
        return [(action.agent_id, action) for action in actions]

    def snapshot(self):
        return dict(
            phase='population' if self.explorer.planner.population_phase else 'exploration',
            estimated_agents={aid: dict(position=p.position.tolist(), heading=p.heading,
                uncertainty=p.uncertainty, group=p.group_id) for aid, p in self.estimator.poses.items()},
        )
