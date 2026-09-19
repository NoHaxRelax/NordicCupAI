"""Guard the standard the whole controller stack is built on.

Constants and mechanics recovered by reading the simulator are fair game: a
predator's walking speed is a fixed number anyone with the source can know.
Runtime hidden state is not: a predator's current energy, its rest flag, the
biome under someone else's feet, the true map size, an unsighted creature's
position. None of that is in the observation payload and none of it may reach
a decision.

The decisive test is replay. If every action is a pure function of the
observation stream, then feeding a recorded stream back with no simulator in
the process has to reproduce those actions exactly. Anything that had reached
into hidden state would diverge, because the state is no longer there.
"""

import copy
import json
from pathlib import Path
import unittest

from src.core import SimulationCore
from src.utils.DTOs import ObservationResponse, StepResponse
from src.utils.controllers import expert_policy as expert_policy_module
from src.utils.controllers.expert_policy import ExpertConfig, ExpertPolicy, load_config
from src.utils.controllers.global_planner import PlannerConfig, load_planner_config


CONTROLLERS = Path(expert_policy_module.__file__).parent


def policy(**belief):
    """The live tuned configuration, with the belief switched on."""
    expert = load_config().model_dump()
    expert["harvest"]["enabled"] = True
    planner = load_planner_config().model_dump()
    planner["draw_overlay"] = False
    planner["biome_inference"]["adapt_to_wall_clock"] = False
    planner["predator_belief"].update(dict(enabled=True), **belief)
    return ExpertPolicy(ExpertConfig.model_validate(expert),
                        PlannerConfig.model_validate(planner))


def record(ticks=150, seed=1007, predators=4):
    """Drive a real episode, keeping the payloads and the actions taken."""
    controller = policy()
    sim = SimulationCore(seed=seed, starting_predators=predators)
    state = sim.step([])
    tape, actions = [], []
    while len(tape) < ticks and state["num_agents"]:
        tape.append((state["sim_time"], copy.deepcopy(state["observations"])))
        step = controller.actions_for_step(state["observations"], sim_time=state["sim_time"])
        actions.append([action.model_dump() for action in step])
        state = sim.step([(action.agent_id, action) for action in step])
    return tape, actions


class ObservationOnlyTests(unittest.TestCase):
    def setUp(self):
        self.tape, self.actions = record()
        self.assertGreater(len(self.tape), 50, "episode ended before it could prove anything")

    def test_replaying_the_stream_without_a_simulator_reproduces_every_action(self):
        # Nothing but the recorded payloads is in scope here. A dependence on
        # any hidden value could not survive this.
        echo = policy()
        replayed = [[action.model_dump() for action in
                     echo.actions_for_step(copy.deepcopy(payload), sim_time=sim_time)]
                    for sim_time, payload in self.tape]
        self.assertEqual(self.actions, replayed)

    def test_the_observation_stream_is_plain_serialisable_data(self):
        """A live object in the payload would be a channel to hidden state."""
        json.dumps(self.tape, allow_nan=False)

    def test_payloads_carry_exactly_the_documented_fields(self):
        """An extra key would be a leak the replay test could not detect.

        Replay reproduces actions from whatever the payload held, so a payload
        that itself smuggled hidden state would replay happily. This pins the
        payload to the published DTO instead.
        """
        allowed = set(ObservationResponse.model_fields)
        for _, payload in self.tape:
            for state in payload:
                self.assertEqual(set(state), allowed)

    def test_controllers_never_import_the_simulator(self):
        """The mechanical half of the rule, kept cheap enough to always run."""
        offenders = {}
        for path in sorted(CONTROLLERS.glob("*.py")):
            code = " ".join(line.split("#")[0]
                            for line in path.read_text(encoding="utf-8").splitlines())
            for forbidden in ("src.elements", "src.core", "SimulationCore",
                              "biome_map", "agents_dict", "env."):
                if forbidden in code:
                    offenders.setdefault(path.name, []).append(forbidden)
        self.assertEqual(offenders, {})

    def test_no_controller_hardcodes_the_true_map_size(self):
        """World size has to be inferred from observed boundary edges.

        Unlike a predator's speed, the map's dimensions are per-episode state,
        so a literal here would be exactly the kind of knowledge the rule bars.
        """
        for path in sorted(CONTROLLERS.glob("*.py")):
            code = " ".join(line.split("#")[0]
                            for line in path.read_text(encoding="utf-8").splitlines())
            for forbidden in ("1600", "1200"):
                self.assertNotIn(forbidden, code, f"{path.name} hardcodes {forbidden}")

    def test_the_belief_samples_predator_energy_instead_of_reading_it(self):
        """Rest and speed depend on energy, which observations never expose.

        A freshly seen predator must therefore carry the whole active range,
        not the true value, so the 11-vs-15 speed ambiguity stays honest.
        """
        controller = policy()
        tracker = controller.planner.predator_tracker
        drawn = tracker._sample_energy(512)
        self.assertGreater(float(drawn.max()) - float(drawn.min()), 100.)
        self.assertGreaterEqual(float(drawn.min()), 0.)

    def test_step_response_is_the_only_channel_into_the_endpoint(self):
        """The served endpoint accepts the published DTO and nothing else."""
        self.assertEqual(set(StepResponse.model_fields),
                         {"game_status", "score", "sim_time", "n_agents", "agent_status"})


if __name__ == "__main__":
    unittest.main()
