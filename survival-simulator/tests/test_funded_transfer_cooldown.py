"""Funded lineage transfers obey colony spacing instead of growth cooldowns."""

from types import SimpleNamespace
import unittest

import test_harvest
from test_expert_policy import agent_state


class FundedTransferCooldownTests(unittest.TestCase):
    setUp = test_harvest.GlobalReproductionTests.setUp
    plan = test_harvest.GlobalReproductionTests.plan
    add_food = test_harvest.GlobalReproductionTests.add_food
    observe_metabolism = test_harvest.GlobalReproductionTests.observe_metabolism

    def apply_birth(self, states, parent_id, now):
        action = SimpleNamespace(agent_id=parent_id, spawn_agent=True,
                                 move_distance=0., turn_angle=0.)
        self.population.remember_actions([action], now)
        self.harvest.active = True
        self.harvest.remember_actions([action], now)
        next(state for state in states if state["agent_id"] == parent_id)["energy"] -= 100.
        states.append(agent_state(agent_id=max(state["agent_id"] for state in states) + 1,
                                  age=0., energy=75.))

    @staticmethod
    def advance(states, seconds, senescent_ids):
        for _ in range(round(seconds * 10)):
            for state in states:
                state["age"] += .1
                state["energy"] -= .1 + (state["age"] * .01
                    if state["agent_id"] in senescent_ids else 0.)

    def test_confirmed_parent_transfers_again_after_one_second_before_energy_burns(self):
        states = [agent_state(agent_id=0, age=100., energy=250.)]
        self.observe_metabolism(states, {0})
        hints = self.plan(states, 2700.)
        self.assertTrue(hints[0].allowed)
        self.apply_birth(states, 0, 2700.)
        self.advance(states, .5, {0})
        self.assertFalse(any(h.allowed for h in self.plan(states, 2700.5).values()))
        self.assertEqual(self.harvest.population_plan["reason"], "spacing generations")
        self.advance(states, .5, {0})
        hints = self.plan(states, 2701.)
        self.assertGreater(states[0]["energy"], 107.)
        self.assertFalse(self.population.reproduction_hint(0, 2701., 350.).allowed)
        self.assertTrue(hints[0].allowed)
        self.assertTrue(hints[0].preserve_lineage)
        self.assertEqual(hints[0].minimum_energy_reserve, 5.)
        self.assertEqual(self.harvest.planned_birth_interval, 1.)

    def test_transfer_stops_when_the_actual_young_target_is_met(self):
        states = [agent_state(agent_id=0, age=100., energy=250.)]
        self.observe_metabolism(states, {0})
        self.plan(states, 2700.)
        self.apply_birth(states, 0, 2700.)
        self.advance(states, 1., {0})
        self.assertTrue(self.plan(states, 2701.)[0].allowed)
        self.apply_birth(states, 0, 2701.)
        self.advance(states, 1., {0})
        states[0]["energy"] = 300.
        hints = self.plan(states, 2702.)
        self.assertEqual(self.harvest.population_plan["healthy_young"], 2)
        self.assertEqual(self.harvest.population_plan["missing_healthy_successors"], 0)
        self.assertFalse(self.harvest.population_plan["funded_successor_transfer"])
        self.assertFalse(any(h.allowed for h in hints.values()))

    def test_ordinary_young_parent_still_obeys_its_cooldown(self):
        states = [agent_state(agent_id=0, age=10., energy=400.)]
        self.add_food(4)
        self.assertTrue(self.plan(states)[0].allowed)
        self.apply_birth(states, 0, 0.)
        self.advance(states, 3., set())
        states[0]["energy"] = 400.
        hints = self.plan(states, 3.)
        self.assertFalse(self.population.reproduction_hint(0, 3., 350.).allowed)
        self.assertFalse(self.harvest.population_plan["funded_successor_transfer"])
        self.assertFalse(hints[0].allowed)


if __name__ == "__main__":
    unittest.main()
