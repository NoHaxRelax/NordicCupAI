import math
from types import SimpleNamespace
import unittest

from src.utils.controllers.metabolism import MetabolismTracker


def state(agent_id=1, age=10., energy=200., **kwargs):
    return dict(agent_id=agent_id, age=age, energy=energy,
                speed=kwargs.get("speed", 10.), sprint_speed=kwargs.get("sprint_speed", 20.),
                max_energy=kwargs.get("max_energy", 500.))


def action(agent_id=1, distance=0., turn=0., birth=False):
    return SimpleNamespace(agent_id=agent_id, move_distance=distance,
                           turn_angle=turn, spawn_agent=birth)


class MetabolismTests(unittest.TestCase):
    def test_young_walking_cost_is_removed_from_passive_drain(self):
        tracker = MetabolismTracker()
        tracker.update([state()], 0.)
        tracker.remember_actions([action(distance=10.)])
        rates = tracker.update([state(age=10.1, energy=199.4)], .1)
        self.assertAlmostEqual(rates[1], 1.2)
        self.assertFalse(tracker.history[1].aging)

    def test_turning_and_sprinting_costs_are_removed(self):
        tracker = MetabolismTracker()
        tracker.update([state(age=70.)], 0.)
        tracker.remember_actions([action(distance=20., turn=math.pi)])
        rates = tracker.update([state(age=70.1, energy=193.9)], .1)
        self.assertAlmostEqual(rates[1], 1.2)

    def test_engine_clamps_sprint_when_energy_is_below_twenty_percent(self):
        tracker = MetabolismTracker()
        tracker.update([state(age=70., energy=90.)], 0.)
        tracker.remember_actions([action(distance=30., turn=2 * math.pi)])
        rates = tracker.update([state(age=70.1, energy=88.9)], .1)
        self.assertAlmostEqual(rates[1], 1.2)

    def test_birth_is_deducted_after_action_costs(self):
        tracker = MetabolismTracker()
        tracker.update([state(age=70.)], 0.)
        tracker.remember_actions([action(distance=10., birth=True)])
        rates = tracker.update([state(age=70.1, energy=99.4)], .1)
        self.assertAlmostEqual(rates[1], 1.2)
        self.assertFalse(tracker.history[1].aging)

    def test_failed_birth_request_is_not_deducted(self):
        tracker = MetabolismTracker()
        tracker.update([state(age=70., energy=100.4)], 0.)
        tracker.remember_actions([action(distance=10., birth=True)])
        rates = tracker.update([state(age=70.1, energy=99.8)], .1)
        self.assertAlmostEqual(rates[1], 1.2)

    def test_aging_onset_is_detected_without_private_lifespan(self):
        tracker = MetabolismTracker()
        tracker.update([state(age=80.)], 0.)
        tracker.remember_actions([action(distance=10.)])
        first = tracker.update([state(age=80.1, energy=198.599)], .1)
        self.assertAlmostEqual(first[1], 9.21)
        self.assertFalse(tracker.history[1].aging)
        tracker.remember_actions([action()])
        second = tracker.update([state(age=80.2, energy=197.697)], .2)
        self.assertTrue(tracker.history[1].aging)
        self.assertAlmostEqual(second[1], 9.22)

    def test_food_gain_and_energy_cap_cannot_clear_confirmed_aging(self):
        tracker = MetabolismTracker()
        tracker.update([state(age=80.)], 0.)
        for age, energy, time in ((80.1, 199.099, .1), (80.2, 198.197, .2)):
            tracker.remember_actions([action()])
            tracker.update([state(age=age, energy=energy)], time)
        tracker.remember_actions([action()])
        gain = tracker.update([state(age=80.3, energy=257.294)], .3)
        self.assertAlmostEqual(gain[1], 9.23)
        tracker.remember_actions([action()])
        capped = tracker.update([state(age=80.4, energy=500.)], .4)
        self.assertTrue(tracker.history[1].aging)
        self.assertAlmostEqual(capped[1], 9.24)

    def test_food_mask_is_not_negative_drain_or_evidence_of_youth(self):
        tracker = MetabolismTracker()
        tracker.update([state(age=90.)], 0.)
        tracker.remember_actions([action()])
        rates = tracker.update([state(age=90.1, energy=220.)], .1)
        self.assertGreater(rates[1], 1.2)
        self.assertIsNone(tracker.history[1].healthy_at_age)
        self.assertFalse(tracker.history[1].aging)

    def test_fresh_normal_upkeep_narrows_unknown_onset_interval(self):
        tracker = MetabolismTracker()
        fallback = tracker.update([state(age=90.)], 0.)
        self.assertAlmostEqual(fallback[1], 5.7)
        tracker.remember_actions([action()])
        measured = tracker.update([state(age=90.1, energy=199.9)], .1)
        self.assertAlmostEqual(measured[1], 1.2)
        tracker.remember_actions([action()])
        masked = tracker.update([state(age=90.2, energy=219.8)], .2)
        self.assertGreater(masked[1], 1.2)
        self.assertLess(masked[1], 1.3)

    def test_unchanged_age_only_updates_action_energy_bookkeeping(self):
        tracker = MetabolismTracker()
        tracker.update([state(age=80.)], 0.)
        tracker.remember_actions([action(distance=10.)])
        tracker.update([state(age=80., energy=199.5)], .1)
        self.assertEqual(tracker.history[1].positive_samples, 0)
        tracker.remember_actions([action(distance=10.)])
        # World time advanced .2 since the first state, but passive drain was
        # charged on only one .1-second increment of this agent's public age.
        tracker.update([state(age=80.1, energy=198.099)], .2)
        self.assertEqual(tracker.history[1].positive_samples, 1)
        self.assertAlmostEqual(tracker.rates[1], 9.21)

    def test_missing_actions_and_missing_frames_do_not_invent_aging(self):
        tracker = MetabolismTracker()
        tracker.update([state(age=80.)], 0.)
        tracker.update([state(age=80.1, energy=100.)], .1)
        self.assertEqual(tracker.history[1].positive_samples, 0)
        tracker.remember_actions([action()])
        tracker.update([state(age=80.3, energy=50.)], .3)
        self.assertEqual(tracker.history[1].positive_samples, 0)

    def test_duplicate_timestamp_preserves_pending_action(self):
        tracker = MetabolismTracker()
        initial = state(age=70.)
        tracker.update([initial], 0.)
        tracker.remember_actions([action(distance=10.)])
        tracker.update([initial], 0.)
        self.assertIn(1, tracker.pending_costs)
        self.assertAlmostEqual(tracker.update([state(age=70.1, energy=199.4)], .1)[1], 1.2)

    def test_dead_agents_pruned_and_reused_id_or_rewind_resets_history(self):
        tracker = MetabolismTracker()
        tracker.update([state(), state(agent_id=2)], 1.)
        tracker.remember_actions([action(), action(agent_id=2)])
        tracker.update([state(age=10.1, energy=199.9)], 1.1)
        self.assertEqual(set(tracker.history), {1})
        self.assertNotIn(2, tracker.pending_costs)
        tracker.history[1].aging = True
        tracker.update([state(age=0., energy=75.)], 1.2)
        self.assertFalse(tracker.history[1].aging)
        tracker.update([state(age=50.)], .5)
        self.assertIsNone(tracker.history[1].healthy_at_age)
        self.assertEqual(tracker.update([], .6), {})
        self.assertEqual(tracker.history, {})

    def test_nondefault_tick_uses_public_age_increment(self):
        tracker = MetabolismTracker()
        tracker.update([state(age=80.)], 0., dt=.2)
        tracker.remember_actions([action()])
        result = tracker.update([state(age=80.2, energy=198.998)], .2, dt=.2)
        self.assertAlmostEqual(result[1], 5.21)


if __name__ == "__main__":
    unittest.main()
