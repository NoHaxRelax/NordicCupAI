from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from src.utils.controllers.biome_estimator import BiomeInferenceConfig
from src.utils.controllers.world_estimator import BiomeSample, MapGroup
from test_biome_estimator import fixture


CLOCK = "src.utils.controllers.biome_estimator.perf_counter"


class BiomeScheduleTests(unittest.TestCase):
    def test_deterministic_schedule_ignores_measured_cpu_cost(self):
        fast, group_fast, poses_fast = fixture(adapt_to_wall_clock=False)
        slow, group_slow, poses_slow = fixture(adapt_to_wall_clock=False)
        with patch(CLOCK, side_effect=[0., .001]):
            fast.update({1: group_fast}, poses_fast, 100)
        with patch(CLOCK, side_effect=[0., 90.]):
            slow.update({1: group_slow}, poses_slow, 100)
        for _ in range(32):
            self.assertEqual(fast.next_fit, slow.next_fit)
            self.assertEqual(fast.interval, slow.interval)
            now = fast.next_fit
            fast.update({1: group_fast}, poses_fast, now)
            slow.update({1: group_slow}, poses_slow, now)
        self.assertEqual(slow.interval, 30)
        self.assertNotEqual(fast.average_fit_seconds, slow.average_fit_seconds)

    def start(self, **config):
        mapper, group, poses = fixture(**config)
        with patch(CLOCK, side_effect=[0., .01]):
            mapper.update({1: group}, poses, 100)
        return mapper, group, poses

    def mature(self):
        mapper, group, poses = self.start()
        while mapper.interval < mapper.config.max_refit_interval_seconds:
            mapper.update({1: group}, poses, mapper.next_fit)
        return mapper, group, poses

    def test_two_second_warmup_checks_then_linear_increments_capped_at_30(self):
        mapper, group, poses = self.start()
        self.assertEqual(mapper.aligned_at, 100)
        self.assertEqual(mapper.warmup_until, 130)
        for now in range(102, 130, 2):
            mapper.update({1: group}, poses, now)
            self.assertEqual(mapper.interval, 2)
            self.assertEqual(mapper.next_fit, now + 2)
        for now, interval, due in ((130, 4, 134), (134, 6, 140), (140, 8, 148), (148, 10, 158)):
            mapper.update({1: group}, poses, now)
            self.assertEqual((mapper.interval, mapper.next_fit), (interval, due))
            mapper.update({1: group}, poses, now)  # Same-time retries cannot increase the interval twice.
            self.assertEqual(mapper.next_fit, due)
        for _ in range(100):
            now = mapper.next_fit
            mapper.update({1: group}, poses, now)
            self.assertLessEqual(mapper.next_fit - now, 30)
        self.assertEqual(mapper.interval, 30)
        # Rechecking identical points never pretends they are new validation.
        status = mapper.snapshot(1)["refit_schedule"]
        self.assertEqual(status["fit_count"], 1)
        self.assertIsNone(status["validation_agreement"])
        self.assertEqual(status["validation_sample_count"], 0)

    def test_explicit_custom_schedule_preserves_legacy_cadence_and_cap(self):
        mapper, group, poses = self.start(
            refit_interval_seconds=5, refit_interval_increment_seconds=5,
            max_refit_interval_seconds=500, fit_compute_budget_fraction=.05)
        for now in range(105, 130, 5):
            mapper.update({1: group}, poses, now)
            self.assertEqual(mapper.interval, 5)
        for now, interval in ((130, 10), (140, 15), (155, 20)):
            mapper.update({1: group}, poses, now)
            self.assertEqual(mapper.interval, interval)
        for _ in range(100):
            mapper.update({1: group}, poses, mapper.next_fit)
        self.assertEqual(mapper.interval, 500)
        self.assertEqual(mapper.fit_count, 1)

    def test_representative_fit_cost_allows_two_second_refits_without_duplicate_work(self):
        mapper, group, poses = fixture()
        # A measured 185ms fit uses 9.25% of a two-second simulation interval.
        with patch(CLOCK, side_effect=[0., .185, 1., 1.185]):
            mapper.update({1: group}, poses, 0)
            self.assertEqual(mapper.next_fit, 2)
            group.biomes[(0, 0)].position[1] += 1
            mapper.update({1: group}, poses, 1.9)
            self.assertEqual(mapper.fit_count, 1)
            mapper.update({1: group}, poses, 2)
            self.assertEqual(mapper.snapshot(1)["updated_at"], 2)
            self.assertEqual(mapper.fit_count, 2)
            self.assertEqual(mapper.interval, 2)
            for now in range(4, 32, 2):
                mapper.update({1: group}, poses, now)
        self.assertEqual(mapper.fit_count, 2)
        self.assertEqual(mapper.reason, "unchanged evidence; fit reused")

    def test_agreeing_new_samples_are_fitted_on_schedule_and_allow_backoff(self):
        mapper, group, poses = self.start()
        mapper.update({1: group}, poses, 130)
        for sample in group.biomes.values():
            sample.position[1] += 1
        mapper.update({1: group}, poses, 132)
        self.assertEqual(mapper.fit_count, 1)
        self.assertEqual(mapper.next_fit, 134)
        with patch(CLOCK, side_effect=[0., .01]):
            mapper.update({1: group}, poses, 134)
        self.assertEqual(mapper.fit_count, 2)
        self.assertEqual(mapper.snapshot(1)["updated_at"], 134)
        self.assertEqual(mapper.validation_agreement, 1)
        self.assertEqual(mapper.interval, 6)

    def test_contradictions_interrupt_a_long_interval_and_restart_learning(self):
        mapper, group, poses = self.mature()
        old_due = mapper.next_fit
        now = mapper.last_time + 2
        for sample in list(group.biomes.values())[:12]:
            sample.biome = "desert"
        with patch(CLOCK, side_effect=[0., .01]):
            mapper.update({1: group}, poses, now)
        self.assertLess(now, old_due)
        self.assertEqual(mapper.snapshot(1)["updated_at"], now)
        self.assertEqual(mapper.validation_agreement, 0)
        self.assertEqual(mapper.validation_sample_count, 12)
        self.assertEqual(mapper.interval, 2)
        self.assertEqual(mapper.warmup_until, now + 30)

    def test_one_new_biome_is_enough_but_uncertain_samples_do_not_trigger(self):
        mapper, group, poses = self.mature()
        now, old_due = mapper.last_time + 2, mapper.next_fit
        group.biomes[(999, 0)] = BiomeSample(np.array([800., 400.]), "river", now, 30)
        mapper.update({1: group}, poses, now)
        self.assertEqual(mapper.next_fit, old_due)
        self.assertEqual(mapper.fit_count, 1)
        group.biomes[(999, 0)].uncertainty = 1
        with patch(CLOCK, side_effect=[0., .01]):
            mapper.update({1: group}, poses, now + 2)
        self.assertEqual(mapper.fit_count, 2)
        self.assertEqual(mapper.interval, 2)
        self.assertEqual(mapper.warmup_until, now + 32)

    def test_expensive_fits_spread_out_work_and_respect_the_cap(self):
        mapper, group, poses = fixture()
        with patch(CLOCK, side_effect=[0., 2.]):
            mapper.update({1: group}, poses, 0)
        self.assertEqual(mapper.interval, 20)  # 2 seconds / 10% compute budget.
        self.assertEqual(mapper.next_fit, 20)
        mapper.reset()
        with patch(CLOCK, side_effect=[0., 100.]):
            mapper.update({1: group}, poses, 0)
        self.assertEqual(mapper.interval, 30)
        self.assertEqual(mapper.snapshot(1)["refit_schedule"]["last_fit_ms"], 100000)

    def test_early_refits_still_respect_compute_budget(self):
        mapper, group, poses = fixture()
        with patch(CLOCK, side_effect=[0., 2.]):
            mapper.update({1: group}, poses, 0)
        group.biomes[(999, 0)] = BiomeSample(np.array([800., 400.]), "river", 5, 1)
        mapper.update({1: group}, poses, 5)
        self.assertEqual(mapper.fit_count, 1)
        self.assertEqual(mapper.next_fit, 20)

    def test_newborn_disconnection_keeps_schedule_but_new_frame_restarts_it(self):
        mapper, group, poses = self.mature()
        now, old_due = mapper.last_time + 2, mapper.next_fit
        poses[2] = SimpleNamespace(group_id=2)
        mapper.update({1: group, 2: MapGroup(2)}, poses, now)
        self.assertEqual(mapper.next_fit, old_due)
        poses[2].group_id = 1
        mapper.update({1: group}, poses, now + 1)
        self.assertEqual(mapper.next_fit, old_due)
        group.frame_revision += 1
        with patch(CLOCK, side_effect=[0., .01]):
            mapper.update({1: group}, poses, now + 2)
        self.assertEqual(mapper.aligned_at, now + 2)
        self.assertEqual(mapper.interval, 2)
        group.anchored = False
        mapper.update({1: group}, poses, now + 3)
        self.assertIsNone(mapper.snapshot(1))
        group.anchored = True
        with patch(CLOCK, side_effect=[0., .01]):
            mapper.update({1: group}, poses, now + 4)
        self.assertEqual(mapper.aligned_at, now + 4)
        mapper.update({1: group}, poses, now + 4.1)
        self.assertEqual(mapper.snapshot(1)["updated_at"], now + 4)

    def test_new_dimensions_trigger_a_refit_and_clock_rollback_resets_schedule(self):
        mapper, group, poses = self.mature()
        now = mapper.last_time + 2
        group.world_size = (1200, 1000)
        with patch(CLOCK, side_effect=[0., .01]):
            mapper.update({1: group}, poses, now)
        self.assertEqual(mapper.snapshot(1)["bounds"], [0, 0, 1200, 1000])
        self.assertEqual(mapper.interval, 2)
        with patch(CLOCK, side_effect=[0., .01]):
            mapper.update({1: group}, poses, 0)
        self.assertEqual(mapper.aligned_at, 0)
        self.assertEqual(mapper.fit_count, 1)
        self.assertEqual(mapper.next_fit, 2)

    def test_cap_cannot_be_below_initial_interval(self):
        with self.assertRaises(ValidationError):
            BiomeInferenceConfig(refit_interval_seconds=10, max_refit_interval_seconds=5)


if __name__ == "__main__":
    unittest.main()
