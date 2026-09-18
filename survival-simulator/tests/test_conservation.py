import unittest

from pydantic import ValidationError

from src.utils.controllers.conservation import ConservationConfig, ConservationSchedule


class ConservationTests(unittest.TestCase):
    def test_dial_is_continuous_and_bounded(self):
        schedule = ConservationSchedule()
        for now, fraction, duty, interval in ((0., 0., 1., 6.), (300., 0., 1., 6.),
                                              (1050., .5, .65, 12.),
                                              (1800., 1., .3, 18.),
                                              (3000., 1., .3, 18.)):
            with self.subTest(now=now):
                self.assertAlmostEqual(schedule.fraction(now), fraction)
                self.assertAlmostEqual(schedule.scouting_fraction(now), duty)
                self.assertAlmostEqual(schedule.scan_interval(now), interval)

    def test_early_scouting_is_continuous_and_late_duty_is_staggered(self):
        schedule = ConservationSchedule()
        self.assertTrue(all(schedule.should_scout(agent, tick / 10.)
                            for agent in range(10) for tick in range(100)))
        for agent in (0, 1, 27):
            active = sum(schedule.should_scout(agent, 1800. + tick / 10.)
                         for tick in range(100))
            self.assertEqual(active, 30)
        self.assertNotEqual(schedule.should_scout(0, 1800.), schedule.should_scout(20, 1800.))

    def test_disabled_exactly_preserves_legacy_scan_and_full_scouting(self):
        schedule = ConservationSchedule(ConservationConfig(enabled=False))
        for agent in (0, 1, 27):
            for tick in range(100):
                now = 2990. + tick / 10.
                self.assertEqual(schedule.scanning(agent, now, .1), (now + agent * .37) % 6. < .8)
                self.assertTrue(schedule.should_scout(agent, now))
        self.assertEqual(schedule.fraction(3000.), 0.)
        self.assertEqual(schedule.scan_interval(3000.), 6.)
        self.assertFalse(schedule.scans)

    def test_scan_runs_eight_consecutive_ticks_despite_changing_interval(self):
        schedule = ConservationSchedule(ConservationConfig(start_seconds=0., full_seconds=1.))
        self.assertEqual([schedule.scanning(0, tick / 10., .1) for tick in range(10)],
                         [True] * 8 + [False] * 2)
        # The first deadline was set at t=0, not moved every tick as the
        # requested scan period grows from6s to18s.
        self.assertFalse(schedule.scanning(0, 5.9, .1))
        self.assertTrue(schedule.scanning(0, 6., .1))
        self.assertEqual(schedule.scans[0].next_scan, 24.)

    def test_duplicate_time_calls_do_not_shorten_scan(self):
        schedule = ConservationSchedule()
        for tick in range(8):
            now = tick / 10.
            self.assertTrue(schedule.scanning(0, now, .1))
            self.assertTrue(schedule.scanning(0, now, .1))
        self.assertFalse(schedule.scanning(0, .8, .1))

    def test_interrupted_scan_restarts_a_complete_sweep(self):
        schedule = ConservationSchedule()
        self.assertTrue(schedule.scanning(0, 0., .1))
        self.assertTrue(schedule.scanning(0, .1, .1))
        self.assertTrue(all(schedule.scanning(0, 3. + tick / 10., .1) for tick in range(8)))
        self.assertFalse(schedule.scanning(0, 3.8, .1))
        self.assertEqual(schedule.scans[0].next_scan, 9.)

    def test_scan_deadlines_are_staggered_and_can_be_pruned(self):
        schedule = ConservationSchedule()
        for agent in range(8):
            schedule.scanning(agent, 0., .1)
        self.assertEqual(len({scan.next_scan for scan in schedule.scans.values()}), 8)
        schedule.prune([2, 4])
        self.assertEqual(set(schedule.scans), {2, 4})
        schedule.reset()
        self.assertFalse(schedule.scans)

    def test_invalid_config_is_rejected(self):
        for kwargs in ({"start_seconds": 1800.}, {"late_scouting_fraction": 1.1},
                       {"scouting_cycle_seconds": 0.}, {"late_scan_interval_seconds": 3.},
                       {"full_seconds": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                ConservationConfig(**kwargs)

    def test_invalid_scan_clock_is_rejected(self):
        schedule = ConservationSchedule()
        for now, dt in ((0., 0.), (0., float("nan")), (float("inf"), .1)):
            with self.subTest(now=now, dt=dt), self.assertRaises(ValueError):
                schedule.scanning(1, now, dt)


if __name__ == "__main__":
    unittest.main()
