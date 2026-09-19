import contextlib
import io
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from scripts.playback import PlaybackControls


class PlaybackTests(unittest.TestCase):
    def test_each_speed_runs_every_fixed_tick_at_requested_rate(self):
        for speed in PlaybackControls.SPEEDS:
            control = PlaybackControls(speed)
            steps = 0
            for _ in range(1000):
                control.advance(.01)
                while control.consume_step(.1):
                    steps += 1
            self.assertEqual(steps, 100 * speed)

    def test_slowdown_clears_catchup_work_and_long_pause_is_bounded(self):
        control = PlaybackControls(20)
        control.advance(60.)
        self.assertEqual(control.pending_seconds, 5.)
        control.select(1)
        self.assertFalse(control.consume_step(.1))
        control.advance(.1)
        self.assertTrue(control.consume_step(.1))
        self.assertFalse(control.consume_step(.1))

    def test_actual_speed_reports_compute_limit(self):
        control = PlaybackControls(20)
        for _ in range(11):
            control.advance(.1)
            control.consume_step(.1)
        self.assertLessEqual(control.actual_speed, 1.01)
        self.assertGreater(control.actual_speed, .7)

    def test_buttons_and_keyboard_select_all_requested_speeds(self):
        control = PlaybackControls()
        for speed, rect in control.buttons((1000, 600)):
            event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rect.center)
            self.assertTrue(control.handle_event(event, (1000, 600)))
            self.assertEqual(control.speed, speed)
        for key, speed in zip((pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5), control.SPEEDS):
            self.assertTrue(control.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key), (600, 400)))
            self.assertEqual(control.speed, speed)
        self.assertFalse(control.handle_event(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=1, pos=(50, 100)), (600, 400)))


if __name__ == '__main__':
    unittest.main()
