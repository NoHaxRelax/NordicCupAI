import contextlib
import io
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from src.utils.playback import PlaybackControls


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

    def test_graphical_speed_keeps_action_order_and_stops_inside_batch(self):
        from local_playground import local_simulation

        class Clock:
            def tick(self, fps):
                return 50

        class Policy:
            def __init__(self):
                self.config = SimpleNamespace(harvest=SimpleNamespace(enabled=True))
                self.planner = SimpleNamespace()
                self.harvest = SimpleNamespace(active=False)
                self.times = []

            def actions_for_step(self, observations, sim_time):
                self.times.append(sim_time)
                return [SimpleNamespace(agent_id=1, decision=len(self.times))]

            def reset(self):
                pass

        class Sim:
            dt = .1
            env_width = 400
            env_height = 300

            def __init__(self, extinct=False):
                self.env = SimpleNamespace(time=0., score=0., agents=[1], draw=lambda surface: None)
                self.received = []
                self.extinct = extinct

            def step(self, actions):
                self.received.append([a.decision for _, a in actions])
                self.env.time = len(self.received) * self.dt
                if self.extinct and len(self.received) == 3:
                    self.env.agents = []
                return dict(observations=[], sim_time=self.env.time, score=0., num_agents=len(self.env.agents))

        for extinct, expected in ((False, 5), (True, 3)):
            for speed in PlaybackControls.SPEEDS:
                with self.subTest(speed=speed, extinct=extinct):
                    sim, policy = Sim(extinct), Policy()
                    with patch("local_playground.SimulationCore", return_value=sim), \
                         patch("local_playground.ExpertPolicy", return_value=policy), \
                         patch("local_playground.draw_planner_overlay"), \
                         patch("local_playground.pygame.time.Clock", return_value=Clock()), \
                         patch("local_playground.pygame.event.get", return_value=[]), \
                         contextlib.redirect_stdout(io.StringIO()):
                        local_simulation(seed=42, diagnostics=False, map_view=False,
                                         max_seconds=.5, speed=speed)
                    self.assertEqual(sim.received, [[]] + [[n] for n in range(1, expected)])
                    self.assertEqual(policy.times, [n * .1 for n in range(1, expected + 1)])


if __name__ == "__main__":
    unittest.main()
