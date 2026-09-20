"""Regression checks for rendering on smaller/resized display surfaces.

python -m unittest discover -s survival-simulator/scripts -p test_predator_corner_demo.py
"""
import os
from types import SimpleNamespace
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import predator_corner_demo as demo


class CornerRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        demo.pygame.display.init()
        cls.trial = demo.Trial(SimpleNamespace(
            count=64, seed=1, corner="top-left", start="touching",
            biome="grassland", seconds=6,
        ))
        cls.renderer = demo.Renderer()

    @classmethod
    def tearDownClass(cls):
        demo.pygame.quit()

    def assert_scene_and_sidebar_visible(self, surface):
        pixels = demo.pygame.surfarray.array3d(surface).astype("int16")
        # The arena's gray boundary and green sidebar counter both remain
        # visible, so fitting cannot merely clip away the right/bottom area.
        # Smooth scaling blends pixels and can slightly round even flat colors.
        walls = (abs(pixels - (83, 96, 109)) <= 8).all(axis=2)
        counter = (abs(pixels - demo.GREEN) <= 8).all(axis=2)
        self.assertTrue(walls.any(), "Arena boundary is missing")
        self.assertTrue(counter.any(), "Sidebar escape counter is missing")

    def test_small_and_different_aspect_surfaces(self):
        for size in ((640, 360), (800, 450), (400, 700), (1600, 400), (1120, 600)):
            with self.subTest(size=size):
                surface = demo.pygame.Surface(size)
                self.renderer.draw(surface, self.trial)
                self.assert_scene_and_sidebar_visible(surface)

    def test_display_resize_sequence(self):
        for size in ((1120, 600), (640, 360), (900, 500)):
            with self.subTest(size=size):
                demo.pygame.display.set_mode(size, demo.pygame.RESIZABLE)
                surface = demo.pygame.display.get_surface()
                self.renderer.draw(surface, self.trial)
                demo.pygame.display.flip()
                self.assert_scene_and_sidebar_visible(surface)

    def test_minimized_sized_targets_do_not_crash(self):
        for size in ((1, 1), (0, 0), (0, 600)):
            with self.subTest(size=size):
                self.renderer.draw(demo.pygame.Surface(size), self.trial)


if __name__ == "__main__":
    unittest.main()
