import copy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from src.utils import map_renderer
from src.utils.map_renderer import (
    BELIEF_RAMP, BELIEF_SIGMA_CELLS, MapRenderer, draw_comparison, panel_layout,
)


def snapshot_fixture():
    return {
        "sim_time": 20,
        "groups": [
            {"group_id": index, "anchored": index == 0,
             "world_size": [1600, 1200] if index == 0 else None,
             "frame_revision": 0,
             "trees": [{"position": [45, 80], "last_seen": 19}],
             "edges": [{"start": [-60, -50], "end": [150, -50], "last_seen": 10, "sightings": 2}],
             "biomes": [{"position": [30, 50], "biome": "forest", "last_seen": 20, "uncertainty": 3}]}
            for index in range(5)
        ],
        "agents": [
            {"agent_id": index, "group_id": index, "position": [30, 50],
             "heading": index / 2, "uncertainty": 10, "role": "scout"}
            for index in range(5)
        ] + [{"agent_id": 10, "group_id": 1, "position": [90, 10],
              "heading": 1, "uncertainty": 20, "role": "resident"}],
        "links": [{"observer_id": 1, "target_id": 10, "last_seen": 8,
                   "first_seen": 2, "count": 12, "distance": 60, "angle": 0,
                   "relative_heading": 0}],
    }


def belief_fixture(tracks):
    snapshot = snapshot_fixture()
    snapshot["groups"][0]["predator_belief"] = dict(enabled=True, sim_time=20, tracks=tracks,
                                                    threats={})
    return snapshot


def track(position, spread=8., unseen=0., count=48, track_id=1):
    return dict(track_id=track_id, group_id=0, mean=list(position), spread=spread,
                particles=[list(position)] * count, weights=[1. / count] * count,
                seconds_unseen=unseen, heading=.4, resting_probability=0.,
                first_seen=0., last_seen=20., sightings=3, observers=[0])


class PredatorBeliefRampTests(unittest.TestCase):
    """Lock in the ramp properties the palette validator checked."""

    @staticmethod
    def _oklab_l(color):
        linear = [(value / 255) / 12.92 if value / 255 <= .04045
                  else (((value / 255) + .055) / 1.055) ** 2.4 for value in color]
        red, green, blue = linear
        long = .4122214708 * red + .5363325363 * green + .0514459929 * blue
        medium = .2119034982 * red + .6806995451 * green + .1073969566 * blue
        short = .0883024619 * red + .2817188376 * green + .6299787005 * blue
        roots = [value ** (1 / 3) if value > 0 else -((-value) ** (1 / 3))
                 for value in (long, medium, short)]
        return .2104542553 * roots[0] + .7936177850 * roots[1] - .0040720468 * roots[2]

    def test_ramp_is_one_hue_and_monotone_in_lightness(self):
        # A sequential ramp must carry magnitude in lightness, so it survives
        # colour-blind vision and greyscale. A rainbow would not.
        lightness = [self._oklab_l(color) for color in BELIEF_RAMP]
        self.assertEqual(lightness, sorted(lightness))
        gaps = [b - a for a, b in zip(lightness, lightness[1:])]
        self.assertTrue(all(gap >= .06 for gap in gaps), gaps)
        for red, green, blue in BELIEF_RAMP:
            self.assertGreater(red, green)
            self.assertGreater(red, blue)


class PredatorBeliefRenderTests(unittest.TestCase):
    def setUp(self):
        self.renderer = MapRenderer()
        self.surface = pygame.Surface((900, 900))

    def _rasterizer(self, snapshot):
        """Capture the real panel rect and view the renderer itself derives.

        Recomputing that geometry in the test would only assert that two
        copies of the same arithmetic agree.
        """
        captured = {}
        original = MapRenderer._belief_density

        def record(renderer, plot, view, tracks):
            captured["args"] = (plot, view)
            return original(renderer, plot, view, tracks)

        with patch.object(MapRenderer, "_belief_density", record):
            self.renderer.draw(self.surface, snapshot)
        plot, view = captured["args"]
        return lambda tracks: original(self.renderer, plot, view, tracks)

    def test_density_is_normalised_so_a_pinpointed_track_reaches_one(self):
        snapshot = belief_fixture([track((30, 50))])
        density = self._rasterizer(snapshot)(snapshot["groups"][0]["predator_belief"]["tracks"])
        self.assertIsNotNone(density)
        # All mass on one cell, blurred by the same kernel the reference uses,
        # so the normalised peak is 1.0 by construction.
        self.assertAlmostEqual(float(density.max()), 1., places=2)
        self.assertGreaterEqual(float(density.min()), 0.)

    def test_a_spread_belief_stays_dimmer_than_a_pinpointed_one(self):
        rasterize = self._rasterizer(belief_fixture([track((30, 50))]))
        tight = rasterize([track((30, 50))])
        scattered = rasterize([dict(track((30, 50)),
                                    particles=[[30 + index * 9, 50 + index * 7]
                                               for index in range(48)])])
        self.assertLess(float(scattered.max()), float(tight.max()))

    def test_particles_outside_the_panel_are_dropped_not_clamped(self):
        rasterize = self._rasterizer(belief_fixture([track((30, 50))]))
        self.assertIsNone(rasterize([track((1e7, 1e7))]))

    def test_belief_renders_toggles_and_leaves_the_snapshot_untouched(self):
        snapshot = belief_fixture([track((30, 50), unseen=0.), track((60, 20), unseen=7., track_id=2)])
        original = copy.deepcopy(snapshot)
        self.renderer.draw(self.surface, snapshot)
        painted = pygame.surfarray.array3d(self.surface).astype(int).sum()
        self.renderer.show_predator_belief = False
        blank = pygame.Surface((900, 900))
        self.renderer.draw(blank, snapshot)
        self.assertNotEqual(painted, pygame.surfarray.array3d(blank).astype(int).sum())
        self.assertEqual(snapshot, original)

    def test_a_track_without_particles_or_weights_still_renders(self):
        # Older snapshots, and any future payload change, must not crash the map.
        bare = dict(track((30, 50)))
        bare.pop("weights")
        bare["particles"] = [[30, 50]] * 4
        snapshot = belief_fixture([bare])
        self.renderer.draw(self.surface, snapshot)
        empty = belief_fixture([dict(track((30, 50)), particles=[], weights=[])])
        self.renderer.draw(self.surface, empty)


class MapRendererTests(unittest.TestCase):

    def test_estimated_biomes_render_independently_toggle_and_zoom_without_large_allocations(self):
        snapshot = snapshot_fixture()
        snapshot["groups"] = snapshot["groups"][:1]
        snapshot["agents"] = snapshot["agents"][:1]
        snapshot["groups"][0]["biome_estimate"] = dict(
            fingerprint="fixture", bounds=[0, 0, 1600, 1200], labels=[[0, 1], [0, -1]],
            confidence=[[1, .7], [.5, 0]], palette=["forest", "river"], sites=[],
            land_sample_agreement=.9,
            borders=[dict(start=[800, 0], end=[800, 1200], kind="land", confidence=.8)])
        original = copy.deepcopy(snapshot)
        renderer = MapRenderer()
        surface = pygame.Surface((800, 900))
        renderer.draw(surface, snapshot)
        shown = pygame.image.tobytes(surface, "RGB")
        renderer.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b))
        renderer.draw(surface, snapshot)
        self.assertNotEqual(shown, pygame.image.tobytes(surface, "RGB"))
        renderer.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b))
        renderer.views[0].zoom = 20
        with patch("pygame.transform.scale", wraps=pygame.transform.scale) as scale:
            renderer.draw(surface, snapshot)
            for call in scale.call_args_list:
                self.assertLessEqual(max(call.args[1]), 1200)
        self.assertEqual(snapshot, original)

    def test_five_clusters_have_separate_readable_panels(self):
        container = pygame.Rect(800, 80, 780, 740)
        panels = panel_layout(container, 5)
        self.assertEqual(len(panels), 5)
        for index, panel in enumerate(panels):
            self.assertTrue(container.contains(panel))
            self.assertGreaterEqual(panel.width, 350)
            self.assertGreaterEqual(panel.height, 220)
            for other in panels[index + 1:]:
                self.assertFalse(panel.colliderect(other))

    def test_internal_render_is_independent_of_actual_pixels_and_preserves_snapshot(self):
        snapshot = snapshot_fixture()
        before = copy.deepcopy(snapshot)
        images = []
        for color in [(200, 50, 30), (20, 50, 200)]:
            actual = pygame.Surface((640, 480))
            actual.fill(color)
            surface = pygame.Surface((1600, 900))
            draw_comparison(surface, actual, snapshot, MapRenderer(), "comparison")
            images.append(pygame.image.tobytes(surface.subsurface((801, 0, 799, 900)), "RGB"))
        self.assertEqual(images[0], images[1])
        self.assertEqual(snapshot, before)

    def test_internal_panel_depends_on_the_snapshot_alone(self):
        """The estimated view must be a pure function of observation-derived data.

        Rendering the same snapshot beside two completely different worlds has
        to give identical pixels on the right. If any privileged value ever
        reached the renderer, these two would diverge.
        """
        snapshot = belief_fixture([track((30, 50), unseen=1.2),
                                   track((70, 15), spread=40., unseen=6., track_id=2)])
        before = copy.deepcopy(snapshot)
        panels = []
        for fill, size in (((200, 50, 30), (640, 480)), ((20, 50, 200), (900, 300))):
            actual = pygame.Surface(size)
            actual.fill(fill)
            surface = pygame.Surface((1600, 900))
            draw_comparison(surface, actual, snapshot, MapRenderer(), "comparison")
            panels.append(pygame.image.tobytes(surface.subsurface((801, 0, 799, 900)), "RGB"))
        self.assertEqual(panels[0], panels[1])
        self.assertEqual(snapshot, before)

    def test_the_renderer_module_names_no_simulator_state(self):
        """A grep-level guard against the easy regression.

        The estimated panel is only trustworthy while the renderer cannot see
        the environment at all, so importing or touching one has to fail here
        rather than quietly ship a privileged pixel.
        """
        source = Path(map_renderer.__file__).read_text(encoding="utf-8")
        # Comments and the module docstring legitimately discuss the simulator;
        # only executable references would leak it.
        code = " ".join(line.split("#")[0] for line in source.splitlines())
        for forbidden in ("src.elements", "src.core", "SimulationCore",
                          "biome_map", "agents_dict", ".predators"):
            self.assertNotIn(forbidden, code, f"renderer references {forbidden}")

    def test_zoom_pan_and_frame_reset_are_local_to_one_cluster(self):
        snapshot = snapshot_fixture()
        renderer = MapRenderer()
        surface = pygame.Surface((800, 900))
        renderer.draw(surface, snapshot)
        view = renderer.views[1]
        point = view.rect.center
        renderer.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=point, rel=(0, 0)))
        renderer.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, y=1))
        self.assertGreater(view.zoom, 1)
        self.assertEqual(renderer.views[2].zoom, 1)
        renderer.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=point, button=1))
        renderer.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=point, rel=(20, 10)))
        self.assertEqual(tuple(view.pan), (20, 10))
        snapshot["groups"][1]["frame_revision"] += 1
        renderer.draw(surface, snapshot)
        self.assertEqual(view.zoom, 1)
        self.assertEqual(tuple(view.pan), (0, 0))

    def test_empty_snapshot_renders_without_simulator(self):
        renderer = MapRenderer()
        renderer.draw(pygame.Surface((800, 900)), {})
        self.assertEqual(renderer.views, {})


if __name__ == "__main__":
    unittest.main()
