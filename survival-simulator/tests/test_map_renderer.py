import copy
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from src.utils.map_renderer import MapRenderer, TRAP_COLORS, draw_comparison, panel_layout


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


class MapRendererTests(unittest.TestCase):
    def test_trap_types_render_toggle_and_mark_stale_without_mutating_snapshot(self):
        snapshot = snapshot_fixture()
        snapshot["groups"] = snapshot["groups"][:1]
        snapshot["agents"] = snapshot["agents"][:1]
        estimate = dict(stale=False, sites=[dict(kind=kind, bait=[400 + 350 * i, 500],
            predator_side=[360 + 350 * i, 500]) for i, kind in enumerate(("wall", "slot", "shelter"))])
        snapshot["groups"][0]["trap_estimate"] = estimate
        original = copy.deepcopy(snapshot)
        renderer = MapRenderer()
        surface = pygame.Surface((800, 900))
        renderer.draw(surface, snapshot)
        for site in estimate["sites"]:
            x, y = renderer.views[0].pixel(site["bait"])
            y -= 10 if site["kind"] == "shelter" else 12
            self.assertEqual(surface.get_at((x, y))[:3], TRAP_COLORS[site["kind"]])
        shown = pygame.image.tobytes(surface, "RGB")
        renderer.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_t))
        renderer.draw(surface, snapshot)
        self.assertNotEqual(shown, pygame.image.tobytes(surface, "RGB"))
        self.assertEqual(snapshot, original)
        renderer.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_t))
        estimate["stale"] = True
        renderer.draw(surface, snapshot)
        self.assertNotEqual(shown, pygame.image.tobytes(surface, "RGB"))

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
