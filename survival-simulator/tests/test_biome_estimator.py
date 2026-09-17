import json
from types import SimpleNamespace
import unittest

import numpy as np

from src.utils.controllers.biome_estimator import (
    BIOMES, BiomeEstimator, BiomeInferenceConfig, fit_sites, voronoi_borders,
)
from src.utils.controllers.world_estimator import BiomeSample, MapGroup


def fixture(**config):
    group = MapGroup(1, anchored=True, world_size=(1000, 1000))
    for index, (x, y) in enumerate((x, y) for x in range(100, 901, 100) for y in range(100, 901, 100)):
        group.biomes[(index, 0)] = BiomeSample(np.array([x, y], dtype=float),
                                             "forest" if x < 500 else "desert", 0, 1)
    poses = {1: SimpleNamespace(group_id=1)}
    return BiomeEstimator(BiomeInferenceConfig(enabled=True, **config)), group, poses


class BiomeEstimatorTests(unittest.TestCase):
    def test_collapsed_warm_sites_separate_and_new_labels_respect_site_budget(self):
        rng = np.random.default_rng(5)
        points = rng.uniform(0, 1000, (400, 2))
        labels = np.where(points[:, 0] < 500, "forest", "desert")
        previous = [dict(position=[500., 500.], biome=label) for label in ("forest", "desert")]
        sites, classes, fit = fit_sites(points, labels, np.ones(400), (0, 0, 1000, 1000),
                                        BiomeInferenceConfig(), previous)
        self.assertGreater(fit, .97)
        self.assertGreater(np.linalg.norm(sites[0] - sites[1]), 100)
        previous = [dict(position=[100. + i * 20, 100.], biome="forest") for i in range(10)]
        sites, classes, _ = fit_sites(points, labels, np.ones(400), (0, 0, 1000, 1000),
                                     BiomeInferenceConfig(), previous)
        self.assertLessEqual(len(sites), 10)
        self.assertIn("desert", classes)

    def test_fits_diagonal_border_on_unseen_points(self):
        rng = np.random.default_rng(17)
        true_sites = np.array([[200, 250], [700, 650]])
        labels = np.array(["forest", "desert"])
        points = rng.uniform(0, 1000, (400, 2))
        truth = labels[np.argmin(((points[:, None] - true_sites) ** 2).sum(2), axis=1)]
        sites, classes, fit = fit_sites(points, truth, np.ones(400), (0, 0, 1000, 1000), BiomeInferenceConfig())
        query = rng.uniform(0, 1000, (2000, 2))
        expected = labels[np.argmin(((query[:, None] - true_sites) ** 2).sum(2), axis=1)]
        predicted = classes[np.argmin(((query[:, None] - sites) ** 2).sum(2), axis=1)]
        self.assertGreater(fit, .98)
        self.assertGreater(np.mean(expected == predicted), .98)

    def test_repeated_biome_labels_can_have_disconnected_regions(self):
        rng = np.random.default_rng(23)
        points = rng.uniform(0, 1000, (600, 2))
        labels = np.where((points[:, 0] < 325) | (points[:, 0] > 675), "forest", "desert")
        cfg = BiomeInferenceConfig(max_iterations=150, max_site_additions=6)
        sites, classes, fit = fit_sites(points, labels, np.ones(600), (0, 0, 1000, 1000), cfg)
        self.assertGreater(fit, .97)
        self.assertGreater(np.sum(classes == "forest"), 1)
        self.assertLessEqual(len(sites), 10)
        query = np.array([[100, 500], [500, 500], [900, 500]])
        np.testing.assert_array_equal(classes[np.argmin(((query[:, None] - sites) ** 2).sum(2), axis=1)],
                                      ["forest", "desert", "forest"])

    def test_exact_bisectors_clip_at_third_region_and_hide_same_label_seams(self):
        sites = np.array([[100., 200], [300, 200], [200, 400]])
        borders = voronoi_borders(sites, ["forest", "desert", "swamp"], (0, 0, 500, 500))
        self.assertEqual(len(borders), 3)
        first = next(border for border in borders if border[2:] == ("forest", "desert"))
        np.testing.assert_allclose([first[0][0], first[1][0]], [200, 200])
        self.assertAlmostEqual(max(first[0][1], first[1][1]), 275)
        for start, end, _, _ in borders:
            self.assertTrue(np.all(np.array([start, end]) >= -1e-8))
            self.assertTrue(np.all(np.array([start, end]) <= 500 + 1e-8))
        self.assertEqual(voronoi_borders(sites, ["forest"] * 3, (0, 0, 500, 500)), [])

    def test_requires_shared_absolute_coordinates_and_refits_at_bounded_intervals(self):
        mapper, group, poses = fixture()
        group.anchored = False
        mapper.update({1: group}, poses, 0)
        self.assertIsNone(mapper.snapshot(1))
        group.anchored = True
        poses[2] = SimpleNamespace(group_id=2)
        mapper.update({1: group, 2: MapGroup(2)}, poses, 1)
        self.assertIsNone(mapper.snapshot(1))
        poses[2].group_id = 1
        mapper.update({1: group}, poses, 2)
        self.assertEqual(mapper.snapshot(1)["updated_at"], 2)
        mapper.update({1: group}, poses, 3.9)
        self.assertEqual(mapper.snapshot(1)["updated_at"], 2)
        group.biomes[(0, 0)].position[1] += 1
        mapper.update({1: group}, poses, 4)
        self.assertEqual(mapper.snapshot(1)["updated_at"], 4)
        # A birth retains the old shared map but cannot trigger an unshared fit.
        poses[3] = SimpleNamespace(group_id=3)
        mapper.update({1: group, 3: MapGroup(3)}, poses, 6)
        self.assertEqual(mapper.snapshot(1)["updated_at"], 4)

    def test_frame_change_and_reset_invalidate_estimates(self):
        mapper, group, poses = fixture()
        mapper.update({1: group}, poses, 10)
        group.frame_revision += 1
        group.anchored = False
        mapper.update({1: group}, poses, 10.1)
        self.assertIsNone(mapper.snapshot(1))
        group.anchored = True
        mapper.update({1: group}, poses, 10.2)
        self.assertEqual(mapper.snapshot(1)["updated_at"], 10.2)
        mapper.update({1: group}, poses, 0)
        self.assertEqual(mapper.snapshot(1)["updated_at"], 0)
        mapper.reset()
        self.assertIsNone(mapper.snapshot(1))

    def test_rivers_are_local_overlays_not_land_generators(self):
        mapper, group, poses = fixture(grid_cell_size=20)
        for index, y in enumerate(range(100, 901, 40)):
            group.biomes[(index, 1)] = BiomeSample(np.array([500., y]), "river", 0, 1)
        mapper.update({1: group}, poses, 0)
        layer = mapper.snapshot(1)
        self.assertNotIn("river", [site["biome"] for site in layer["sites"]])
        labels = np.array(layer["labels"])
        self.assertIn(BIOMES.index("river"), labels)
        self.assertTrue(any(border["kind"] == "river" for border in layer["borders"]))
        self.assertTrue(np.all(labels[:, :10] != BIOMES.index("river")))

    def test_sparse_or_uncertain_evidence_does_not_fill_unknown_world(self):
        mapper, group, poses = fixture(support_distance=60, max_sample_uncertainty=2)
        group.biomes = {(i, 0): BiomeSample(np.array([100. + i, 100.]), "forest", 0, 1) for i in range(12)}
        group.biomes[(99, 0)] = BiomeSample(np.array([900., 900.]), "desert", 0, 30)
        group.biomes[(100, 0)] = BiomeSample(np.array([np.nan, 100.]), "desert", 0, 0)
        mapper.update({1: group}, poses, 0)
        layer = mapper.snapshot(1)
        self.assertEqual(layer["sample_count"], 12)
        self.assertEqual(layer["labels"][-1][-1], -1)
        self.assertEqual(layer["confidence"][-1][-1], 0)
        self.assertEqual(layer["borders"], [])

    def test_partial_bounds_grid_budget_and_rare_biomes(self):
        mapper, group, poses = fixture(max_samples=12, max_grid_cells=120)
        group.world_size = None
        group.known_width = 1000
        group.biomes = {(i, 0): BiomeSample(np.array([50. + i, 100.]), "forest", 0, 1) for i in range(800)}
        for index, label in enumerate(BIOMES[1:]):
            group.biomes[(index, 1)] = BiomeSample(np.array([150. + index * 50, 200.]), label, 0, 1)
        mapper.update({1: group}, poses, 0)
        layer = mapper.snapshot(1)
        self.assertFalse(layer["bounds_complete"])
        self.assertEqual(layer["bounds"][2], 1000)
        self.assertEqual(layer["bounds"][3], 440)
        self.assertLessEqual(layer["sample_count"], 12)
        self.assertEqual({s["biome"] for s in layer["sites"]}, set(BIOMES) - {"river"})
        self.assertLessEqual(np.array(layer["labels"]).size, 120)
        json.dumps(layer, allow_nan=False)
        layer["labels"][0][0] = 99
        self.assertNotEqual(mapper.snapshot(1)["labels"][0][0], 99)

    def test_all_river_samples_and_disabled_feature(self):
        mapper, group, poses = fixture()
        for sample in group.biomes.values():
            sample.biome = "river"
        mapper.update({1: group}, poses, 0)
        layer = mapper.snapshot(1)
        self.assertEqual(layer["sites"], [])
        self.assertIsNone(layer["land_sample_agreement"])
        self.assertTrue(any(BIOMES.index("river") in row for row in layer["labels"]))
        disabled = BiomeEstimator(BiomeInferenceConfig())
        disabled.update({1: group}, poses, 0)
        self.assertIsNone(disabled.snapshot(1))


if __name__ == "__main__":
    unittest.main()
