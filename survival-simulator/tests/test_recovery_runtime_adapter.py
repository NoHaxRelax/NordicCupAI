import unittest

from scripts.research_recovery_runtime import fallback_paths, fleet_budget


class FallbackPathsTests(unittest.TestCase):
    def test_unused_orchard_parameter_cannot_prevent_batch_start(self):
        inventory = [dict(path='guide.safe_distance', tunable=True),
                     dict(path='trapping_orchard.hungry', tunable=False),
                     dict(path='features.stalled_bait', tunable=True)]
        self.assertEqual(fallback_paths(
            ['trapping_orchard.hungry', 'guide.safe_distance', 'features.stalled_bait', 'missing'],
            inventory, 16), ['guide.safe_distance'])

    def test_order_and_dimension_limit_are_preserved(self):
        inventory = [dict(path=p, tunable=True) for p in ('a', 'b', 'c')]
        self.assertEqual(fallback_paths(['c', 'c', 'a', 'b'], inventory, 2), ['c', 'a'])

    def test_no_valid_dimensions_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'No valid fallback'):
            fallback_paths(['unused'], [dict(path='unused', tunable=False)], 16)

    def test_four_pods_reserve_fits_existing_budget(self):
        base={'budget':dict(total_usd=40,reserve_usd=5,external_spend_usd=10,max_hours=24,hourly_rate_usd=.985)}
        updated=fleet_budget(base,dict(total_usd=40,reserve_usd=5,auxiliary_allowance_usd=8))
        b=updated['budget']
        self.assertAlmostEqual(b['external_spend_usd']+b['reserve_usd']+b['max_hours']*b['hourly_rate_usd'],39.99125)
        self.assertEqual(base['budget']['max_hours'],24)

    def test_fleet_cannot_expand_authorized_budget(self):
        with self.assertRaisesRegex(ValueError,'authorization'):
            fleet_budget({},dict(total_usd=50,reserve_usd=5,auxiliary_allowance_usd=8))


if __name__ == '__main__':
    unittest.main()
