import unittest
from guide_multi import replacement_side


class ReplacementTests(unittest.TestCase):
    def test_distinguishes_front_from_replacement_entrance(self):
        site=dict(mouth=(0.,0.),inward=(0.,1.),overlap=20.,goal=(0.,5.),replacement_entry=(0.,40.))
        self.assertFalse(replacement_side((0.,-20.),site))
        self.assertTrue(replacement_side((0.,35.),site))
        self.assertFalse(replacement_side((0.,150.),site))
        self.assertFalse(replacement_side((150.,35.),site))

    def test_rotated_gap(self):
        site=dict(mouth=(0.,0.),inward=(-1.,0.),overlap=20.,goal=(-5.,0.),replacement_entry=(-40.,0.))
        self.assertFalse(replacement_side((20.,0.),site))
        self.assertTrue(replacement_side((-35.,0.),site))


if __name__=='__main__':
    unittest.main()
