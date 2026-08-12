import unittest

from temps import c_to_f, f_to_c, kelvin_to_c


class TestConversions(unittest.TestCase):
    def test_c_to_f(self):
        self.assertEqual(c_to_f(100), 212)

    def test_f_to_c(self):
        self.assertEqual(f_to_c(212), 100)

    def test_kelvin_to_c(self):
        self.assertAlmostEqual(kelvin_to_c(373.15), 100.0, places=2)


if __name__ == "__main__":
    unittest.main()
