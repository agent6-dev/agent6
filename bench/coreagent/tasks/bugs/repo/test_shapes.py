"""Starter tests for shapes (runs via ./verify.sh). See spec.md."""

from __future__ import annotations

import unittest

import shapes as s


class TestClamp(unittest.TestCase):
    def test_in_range(self) -> None:
        self.assertEqual(s.clamp(5, 0, 10), 5)

    def test_above(self) -> None:
        self.assertEqual(s.clamp(15, 0, 10), 10)

    def test_below(self) -> None:
        self.assertEqual(s.clamp(-5, 0, 10), 0)

    def test_bounds_and_negative_range(self) -> None:
        self.assertEqual(s.clamp(0, 0, 10), 0)
        self.assertEqual(s.clamp(10, 0, 10), 10)
        self.assertEqual(s.clamp(-20, -10, -1), -10)


class TestMean(unittest.TestCase):
    def test_basic(self) -> None:
        self.assertEqual(s.mean([1.0, 2.0]), 1.5)

    def test_non_integer(self) -> None:
        self.assertEqual(s.mean([10.0, 5.0]), 7.5)

    def test_empty(self) -> None:
        self.assertEqual(s.mean([]), 0.0)

    def test_single(self) -> None:
        self.assertEqual(s.mean([7.0]), 7.0)


class TestMedian(unittest.TestCase):
    def test_odd_unsorted(self) -> None:
        self.assertEqual(s.median([3.0, 1.0, 2.0]), 2.0)

    def test_even_unsorted(self) -> None:
        self.assertEqual(s.median([4.0, 1.0, 2.0, 3.0]), 2.5)

    def test_empty(self) -> None:
        self.assertEqual(s.median([]), 0.0)

    def test_single(self) -> None:
        self.assertEqual(s.median([5.0]), 5.0)


class TestGcd(unittest.TestCase):
    def test_basic(self) -> None:
        self.assertEqual(s.gcd(12, 8), 4)

    def test_negative(self) -> None:
        self.assertEqual(s.gcd(12, -8), 4)
        self.assertEqual(s.gcd(-12, -8), 4)

    def test_zero(self) -> None:
        self.assertEqual(s.gcd(0, 5), 5)
        self.assertEqual(s.gcd(0, 0), 0)


class TestIsPrime(unittest.TestCase):
    def test_small(self) -> None:
        self.assertTrue(s.is_prime(2))
        self.assertTrue(s.is_prime(3))
        self.assertFalse(s.is_prime(1))

    def test_composite_squares(self) -> None:
        self.assertFalse(s.is_prime(4))
        self.assertFalse(s.is_prime(9))
        self.assertFalse(s.is_prime(25))

    def test_prime_and_composite(self) -> None:
        self.assertTrue(s.is_prime(13))
        self.assertFalse(s.is_prime(15))


class TestRoman(unittest.TestCase):
    def test_subtractive(self) -> None:
        self.assertEqual(s.roman(4), "IV")
        self.assertEqual(s.roman(9), "IX")
        self.assertEqual(s.roman(400), "CD")
        self.assertEqual(s.roman(900), "CM")

    def test_compound(self) -> None:
        self.assertEqual(s.roman(444), "CDXLIV")
        self.assertEqual(s.roman(1994), "MCMXCIV")

    def test_simple(self) -> None:
        self.assertEqual(s.roman(1), "I")
        self.assertEqual(s.roman(2024), "MMXXIV")


class TestRunningMax(unittest.TestCase):
    def test_basic(self) -> None:
        self.assertEqual(s.running_max([1, 3, 2]), [1, 3, 3])

    def test_negative(self) -> None:
        self.assertEqual(s.running_max([-3, -1, -2]), [-3, -1, -1])

    def test_empty(self) -> None:
        self.assertEqual(s.running_max([]), [])

    def test_single_negative(self) -> None:
        self.assertEqual(s.running_max([-5]), [-5])


if __name__ == "__main__":
    unittest.main()
